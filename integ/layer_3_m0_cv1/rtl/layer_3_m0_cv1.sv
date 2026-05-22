// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_3_m0_cv1 — YOLO26n /model.2/m.0/cv1 Conv-BN-SiLU (3x3, stride 1,
// 16→8). First conv inside the C3k2 bottleneck of /model.2. Input is the
// second half of /model.2/cv1's 32-channel output (Split), 160x160.
//
// Composition (identical pattern to integ/layer_1):
//   • NCH_OUT × dotN(N=144)      : signed i8·i8 → i32  (1+clog2(144)=9 cyc)
//   • NCH_OUT × requant          : i32 → fp16·scale + bias → i8  (3 cyc)
//   • NCH_OUT × act_silu         : i8 LUT → i8 SiLU              (1 cyc)
// Total pipeline latency = 13 cycles, throughput = 1 output pixel / cycle
// (all 8 output channels emitted in parallel).
//
// Parallelism rationale: scale_pkg::LAYER_3 has P_PIX=1, P_COUT=8, P_CIN=48,
// product = 384 → 25 600 pix × 144/48 = 76 800 cyc/frame (matches LAYER_3_CYCLES).
//
// The unit-level testbench drives windows from C++; the linebuf_kxk wiring
// is the integrator's job (one per input channel, as noted for layer_1).
//
// Ports mirror integ/layer_1:
//   x_i      : N_LANE lanes of int8 (3x3x16), lane = (kh*K + kw)*Cin + kc
//   w_i      : NCH_OUT × N_LANE int8 weights
//   scale_i  : NCH_OUT fp16 (per-output-channel)
//   bias_i   : NCH_OUT fp16 (per-output-channel)

module layer_3_m0_cv1
  import scale_pkg::*;
#(
  parameter int  NCH_OUT = LAYER_3_COUT,                          // = 8
  parameter int  N_LANE  = LAYER_3_CIN * LAYER_3_K * LAYER_3_K,   // = 16*3*3 = 144
  parameter real S_OUT_PRE  = 2.0 / 127.0,
  parameter real S_OUT_SILU = 2.0 / 127.0
) (
  input  logic                                          clk_i,
  input  logic                                          rst_ni,

  input  logic                                          valid_i,
  input  logic signed [N_LANE-1:0][7:0]                 x_i,
  input  logic signed [NCH_OUT-1:0][N_LANE-1:0][7:0]    w_i,
  input  logic        [NCH_OUT-1:0][15:0]               scale_i,
  input  logic        [NCH_OUT-1:0][15:0]               bias_i,

  output logic                                          valid_o,
  output logic signed [NCH_OUT-1:0][7:0]                y_o
);

  // ── Stage A: parallel dotN ──
  logic [NCH_OUT-1:0]              dot_valid;
  logic signed [NCH_OUT-1:0][31:0] dot_acc;

  genvar gc;
  generate
    for (gc = 0; gc < NCH_OUT; gc++) begin : g_dot
      dotN #(.N(N_LANE)) u_dot (
        .clk_i  (clk_i),
        .rst_ni (rst_ni),
        .en_i   (valid_i),
        .clr_i  (1'b0),
        .a_i    (x_i),
        .b_i    (w_i[gc]),
        .y_o    (dot_acc[gc]),
        .valid_o(dot_valid[gc])
      );
    end
  endgenerate

  // dotN latency = 1 + clog2(N_LANE); N_LANE=144 → 9.
  localparam int DOT_LAT = 1 + $clog2(N_LANE);

  logic [NCH_OUT-1:0][15:0] scale_sr [DOT_LAT];
  logic [NCH_OUT-1:0][15:0] bias_sr  [DOT_LAT];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < DOT_LAT; i++) begin
        scale_sr[i] <= '0;
        bias_sr[i]  <= '0;
      end
    end else begin
      scale_sr[0] <= scale_i;
      bias_sr[0]  <= bias_i;
      for (int i = 1; i < DOT_LAT; i++) begin
        scale_sr[i] <= scale_sr[i-1];
        bias_sr[i]  <= bias_sr[i-1];
      end
    end
  end

  // ── Stage B: parallel requant ──
  logic [NCH_OUT-1:0]              rq_valid;
  logic signed [NCH_OUT-1:0][7:0]  rq_y;

  generate
    for (gc = 0; gc < NCH_OUT; gc++) begin : g_rq
      requant u_rq (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .valid_i      (dot_valid[gc]),
        .acc_i        (dot_acc[gc]),
        .scale_fp16_i (scale_sr[DOT_LAT-1][gc]),
        .bias_fp16_i  (bias_sr [DOT_LAT-1][gc]),
        .valid_o      (rq_valid[gc]),
        .y_o          (rq_y[gc])
      );
    end
  endgenerate

  // ── Stage C: parallel act_silu ──
  logic signed [NCH_OUT-1:0][7:0] silu_y;
  generate
    for (gc = 0; gc < NCH_OUT; gc++) begin : g_silu
      act_silu #(.InScale(S_OUT_PRE), .OutScale(S_OUT_SILU)) u_silu (
        .clk_i (clk_i),
        .rst_ni(rst_ni),
        .x_i   (rq_y[gc]),
        .y_o   (silu_y[gc])
      );
    end
  endgenerate

  logic silu_valid_q;
  // verilator lint_off UNUSEDSIGNAL
  logic rq_valid_lane0;
  // verilator lint_on UNUSEDSIGNAL
  assign rq_valid_lane0 = rq_valid[0];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) silu_valid_q <= 1'b0;
    else         silu_valid_q <= rq_valid_lane0;
  end

  // verilator lint_off UNUSEDSIGNAL
  logic [NCH_OUT-2:0] unused_rq_valid;
  // verilator lint_on UNUSEDSIGNAL
  assign unused_rq_valid = rq_valid[NCH_OUT-1:1];

  assign valid_o = silu_valid_q;
  assign y_o     = silu_y;

endmodule
