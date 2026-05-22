// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_5_cv2 — YOLO26n /model.2/cv2 Conv 1×1 (48 → 64) + SiLU.
//
// This is the exit conv of the first C3k2 block. K=1, stride=1, pad=0,
// so the per-pixel input is a 48-channel vector (no spatial linebuffer
// needed) and we produce 64 output channels per pixel.
//
// Composes the same verified IPs as layer_1, just with N_LANE=48:
//   • NCH_OUT × dotN(N=48)  : signed i8·i8 → i32   (1 + clog2(48) = 7 cyc)
//   • NCH_OUT × requant     : i32 → fp16·scale + bias → i8         (3 cyc)
//   • NCH_OUT × act_silu    : i8 LUT → i8 SiLU                     (1 cyc)
// Total pipeline latency = 11 cycles. Throughput = 1 output pixel/cycle.
//
// Parameter defaults come from scale_pkg::LAYER_5_*:
//   LAYER_5_CIN  = 48     LAYER_5_COUT = 64
//   LAYER_5_K    = 1      LAYER_5_STRIDE = 1   LAYER_5_PAD = 0
//   LAYER_5_P_PIX=1  P_COUT=64  P_CIN=16   ⇒  LAYER_5_CYCLES = 76800
//
// The 48-ch input is the external concat of [cv1_first_half(16),
// cv1_second_half(16), m.0_output(16)]; we just accept 48 i8 lanes/beat.

module layer_5_cv2
  import scale_pkg::*;
#(
  parameter int  NCH_OUT = LAYER_5_COUT,                          // 64
  parameter int  N_LANE  = LAYER_5_CIN * LAYER_5_K * LAYER_5_K,   // 48*1*1 = 48
  parameter real S_OUT_PRE  = 12.0 / 127.0,
  parameter real S_OUT_SILU = 12.0 / 127.0
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

  // ── Stage A: NCH_OUT parallel dotN ────────────────────────────────────
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

  // dotN latency = 1 + clog2(N_LANE). N_LANE=48, clog2=6, DOT_LAT=7.
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

  // ── Stage B: NCH_OUT parallel requant ─────────────────────────────────
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

  // ── Stage C: NCH_OUT parallel act_silu ────────────────────────────────
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
