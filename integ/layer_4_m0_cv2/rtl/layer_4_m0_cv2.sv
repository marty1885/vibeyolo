// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_4_m0_cv2 — YOLO26n /model.2/m.0/cv2 Conv-BN-SiLU + bottleneck residual.
//
// Composes verified IPs only:
//   • 16 × dotN(N=72)    : signed i8·i8 → i32  (1 + clog2(72) = 8 cyc)
//   • 16 × requant       : i32 → fp16·scale + bias → i8        (3 cyc)
//   • 16 × act_silu      : i8 LUT → i8 SiLU                    (1 cyc)
//   • 16 × add_rq        : two i8 streams (cv2-post-SiLU, residual) → i8 (5 cyc)
// Total pipeline latency = 17 cycles. One 3x3x8 window per cycle → 16
// output channels in parallel (with the matching 16-channel residual
// sample presented in lockstep with valid_i; an internal delay line
// realigns it to the cv2-post-SiLU stage).
//
// Per the ONNX graph, the residual operand is the bottleneck input
// `/model.2/Slice_1_output_0` (16ch), added AFTER the cv2 SiLU. We model
// the dyn-quant of that residual via its own fp16 scale at add_rq.scale_b.
// (The folded DC term from u8→i8, s_r*(128-zp_r), is baked into the
// add_rq bias by the host TB.)
//
// Parameter defaults come from scale_pkg::LAYER_4_*. Frame-level target
// is 76,800 cycles (matches LAYER_4_CYCLES at P_PIX=1, P_COUT=16, P_CIN=24).
//
// Ports:
//   x_i       : N_LANE lanes of int8 (3*3*8 = 72), lane = (kh*K+kw)*Cin+kc
//   r_i       : NCH_OUT lanes of int8 — residual sample for this pixel
//   w_i       : NCH_OUT channels × N_LANE lanes of int8 weight
//   scale_i   : NCH_OUT fp16 (cv2 pre-SiLU per-channel scale)
//   bias_i    : NCH_OUT fp16 (cv2 pre-SiLU per-channel bias, folded)
//   add_scale_a_i      : fp16 — cv2-post-SiLU stream weight (= S_OUT_SILU)
//   add_scale_b_i      : fp16 — residual stream weight (= s_r)
//   add_inv_out_scale_i: fp16 — 1.0 / S_OUT_ADD
//   add_bias_i         : fp16 — DC fold term (s_r*(128-zp_r)/S_OUT_ADD)
//   valid_i   : input-window valid pulse
//   y_o       : NCH_OUT lanes of int8 (post-residual)
//   valid_o   : output valid (aligned with y_o)

module layer_4_m0_cv2
  import scale_pkg::*;
#(
  parameter int  NCH_OUT = LAYER_4_COUT,                          // = 16
  parameter int  N_LANE  = LAYER_4_CIN * LAYER_4_K * LAYER_4_K,   // = 8*3*3 = 72
  parameter real S_OUT_PRE  = 80.0 / 127.0,
  parameter real S_OUT_SILU = 80.0 / 127.0
) (
  input  logic                                          clk_i,
  input  logic                                          rst_ni,

  input  logic                                          valid_i,
  input  logic signed [N_LANE-1:0][7:0]                 x_i,
  input  logic signed [NCH_OUT-1:0][7:0]                r_i,
  input  logic signed [NCH_OUT-1:0][N_LANE-1:0][7:0]    w_i,
  input  logic        [NCH_OUT-1:0][15:0]               scale_i,
  input  logic        [NCH_OUT-1:0][15:0]               bias_i,

  input  logic        [15:0]                            add_scale_a_i,
  input  logic        [15:0]                            add_scale_b_i,
  input  logic        [15:0]                            add_inv_out_scale_i,
  input  logic        [15:0]                            add_bias_i,

  output logic                                          valid_o,
  output logic signed [NCH_OUT-1:0][7:0]                y_o
);

  // ── Stage A: NCH_OUT parallel dotN ──
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

  localparam int DOT_LAT     = 1 + $clog2(N_LANE);   // 8 for N_LANE=72
  localparam int REQUANT_LAT = 3;
  localparam int SILU_LAT    = 1;
  localparam int CV2_LAT     = DOT_LAT + REQUANT_LAT + SILU_LAT;  // up to add_rq.a input
  // add_rq pipeline = 5 cycles (documented in hw/ip/add_rq).

  // Pipeline scale/bias to align with dot output.
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

  // ── Stage B: NCH_OUT parallel requant ──
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

  // ── Stage C: NCH_OUT parallel act_silu ──
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

  // Track valid for the cv2 stream (lane-0 of rq_valid → +1 silu cycle).
  logic silu_valid_q;
  // verilator lint_off UNUSEDSIGNAL
  logic rq_valid_lane0;
  // verilator lint_on UNUSEDSIGNAL
  assign rq_valid_lane0 = rq_valid[0];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) silu_valid_q <= 1'b0;
    else         silu_valid_q <= rq_valid_lane0;
  end
  // tie off other lanes' rq_valid for lint
  // verilator lint_off UNUSEDSIGNAL
  logic [NCH_OUT-2:0] unused_rq_valid;
  // verilator lint_on UNUSEDSIGNAL
  assign unused_rq_valid = rq_valid[NCH_OUT-1:1];

  // ── Residual delay line: r_i arrives with valid_i; align with silu_y. ──
  // Delay r_i by CV2_LAT cycles so it appears at add_rq.b on the same cycle
  // as silu_y appears at add_rq.a.
  logic signed [NCH_OUT-1:0][7:0] r_sr [CV2_LAT];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < CV2_LAT; i++) r_sr[i] <= '0;
    end else begin
      r_sr[0] <= r_i;
      for (int i = 1; i < CV2_LAT; i++) r_sr[i] <= r_sr[i-1];
    end
  end

  // ── Stage D: NCH_OUT parallel add_rq (residual add + requant) ──
  logic [NCH_OUT-1:0]              add_valid;
  logic signed [NCH_OUT-1:0][7:0]  add_y;

  generate
    for (gc = 0; gc < NCH_OUT; gc++) begin : g_add
      add_rq u_add (
        .clk_i                (clk_i),
        .rst_ni               (rst_ni),
        .valid_i              (silu_valid_q),
        .a_i8_i               (silu_y[gc]),
        .b_i8_i               (r_sr[CV2_LAT-1][gc]),
        .scale_a_fp16_i       (add_scale_a_i),
        .scale_b_fp16_i       (add_scale_b_i),
        .inv_out_scale_fp16_i (add_inv_out_scale_i),
        .bias_fp16_i          (add_bias_i),
        .valid_o              (add_valid[gc]),
        .y_o                  (add_y[gc])
      );
    end
  endgenerate

  // verilator lint_off UNUSEDSIGNAL
  logic [NCH_OUT-2:0] unused_add_valid;
  // verilator lint_on UNUSEDSIGNAL
  assign unused_add_valid = add_valid[NCH_OUT-1:1];

  assign valid_o = add_valid[0];
  assign y_o     = add_y;

endmodule
