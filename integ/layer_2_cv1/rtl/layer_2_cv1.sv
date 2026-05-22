// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_2_cv1 — YOLO26n /model.2/cv1 Conv-BN-SiLU (1x1, stride 1, 32→32,
// 160×160). Entry point of the first C3k2 block.
//
// Parallelism (scale_pkg::LAYER_2_*):
//   P_PIX  = 1
//   P_COUT = 32   (full output-channel parallelism)
//   P_CIN  = 16   (half input-channel parallelism)
//   ⇒ 32 × dotN(N=16) = 512 MAC units.
//
// Each output pixel needs the full CIN=32 input-channel reduction; with
// P_CIN=16 we stream the pixel as 2 consecutive half-windows (phases). A
// small accumulator after each dotN sums the two phase outputs, yielding
// 1 output pixel every 2 cycles. The 160×160 frame finishes in
// 2 × 160 × 160 = 51_200 cycles (matches LAYER_2_CYCLES).
//
// Pipeline (post dotN):
//   dotN(N=16)      : 1 + clog2(16) = 5 cyc
//   2-phase accum   : 1 cyc (latches phase-0 dot, adds phase-1 next cyc)
//   requant         : 3 cyc
//   act_silu        : 1 cyc
//   Total latency from first phase-0 valid_i to valid_o = 10 cyc.
//
// All IPs reused from hw/ip/* unverbatim (no edits there):
//   - dotN, requant (fp16 fma + sat), act_silu.
//   - linebuf_kxk is unused (K=1 ⇒ window == single pixel, no spatial reuse).
//
// Ports:
//   valid_i  : asserted on every phase cycle (2 per output pixel).
//   phase_i  : 0 = first half (lanes 0..15), 1 = second half (16..31).
//              Output is committed on the cycle following phase_i==1.
//   x_i      : N_LANE lanes of int8 (kc within current phase).
//   w_i      : NCH_OUT × N_LANE i8 (weights for the current phase).
//   scale_i  : NCH_OUT fp16, sampled when phase_i==1.
//   bias_i   : NCH_OUT fp16, sampled when phase_i==1.
//   y_o,
//   valid_o  : post-SiLU int8 with aligned valid.

module layer_2_cv1
  import scale_pkg::*;
#(
  parameter int  NCH_OUT = LAYER_2_COUT,                          // = 32
  parameter int  NCH_IN  = LAYER_2_CIN,                           // = 32
  parameter int  N_LANE  = LAYER_2_P_CIN,                         // = 16
  // verilator lint_off UNUSEDPARAM
  parameter int  N_PHASE = NCH_IN / N_LANE,                       // = 2
  // verilator lint_on UNUSEDPARAM
  parameter real S_OUT_PRE  = 80.0 / 127.0,
  parameter real S_OUT_SILU = 80.0 / 127.0
) (
  input  logic                                          clk_i,
  input  logic                                          rst_ni,

  input  logic                                          valid_i,
  input  logic                                          phase_i,  // 0..N_PHASE-1, here 1-bit
  input  logic signed [N_LANE-1:0][7:0]                 x_i,
  input  logic signed [NCH_OUT-1:0][N_LANE-1:0][7:0]    w_i,
  input  logic        [NCH_OUT-1:0][15:0]               scale_i,
  input  logic        [NCH_OUT-1:0][15:0]               bias_i,

  output logic                                          valid_o,
  output logic signed [NCH_OUT-1:0][7:0]                y_o
);

  // ── Stage A: NCH_OUT parallel dotN(N=N_LANE) ──
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

  // dotN latency = 1 + clog2(N_LANE). For N_LANE=16, this is 5.
  localparam int DOT_LAT = 1 + $clog2(N_LANE);

  // Pipeline phase, scale, bias to align with dotN output.
  logic                     phase_sr [DOT_LAT];
  logic [NCH_OUT-1:0][15:0] scale_sr [DOT_LAT];
  logic [NCH_OUT-1:0][15:0] bias_sr  [DOT_LAT];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < DOT_LAT; i++) begin
        phase_sr[i] <= 1'b0;
        scale_sr[i] <= '0;
        bias_sr[i]  <= '0;
      end
    end else begin
      phase_sr[0] <= phase_i;
      scale_sr[0] <= scale_i;
      bias_sr[0]  <= bias_i;
      for (int i = 1; i < DOT_LAT; i++) begin
        phase_sr[i] <= phase_sr[i-1];
        scale_sr[i] <= scale_sr[i-1];
        bias_sr[i]  <= bias_sr[i-1];
      end
    end
  end

  // ── Stage B: per-channel 2-phase accumulator ──
  // On phase_aligned==0: latch dot_acc as partial.
  // On phase_aligned==1: produce sum (partial + dot_acc) with valid=1.
  logic                              dv_aligned;
  logic                              ph_aligned;
  assign dv_aligned = dot_valid[0];   // all NCH_OUT dotNs are in lockstep
  assign ph_aligned = phase_sr[DOT_LAT-1];

  // Tie off the unused upper dot_valid bits (lint).
  // verilator lint_off UNUSEDSIGNAL
  logic [NCH_OUT-2:0] unused_dot_valid;
  // verilator lint_on UNUSEDSIGNAL
  assign unused_dot_valid = dot_valid[NCH_OUT-1:1];

  logic signed [NCH_OUT-1:0][31:0] partial_q;
  logic signed [NCH_OUT-1:0][31:0] sum_q;
  logic                            sum_valid_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      partial_q   <= '0;
      sum_q       <= '0;
      sum_valid_q <= 1'b0;
    end else begin
      sum_valid_q <= 1'b0;
      if (dv_aligned) begin
        if (ph_aligned == 1'b0) begin
          // Latch phase-0 partial result.
          partial_q <= dot_acc;
        end else begin
          // Combine: sum = partial + phase-1 dot.
          for (int c = 0; c < NCH_OUT; c++) begin
            sum_q[c] <= partial_q[c] + dot_acc[c];
          end
          sum_valid_q <= 1'b1;
        end
      end
    end
  end

  // Pipeline scale/bias one more cycle to align with sum_q (the accumulator
  // latches scale_sr[DOT_LAT-1] on the phase-1 cycle and exposes it next).
  logic [NCH_OUT-1:0][15:0] scale_at_sum_q;
  logic [NCH_OUT-1:0][15:0] bias_at_sum_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      scale_at_sum_q <= '0;
      bias_at_sum_q  <= '0;
    end else if (dv_aligned && ph_aligned == 1'b1) begin
      scale_at_sum_q <= scale_sr[DOT_LAT-1];
      bias_at_sum_q  <= bias_sr [DOT_LAT-1];
    end
  end

  // ── Stage C: NCH_OUT parallel requant ──
  logic [NCH_OUT-1:0]              rq_valid;
  logic signed [NCH_OUT-1:0][7:0]  rq_y;

  generate
    for (gc = 0; gc < NCH_OUT; gc++) begin : g_rq
      requant u_rq (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .valid_i      (sum_valid_q),
        .acc_i        (sum_q[gc]),
        .scale_fp16_i (scale_at_sum_q[gc]),
        .bias_fp16_i  (bias_at_sum_q [gc]),
        .valid_o      (rq_valid[gc]),
        .y_o          (rq_y[gc])
      );
    end
  endgenerate

  // ── Stage D: NCH_OUT parallel act_silu ──
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

  // verilator lint_off UNUSEDSIGNAL
  logic [NCH_OUT-2:0] unused_rq_valid;
  // verilator lint_on UNUSEDSIGNAL
  assign unused_rq_valid = rq_valid[NCH_OUT-1:1];

  logic silu_valid_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) silu_valid_q <= 1'b0;
    else         silu_valid_q <= rq_valid[0];
  end

  assign valid_o = silu_valid_q;
  assign y_o     = silu_y;

endmodule
