// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_13_m0_cv1 — YOLO26n /model.6/m.0/cv1 Conv-BN-SiLU (1x1, stride 1,
// 64→32, 40×40). First conv of the bottleneck inside C3k2 block #3.
// YOLO26n's deeper stages use c3k=False bottlenecks → the conv is K=1
// (not K=3 like the shallower /model.4 stage in layer_8_m0_cv1).
//
// Parallelism (scale_pkg::LAYER_13_*):
//   P_PIX  = 1
//   P_COUT = 32    (full output-channel parallelism)
//   P_CIN  = 2     (2-lane input-channel parallelism)
//   ⇒ 32 × dotN(N=2) = 64 MAC units.
//
// Each output pixel needs the full CIN=64 reduction; with P_CIN=2 we stream
// the pixel as 32 consecutive phases. A per-channel accumulator after each
// dotN sums the 32 phase outputs, yielding 1 output pixel every 32 cycles.
// The 40×40 frame finishes in 32 × 40 × 40 = 51_200 cycles (matches
// LAYER_13_CYCLES, target T_FRAME=100_000).
//
// Pipeline (post dotN):
//   dotN(N=2)       : 1 + clog2(2) = 2 cyc
//   N-phase accum   : 1 cyc commit cycle (after phase N_PHASE-1)
//   requant         : 3 cyc
//   act_silu        : 1 cyc
//   + 1 output reg
//   Total latency from phase-(N_PHASE-1) valid_i to valid_o = 8 cyc.
//
// Per-pixel accumulator pre-scale:
//   Empirical worst-case |sum_q| across YOLO26n L13 ROIs reaches ~68k
//   (extract.py: 67759), exceeding fp16's normal max (65504). We
//   arithmetic-shift-right sum_q by ACC_SHIFT=2 before requant; extract.py
//   pre-multiplies scale_fp16 by 2^ACC_SHIFT so the math is equivalent.
//
// All IPs reused from hw/ip/* without modification:
//   - dotN, requant (fp16 fma + sat), act_silu.
//   - linebuf_kxk is unused (K=1).
//
// Ports:
//   valid_i  : asserted on every phase cycle (N_PHASE per output pixel).
//   phase_i  : 0..N_PHASE-1. Output committed cycle after phase==N_PHASE-1.
//   x_i      : N_LANE lanes of int8 (kc within current phase).
//   w_i      : NCH_OUT × N_LANE i8 (weights for the current phase).
//   scale_i  : NCH_OUT fp16, sampled when phase_i == N_PHASE-1.
//   bias_i   : NCH_OUT fp16, sampled when phase_i == N_PHASE-1.
//   y_o,
//   valid_o  : post-SiLU int8 with aligned valid.

module layer_13_m0_cv1
  import scale_pkg::*;
#(
  parameter int  NCH_OUT = LAYER_13_COUT,                         // = 32
  parameter int  NCH_IN  = LAYER_13_CIN,                          // = 64
  parameter int  N_LANE  = LAYER_13_P_CIN,                        // = 2
  // verilator lint_off UNUSEDPARAM
  parameter int  N_PHASE = NCH_IN / N_LANE,                       // = 32
  // verilator lint_on UNUSEDPARAM
  parameter int  PHASE_BITS = (N_PHASE > 1) ? $clog2(N_PHASE) : 1,
  parameter real S_OUT_PRE  = 4.0 / 127.0,
  parameter real S_OUT_SILU = 4.0 / 127.0
) (
  input  logic                                          clk_i,
  input  logic                                          rst_ni,

  input  logic                                          valid_i,
  input  logic        [PHASE_BITS-1:0]                  phase_i,
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

  // dotN latency = 1 + clog2(N_LANE). For N_LANE=2, this is 2.
  localparam int DOT_LAT = 1 + $clog2(N_LANE);

  // Pipeline phase, scale, bias to align with dotN output.
  logic [PHASE_BITS-1:0]    phase_sr [DOT_LAT];
  logic [NCH_OUT-1:0][15:0] scale_sr [DOT_LAT];
  logic [NCH_OUT-1:0][15:0] bias_sr  [DOT_LAT];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < DOT_LAT; i++) begin
        phase_sr[i] <= '0;
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

  // ── Stage B: per-channel N-phase accumulator ──
  // For phase_aligned == 0     : load partial = dot_acc.
  // For phase_aligned 1..N-2   : partial += dot_acc.
  // For phase_aligned == N-1   : sum = partial + dot_acc, valid=1.
  logic                              dv_aligned;
  logic [PHASE_BITS-1:0]             ph_aligned;
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
        if (ph_aligned == '0) begin
          // Start a new pixel accumulation.
          partial_q <= dot_acc;
        end else if (ph_aligned == PHASE_BITS'(N_PHASE-1)) begin
          // Final phase: commit sum.
          for (int c = 0; c < NCH_OUT; c++) begin
            sum_q[c] <= partial_q[c] + dot_acc[c];
          end
          sum_valid_q <= 1'b1;
        end else begin
          // Intermediate phases: accumulate.
          for (int c = 0; c < NCH_OUT; c++) begin
            partial_q[c] <= partial_q[c] + dot_acc[c];
          end
        end
      end
    end
  end

  // Pipeline scale/bias to align with sum_q.
  logic [NCH_OUT-1:0][15:0] scale_at_sum_q;
  logic [NCH_OUT-1:0][15:0] bias_at_sum_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      scale_at_sum_q <= '0;
      bias_at_sum_q  <= '0;
    end else if (dv_aligned && ph_aligned == PHASE_BITS'(N_PHASE-1)) begin
      scale_at_sum_q <= scale_sr[DOT_LAT-1];
      bias_at_sum_q  <= bias_sr [DOT_LAT-1];
    end
  end

  // ── Accumulator pre-scale (see header for rationale) ──
  // Index a packed signed slice via $signed() to ensure arithmetic shift.
  localparam int ACC_SHIFT = 2;
  logic signed [NCH_OUT-1:0][31:0] sum_q_scaled;
  generate
    for (gc = 0; gc < NCH_OUT; gc++) begin : g_shift
      assign sum_q_scaled[gc] = $signed(sum_q[gc]) >>> ACC_SHIFT;
    end
  endgenerate

  // ── Stage C: NCH_OUT parallel requant ──
  logic [NCH_OUT-1:0]              rq_valid;
  logic signed [NCH_OUT-1:0][7:0]  rq_y;

  generate
    for (gc = 0; gc < NCH_OUT; gc++) begin : g_rq
      requant u_rq (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .valid_i      (sum_valid_q),
        .acc_i        (sum_q_scaled[gc]),
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
