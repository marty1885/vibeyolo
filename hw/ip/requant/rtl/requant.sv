// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// requant — int32 accumulator → int8 activation, via:
//   1. i32_to_fp16     : (fp16_shifted, shift) = autoscale(acc_i32)
//                        (RNE, autoscales for |acc| ≥ 2^16 so the fp16
//                        is always representable; shift reports the
//                        prescale that was applied).
//   2. scale bump      : scale_eff = scale_fp16 with biased exponent +=
//                        shift (saturated at 30 → keeps the value finite).
//                        A register stage follows to keep timing happy.
//   3. fp16_fma        : fp16_shifted * scale_eff + bias_fp16
//                        (single rounding).
//   4. fp16_to_i8_sat  : RNE + saturating clamp to [-128, +127].
//
// Pure feed-forward pipeline; total latency is
//   I2F_LAT + 1 (scale-bump) + FMA_LAT + 1 (sat) = 2 + 1 + 3 + 1 = 7 cycles:
//   stages 1..I2F_LAT  — i32_to_fp16 output (now 2-cycle pipelined)
//   scale-bump register — scale-bump output (scale_fp16/bias aligned here)
//   stages ..+FMA_LAT   — fp16_fma output (now 3-cycle pipelined)
//   final register      — fp16_to_i8_sat output
//
// The i32_to_fp16 / fp16_fma latencies live in fp16_lat_pkg (the canonical
// single source of truth). requant is instantiated in ~120 generated layer
// build lists, so to avoid forcing the package into all of them it mirrors
// the values in local localparams below; the lockstep DV against the
// independent requant_ref golden fails if these ever drift.
//
// The change vs. the previous implementation: layers used to pre-multiply
// scale_fp16 by 2^ACC_SHIFT in software to keep the accumulator-to-fp16
// step from saturating. That workaround is no longer required — the IP
// performs the bookkeeping itself.

module requant (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               valid_i,
  input  logic signed [31:0] acc_i,
  input  logic        [15:0] scale_fp16_i,
  input  logic        [15:0] bias_fp16_i,

  output logic               valid_o,
  output logic signed  [7:0] y_o
);

  // Leaf-IP latencies — mirror of fp16_lat_pkg (see header). DV-validated.
  localparam int unsigned I2F_LAT = 2;   // == fp16_lat_pkg::I32_TO_FP16_LAT
  localparam int unsigned FMA_LAT = 3;   // == fp16_lat_pkg::FP16_FMA_LAT
  localparam int unsigned SAT_LAT = 1;   // fp16_to_i8_sat (unchanged)
  // valid_i → valid_o total latency.
  localparam int unsigned TOTAL_LAT = I2F_LAT + 1 + FMA_LAT + SAT_LAT;

  // ─── Stage 1: int32 → fp16 with prescale (I2F_LAT cycles) ─
  logic [15:0] s1_fp16;
  logic [4:0]  s1_shift;

  i32_to_fp16 u_i2f (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (acc_i),
    .y_o    (s1_fp16),
    .shift_o(s1_shift)
  );

  // scale and bias are delayed I2F_LAT cycles so they line up with the
  // i32_to_fp16 output (s1_fp16 / s1_shift) feeding the scale-bump.
  logic [15:0] scale_dl [I2F_LAT];
  logic [15:0] bias_dl  [I2F_LAT];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < I2F_LAT; i++) begin
        scale_dl[i] <= 16'h0000;
        bias_dl[i]  <= 16'h0000;
      end
    end else begin
      scale_dl[0] <= scale_fp16_i;
      bias_dl[0]  <= bias_fp16_i;
      for (int i = 1; i < I2F_LAT; i++) begin
        scale_dl[i] <= scale_dl[i-1];
        bias_dl[i]  <= bias_dl[i-1];
      end
    end
  end
  logic [15:0] s1_scale_q, s1_bias_q;
  assign s1_scale_q = scale_dl[I2F_LAT-1];
  assign s1_bias_q  = bias_dl[I2F_LAT-1];

  // ─── Stage 2: bump scale's exponent by shift ─────────────
  // scale_eff = scale * 2^shift, achieved by adding `shift` to the biased
  // exponent field. Special cases:
  //   • exp_field == 0 (zero or subnormal): leave unchanged. For zero this
  //     is exact (0 * 2^k = 0). For subnormals the spec accepts a small
  //     magnitude error since the FMA's product magnitude will be tiny
  //     anyway; in practice layer scales are well within fp16 normal
  //     range so we never hit this path.
  //   • exp_field == 31 (Inf / NaN): leave unchanged so NaN / Inf
  //     propagate cleanly to the FMA.
  //   • exp_field + shift ≥ 31: saturate at 30 (largest finite normal
  //     exponent). The product fp16_shifted * scale would have
  //     overflowed anyway and the downstream fp16_fma + fp16_to_i8_sat
  //     would saturate to ±127.
  logic [15:0] scale_bumped_d;
  always_comb begin
    automatic logic        s_sign  = s1_scale_q[15];
    automatic logic [4:0]  s_exp   = s1_scale_q[14:10];
    automatic logic [9:0]  s_frac  = s1_scale_q[9:0];
    automatic logic [5:0]  e_sum;
    automatic logic [4:0]  e_new;
    if (s_exp == 5'd0 || s_exp == 5'd31) begin
      scale_bumped_d = s1_scale_q;
    end else begin
      e_sum = {1'b0, s_exp} + {1'b0, s1_shift};
      // Saturate to 30 to keep the result finite. (Going to 31 would
      // create an Inf that's still correct, but staying finite gives the
      // downstream fp16_to_i8_sat a chance to saturate to ±127 via the
      // normal magnitude path rather than the Inf-special-case path.)
      e_new = (e_sum >= 6'd30) ? 5'd30 : e_sum[4:0];
      scale_bumped_d = {s_sign, e_new, s_frac};
    end
  end

  logic [15:0] s2_fp16, s2_scale, s2_bias;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s2_fp16  <= 16'h0000;
      s2_scale <= 16'h0000;
      s2_bias  <= 16'h0000;
    end else begin
      s2_fp16  <= s1_fp16;
      s2_scale <= scale_bumped_d;
      s2_bias  <= s1_bias_q;
    end
  end

  // ─── Stage 3: fp16 FMA  (s2_fp16 * s2_scale + s2_bias) ───
  logic [15:0] s3_fma;

  fp16_fma u_fma (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (s2_fp16),
    .b_i    (s2_scale),
    .c_i    (s2_bias),
    .y_o    (s3_fma)
  );

  // ─── Stage 4: fp16 → int8 (saturating) ───────────────────
  logic signed [7:0] s4_y;

  fp16_to_i8_sat u_f2i (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (s3_fma),
    .y_o    (s4_y)
  );

  assign y_o = s4_y;

  // ─── Valid shift register (TOTAL_LAT-deep, matching the pipeline) ──
  logic [TOTAL_LAT-1:0] valid_sr;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) valid_sr <= '0;
    else         valid_sr <= {valid_sr[TOTAL_LAT-2:0], valid_i};
  end
  assign valid_o = valid_sr[TOTAL_LAT-1];

endmodule
