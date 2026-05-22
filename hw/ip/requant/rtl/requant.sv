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
// Pure feed-forward pipeline; total latency is 4 cycles:
//   stage 1 register — i32_to_fp16 output
//   stage 2 register — scale-bump output (and scale_fp16 / bias pipelined here)
//   stage 3 register — fp16_fma output
//   stage 4 register — fp16_to_i8_sat output
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

  // ─── Stage 1: int32 → fp16 with prescale ─────────────────
  logic [15:0] s1_fp16;
  logic [4:0]  s1_shift;

  i32_to_fp16 u_i2f (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (acc_i),
    .y_o    (s1_fp16),
    .shift_o(s1_shift)
  );

  // scale and bias are pipelined alongside the data so the FMA sees the
  // matching per-channel coefficients on the same cycle as s1_fp16.
  logic [15:0] s1_scale_q, s1_bias_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s1_scale_q <= 16'h0000;
      s1_bias_q  <= 16'h0000;
    end else begin
      s1_scale_q <= scale_fp16_i;
      s1_bias_q  <= bias_fp16_i;
    end
  end

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

  // ─── Valid shift register (4-deep, matching the 4-cycle pipeline) ──
  logic [3:0] valid_sr;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) valid_sr <= 4'b0000;
    else         valid_sr <= {valid_sr[2:0], valid_i};
  end
  assign valid_o = valid_sr[3];

endmodule
