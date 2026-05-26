// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// fp16_fma — IEEE-754 binary16 fused multiply-add: y = a*b + c.
//
// Single rounding step at the end (RNE). Subnormal inputs and outputs are
// supported. Underflow flushes to zero (documented in README). Overflow ->
// signed Inf. NaN propagates (canonical 0x7E00). Inf*0 -> NaN.
// Inf+(-Inf) -> NaN. a*b == -0 and adding +0 -> +0 (default RNE rule).
//
// Strategy:
//   1. Unpack a, b, c into (sign, 11-bit significand, signed effective
//      exponent). value = sig11 * 2^eff_exp.
//        normal:    sig11 = {1, frac10}, eff_exp = bexp - 25.
//        subnormal: sig11 = {0, frac10}, eff_exp = -24.
//        zero:      sig11 = 0 (handled by special-case path)
//   2. Compute product sig_p = sig_a * sig_b (22 bits), e_p = e_a + e_b.
//   3. Place product and c into a 100-bit shifted view. Reading off
//      the top 50 bits is the accumulator window; bits below are sticky.
//      Anchor exponent set so dominant operand's high bits land near
//      the top of the window.
//   4. Same-sign add or different-sign subtract on the 50-bit accumulators
//      with sticky bits combined; track result sign.
//   5. Leading-one detect, normalise, extract mantissa+guard+round+sticky,
//      RNE round, repack. Handle subnormal-output via additional right
//      shift before extraction.
//
// LATENCY = 5 cycles (fp16_lat_pkg::FP16_FMA_LAT). Five register stages:
//   s1 unpack+11x11 multiply+align(place_op);
//   s2 add/sub of the aligned operands;
//   s3 leading-one detect (LZD);
//   s4 normalize barrel-shift + exponent/subnormal-shift calc;
//   s5 subnormal right-shift + mantissa extract + RNE round + pack.
// The cuts after the add (s2|s3) and between the LZD and the normalize shift
// (s3|s4) break the serial add→LZD→shift chain that caps Fmax in 7nm; closes
// 1 GHz on ASAP7 (RVT and SLVT). Keep this in lockstep with fp16_macw, which
// is FMA_LAT-pinned to the same depth so flash_attn stays a drop-in swap.
// Consumer DV fails if the package value and this depth ever drift.

module fp16_fma (
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic [15:0] a_i,
  input  logic [15:0] b_i,
  input  logic [15:0] c_i,

  output logic [15:0] y_o
);

  // ───────────────────────── unpack ──────────────────────────
  function automatic void unpack_fp16(
      input  logic [15:0] x,
      output logic        s,
      output logic [10:0] sig,
      output logic signed [7:0] eff_exp,
      output logic        is_zero,
      output logic        is_inf,
      output logic        is_nan
  );
    logic [4:0] bexp;
    logic [9:0] frac;
    begin
      s    = x[15];
      bexp = x[14:10];
      frac = x[9:0];
      is_zero = (bexp == 5'd0)  && (frac == 10'd0);
      is_inf  = (bexp == 5'd31) && (frac == 10'd0);
      is_nan  = (bexp == 5'd31) && (frac != 10'd0);
      if (bexp == 5'd0) begin
        sig     = {1'b0, frac};
        eff_exp = -8'sd24;
      end else begin
        sig     = {1'b1, frac};
        eff_exp = $signed({3'b000, bexp}) - 8'sd25;
      end
    end
  endfunction

  logic        sa, sb, sc;
  logic [10:0] siga, sigb, sigc;
  logic signed [7:0] ea, eb, ec;
  logic        a_zero, a_inf, a_nan;
  logic        b_zero, b_inf, b_nan;
  logic        c_zero, c_inf, c_nan;

  always_comb begin
    unpack_fp16(a_i, sa, siga, ea, a_zero, a_inf, a_nan);
    unpack_fp16(b_i, sb, sigb, eb, b_zero, b_inf, b_nan);
    unpack_fp16(c_i, sc, sigc, ec, c_zero, c_inf, c_nan);
  end

  // ───────────────────────── product ─────────────────────────
  logic               sp;
  logic [21:0]        sig_p;
  logic signed [9:0]  ep;

  assign sp    = sa ^ sb;
  assign sig_p = siga * sigb;
  assign ep    = $signed({{2{ea[7]}}, ea}) + $signed({{2{eb[7]}}, eb});

  // Product is zero when either input is zero (or both subnormals
  // multiply to all zero — but that's caught downstream via sig_p == 0).
  logic prod_zero;
  assign prod_zero = a_zero || b_zero || (sig_p == 22'd0);

  // ────────────────────── special cases ──────────────────────
  logic prod_is_inf;        // a*b is Inf (non-NaN path)
  logic prod_is_nan_special;
  assign prod_is_inf = (a_inf || b_inf) && !(a_zero || b_zero) && !(a_nan || b_nan);
  assign prod_is_nan_special = (a_inf && b_zero) || (b_inf && a_zero);

  logic out_is_nan;
  assign out_is_nan = a_nan || b_nan || c_nan ||
                      prod_is_nan_special ||
                      (prod_is_inf && c_inf && (sp != sc));

  logic out_is_inf;
  assign out_is_inf = !out_is_nan && (prod_is_inf || c_inf);

  logic out_inf_sign;
  assign out_inf_sign = (c_inf && !prod_is_inf) ? sc : sp;

  // c contribution is zero when c is zero.
  logic c_is_zero_eff;
  assign c_is_zero_eff = c_zero;

  // ─────────────────────── alignment ────────────────────────
  // Top exponents (rough — uses worst-case leading bit position, refined
  // later by leading-one detect on the sum).
  //   top_p = ep + 21 (bit 21 of sig_p, which is the worst-case top).
  //   top_c = ec + 10 (bit 10 of sigc).
  // We pick e_low so the dominant operand's worst-case top lands at acc
  // bit 47, leaving room for one carry bit at the top.
  logic signed [11:0] ep_ext, ec_ext;
  assign ep_ext = $signed({{2{ep[9]}}, ep});
  assign ec_ext = $signed({{4{ec[7]}}, ec});

  logic signed [11:0] top_p, top_c;
  assign top_p = ep_ext + 12'sd21;
  assign top_c = ec_ext + 12'sd10;

  logic               anchor_is_c;
  assign anchor_is_c = (top_c > top_p);

  logic signed [11:0] e_low;
  assign e_low = (anchor_is_c ? top_c : top_p) - 12'sd47;

  // Shifts of sig_p and sigc relative to e_low (signed; can be negative).
  logic signed [11:0] sp_shift, sc_shift;
  assign sp_shift = ep_ext - e_low;
  assign sc_shift = ec_ext - e_low;

  // Place each operand into a 100-bit working register. We anchor sig at
  // bit 50 by default; shifting left by sh moves it to bit (50+sh). The
  // top 50 bits [99:50] are the accumulator window; the bottom 50 bits
  // [49:0] are the sticky region. Bits that fall off the bottom are
  // captured in the sticky bit; bits that shift above bit 99 only happen
  // if the operand's leading bit is above the anchor — guarded against
  // by the anchor-swap.
  //
  // Use unsigned shift counts. When the signed shift is negative we use
  // right-shift; when positive we use left-shift. Clip absurd magnitudes
  // (which would only occur for the lesser operand) to a value larger
  // than the wide register so the operand becomes "all sticky" / zero.
  logic [99:0] wide_p, wide_c;

  function automatic logic [99:0] place_op(
      input  logic [21:0]        val,
      input  logic signed [11:0] sh
  );
    logic [99:0] base;
    logic [99:0] out;
    int          mag;
    begin
      base = {78'd0, val} << 50;   // val at bits [71:50]
      if (sh >= 0) begin
        // left shift. By construction (anchor swap on top_p vs top_c)
        // the leading bit cannot rise above wide bit 99.
        mag = int'(sh);
        if (mag > 99) out = 100'd0;
        else          out = base << mag;
      end else begin
        mag = -int'(sh);
        if (mag > 99) out = 100'd0;
        else          out = base >> mag;
      end
      return out;
    end
  endfunction

  assign wide_p = place_op(sig_p,                 sp_shift);
  assign wide_c = place_op({11'd0, sigc},         sc_shift);

  // (sigc is 11 bits; passed as a 22-bit value via zero-extension.)

  // Mask out product or c contributions when their respective values
  // are effectively zero (one of a/b zero, or c zero).
  logic [99:0] wp, wc;
  assign wp = prod_zero     ? 100'd0 : wide_p;
  assign wc = c_is_zero_eff ? 100'd0 : wide_c;

  // ═════════════════ stage-1 register (after align) ══════════
  // Cut after the 11×11 multiply and the 100-bit alignment barrel-shifts.
  // Carry forward the aligned operands, their signs, the anchor exponent,
  // and the special-case decisions (decoded combinationally from the
  // inputs in stage 1) so later stages don't re-derive them.
  logic [99:0]        s1_wp, s1_wc;
  logic               s1_sp, s1_sc;
  logic signed [11:0] s1_e_low;
  logic               s1_out_is_nan, s1_out_is_inf, s1_out_inf_sign;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s1_wp           <= 100'd0;
      s1_wc           <= 100'd0;
      s1_sp           <= 1'b0;
      s1_sc           <= 1'b0;
      s1_e_low        <= 12'sd0;
      s1_out_is_nan   <= 1'b0;
      s1_out_is_inf   <= 1'b0;
      s1_out_inf_sign <= 1'b0;
    end else begin
      s1_wp           <= wp;
      s1_wc           <= wc;
      s1_sp           <= sp;
      s1_sc           <= sc;
      s1_e_low        <= e_low;
      s1_out_is_nan   <= out_is_nan;
      s1_out_is_inf   <= out_is_inf;
      s1_out_inf_sign <= out_inf_sign;
    end
  end

  // ─────────────────── add / subtract magnitudes ─────────────
  // Do the add/sub in full 100-bit precision; we capture sticky only
  // at the final mantissa extraction stage.
  logic same_sign;
  assign same_sign = (s1_sp == s1_sc);

  logic ge_pc;
  assign ge_pc = (s1_wp >= s1_wc);

  logic [100:0] sum_add;       // 101 bits to capture carry
  logic [99:0]  sum_sub;       // |wp - wc|

  assign sum_add = {1'b0, s1_wp} + {1'b0, s1_wc};
  assign sum_sub = ge_pc ? (s1_wp - s1_wc) : (s1_wc - s1_wp);

  logic [100:0] mag_raw;       // 101-bit magnitude (signed-magnitude form)
  logic         result_sign;
  always_comb begin
    if (same_sign) begin
      mag_raw     = sum_add;
      result_sign = s1_sp;
    end else begin
      mag_raw     = {1'b0, sum_sub};
      result_sign = ge_pc ? s1_sp : s1_sc;
    end
  end

  // ═════════════════ stage-2 register (after add) ════════════
  // Cut between the wide add/sub and the leading-zero detect. Register the
  // raw magnitude + result sign + anchor exponent + specials. This split
  // (and the s3 cut below) breaks the add→LZD→normalize-shift serial chain
  // that the 7nm timing shows is the Fmax wall.
  logic [100:0]       s2_mag_raw;
  logic               s2_result_sign;
  logic signed [11:0] s2_e_low;
  logic               s2_out_is_nan, s2_out_is_inf, s2_out_inf_sign;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s2_mag_raw      <= 101'd0;
      s2_result_sign  <= 1'b0;
      s2_e_low        <= 12'sd0;
      s2_out_is_nan   <= 1'b0;
      s2_out_is_inf   <= 1'b0;
      s2_out_inf_sign <= 1'b0;
    end else begin
      s2_mag_raw      <= mag_raw;
      s2_result_sign  <= result_sign;
      s2_e_low        <= s1_e_low;
      s2_out_is_nan   <= s1_out_is_nan;
      s2_out_is_inf   <= s1_out_is_inf;
      s2_out_inf_sign <= s1_out_inf_sign;
    end
  end

  // s2_mag_raw bit i (for i in [0..99]) represents 2^((i-50) + e_low).
  // Bit 100 is the same-sign carry → represents 2^(50 + e_low).
  //
  // Find leading-1: lz = number of leading zeros in s2_mag_raw (0..101).
  logic [7:0] lz;
  logic       any_one;
  always_comb begin
    any_one = 1'b0;
    lz      = 8'd101;
    for (int i = 100; i >= 0; i--) begin
      if (s2_mag_raw[i] && !any_one) begin
        any_one = 1'b1;
        lz      = 8'(100 - i);
      end
    end
  end

  // ═════════════════ stage-3 register (after LZD) ════════════
  // Cut between the leading-zero detect and the normalize barrel-shift —
  // the single cut that mattered most for 7nm Fmax (LZD result *is* the
  // shift amount, so the two are otherwise a serial dependency).
  logic [100:0]       s3_mag_raw;
  logic [7:0]         s3_lz;
  logic               s3_any_one, s3_result_sign;
  logic signed [11:0] s3_e_low;
  logic               s3_out_is_nan, s3_out_is_inf, s3_out_inf_sign;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s3_mag_raw      <= 101'd0;
      s3_lz           <= 8'd0;
      s3_any_one      <= 1'b0;
      s3_result_sign  <= 1'b0;
      s3_e_low        <= 12'sd0;
      s3_out_is_nan   <= 1'b0;
      s3_out_is_inf   <= 1'b0;
      s3_out_inf_sign <= 1'b0;
    end else begin
      s3_mag_raw      <= s2_mag_raw;
      s3_lz           <= lz;
      s3_any_one      <= any_one;
      s3_result_sign  <= s2_result_sign;
      s3_e_low        <= s2_e_low;
      s3_out_is_nan   <= s2_out_is_nan;
      s3_out_is_inf   <= s2_out_is_inf;
      s3_out_inf_sign <= s2_out_inf_sign;
    end
  end

  // Exponent of the leading-1: bit (100-lz) → e = (100-lz - 50) + e_low
  //                                                = 50 - lz + e_low.
  logic signed [11:0] e_msb;
  assign e_msb = 12'sd50 - $signed({4'b0, s3_lz}) + s3_e_low;

  // Normalise: shift s3_mag_raw left by lz so leading-1 sits at bit 100.
  logic [100:0] mag_norm_wide;
  always_comb begin
    if (s3_lz >= 8'd101) mag_norm_wide = 101'd0;
    else                 mag_norm_wide = s3_mag_raw << s3_lz[6:0];
  end

  // We will extract 10 mantissa bits + guard + round + sticky from
  // mag_norm_wide. With leading-1 at bit 100:
  //   mantissa10 = bits[99:90]
  //   guard      = bit[89]
  //   round      = bit[88]
  //   sticky     = OR(bits[87:0])
  // Use mag_norm[100:0] but we'll alias to a 101-bit name `mag_norm`.
  logic [100:0] mag_norm;
  assign mag_norm = mag_norm_wide;

  // Normal-output extraction (if e_msb >= -14):
  //   mantissa10 = mag_norm[99:90]
  //   guard      = mag_norm[89]
  //   round      = mag_norm[88]
  //   sticky     = OR(mag_norm[87:0])
  //   biased exp = e_msb + 15
  logic signed [11:0] biased_exp_pre;
  assign biased_exp_pre = e_msb + 12'sd15;

  // Subnormal path: if biased_exp_pre <= 0, right-shift mag_norm by
  // (1 - biased_exp_pre). Then biased exp field is 0 and we read the
  // mantissa from the same bit positions.
  logic signed [11:0] sub_shift_s;
  assign sub_shift_s = 12'sd1 - biased_exp_pre;

  logic [7:0] sub_shift;
  logic       is_sub_path;
  always_comb begin
    is_sub_path = (biased_exp_pre <= 12'sd0);
    if (!is_sub_path) sub_shift = 8'd0;
    else if (sub_shift_s > 12'sd101) sub_shift = 8'd101;
    else sub_shift = 8'(sub_shift_s);
  end

  // ═════════════════ stage-4 register (after normalize) ══════
  // Cut after the normalise barrel-shift. The subnormal right-shift,
  // mantissa extraction, RNE round, and pack all run in stage 5.
  logic [100:0]       s4_mag_norm;
  logic [7:0]         s4_sub_shift;
  logic               s4_is_sub_path;
  logic signed [11:0] s4_biased_exp_pre;
  logic               s4_result_sign;
  logic               s4_any_one;
  logic               s4_out_is_nan, s4_out_is_inf, s4_out_inf_sign;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s4_mag_norm       <= 101'd0;
      s4_sub_shift      <= 8'd0;
      s4_is_sub_path    <= 1'b0;
      s4_biased_exp_pre <= 12'sd0;
      s4_result_sign    <= 1'b0;
      s4_any_one        <= 1'b0;
      s4_out_is_nan     <= 1'b0;
      s4_out_is_inf     <= 1'b0;
      s4_out_inf_sign   <= 1'b0;
    end else begin
      s4_mag_norm       <= mag_norm;
      s4_sub_shift      <= sub_shift;
      s4_is_sub_path    <= is_sub_path;
      s4_biased_exp_pre <= biased_exp_pre;
      s4_result_sign    <= s3_result_sign;
      s4_any_one        <= s3_any_one;
      s4_out_is_nan     <= s3_out_is_nan;
      s4_out_is_inf     <= s3_out_is_inf;
      s4_out_inf_sign   <= s3_out_inf_sign;
    end
  end

  logic [100:0] mag_post;
  logic         _unused_mag_post_top;
  assign _unused_mag_post_top = mag_post[100];
  logic         sub_dropped;
  always_comb begin
    mag_post    = s4_mag_norm >> s4_sub_shift[6:0];
    sub_dropped = 1'b0;
    for (int i = 0; i < 101; i++) begin
      if ((8'(i) < s4_sub_shift) && s4_mag_norm[i]) sub_dropped = 1'b1;
    end
  end

  logic [9:0] mant10;
  logic       guard_b, round_b, sticky_b;
  assign mant10   = mag_post[99:90];
  assign guard_b  = mag_post[89];
  assign round_b  = mag_post[88];
  assign sticky_b = (|mag_post[87:0]) | sub_dropped;

  // ─────────────────────── RNE round ────────────────────────
  logic round_up;
  assign round_up = guard_b && ((round_b | sticky_b) || mant10[0]);

  logic [10:0] mant_rounded;
  assign mant_rounded = {1'b0, mant10} + 11'(round_up);

  // ─────────────────────── pack ─────────────────────────────
  logic out_zero_exact;
  assign out_zero_exact = !s4_any_one && !s4_out_is_nan && !s4_out_is_inf;

  logic signed [11:0] biased_exp_pack;
  logic [9:0]         frac_final;
  always_comb begin
    if (s4_is_sub_path) begin
      // After the subnormal right-shift, the biased exp field is 0
      // unless mantissa rounding promotes to smallest normal.
      if (mant_rounded[10]) begin
        biased_exp_pack = 12'sd1;
        frac_final      = 10'd0;
      end else begin
        biased_exp_pack = 12'sd0;
        frac_final      = mant_rounded[9:0];
      end
    end else begin
      if (mant_rounded[10]) begin
        biased_exp_pack = s4_biased_exp_pre + 12'sd1;
        frac_final      = 10'd0;
      end else begin
        biased_exp_pack = s4_biased_exp_pre;
        frac_final      = mant_rounded[9:0];
      end
    end
  end

  logic [15:0] y_d;
  always_comb begin
    if (s4_out_is_nan) begin
      y_d = 16'h7E00;
    end else if (s4_out_is_inf) begin
      y_d = {s4_out_inf_sign, 5'b11111, 10'b0};
    end else if (out_zero_exact) begin
      // RNE: exact-zero result from cancellation is +0.
      y_d = 16'h0000;
    end else if (biased_exp_pack >= 12'sd31) begin
      y_d = {s4_result_sign, 5'b11111, 10'b0};
    end else if (biased_exp_pack <= 12'sd0) begin
      // Flush-to-zero on extreme underflow only if mant_rounded is also
      // zero (i.e., the subnormal-shift dropped everything). Otherwise
      // emit a proper subnormal: biased exp field 0, frac = mantissa.
      if (mant_rounded == 11'd0) begin
        y_d = {s4_result_sign, 15'd0};
      end else begin
        y_d = {s4_result_sign, 5'd0, frac_final};
      end
    end else begin
      y_d = {s4_result_sign, biased_exp_pack[4:0], frac_final};
    end
  end

  // ─── unused ─────
  logic _unused;
  assign _unused = ^{c_zero, ec_ext[11], ep_ext[11], 1'b0};

  // ═════════════════ stage-5 register (output) ═══════════════
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) y_o <= 16'h0000;
    else         y_o <= y_d;
  end

endmodule
