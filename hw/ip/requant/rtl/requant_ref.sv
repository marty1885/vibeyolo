// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// requant_ref — INDEPENDENT behavioral golden for `requant`.
//
// This module does NOT instantiate i32_to_fp16 / fp16_fma / fp16_to_i8_sat.
// It implements the full int32 → int8 chain from scratch, in pure SV
// integer arithmetic with a wide accumulator, so the composition of the
// three sub-blocks can be checked end-to-end against a different code path.
//
// Pipeline: combinational compute (cvt → fma → sat), then a 3-deep output
// register chain so the latency matches the DUT exactly.
//
// Algorithm:
//   - cvt_i32_fp16 : sign + leading-1 scan + 10-bit mantissa with RNE.
//   - fma_fp16     : unpack a,b,c → exact product significand (22 bit) and
//                    significand c (11 bit) → place both in a 128-bit
//                    window anchored to the lower exponent → magnitude
//                    add/sub → leading-1 detect → RNE round → pack. Handles
//                    NaN, ±Inf, signed zero, subnormals.
//   - sat_fp16_i8  : unpack → integer right-shift with guard/sticky →
//                    RNE → sign apply → clamp to [-128, +127].

module requant_ref (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               valid_i,
  input  logic signed [31:0] acc_i,
  input  logic        [15:0] scale_fp16_i,
  input  logic        [15:0] bias_fp16_i,

  output logic               valid_o,
  output logic signed  [7:0] y_o
);

  // ───────────────────── int32 → (fp16, shift) ────────────
  // Returns 21 bits {shift[4:0], fp16[15:0]} mirroring i32_to_fp16's
  // dual-output interface.
  function automatic logic [20:0] cvt_i32_fp16(input logic signed [31:0] x);
    logic        s;
    logic [31:0] mag;
    int          msb;
    int          pre_shift;
    int          e_post;
    logic [31:0] mag_sh;
    logic        extra_sticky;
    int          biased;
    logic [10:0] mant_pre;
    logic [11:0] mant_round;
    logic        guard, sticky, lsb, round_up;
    int          sh;
    logic [9:0]  mant_final;
    logic [15:0] fp16;
    begin
      if (x == 32'sd0) return {5'd0, 16'h0000};
      s   = x[31];
      mag = s ? (~x + 32'd1) : x;
      msb = 0;
      for (int i = 31; i >= 0; i--) begin
        if (mag[i] && (msb == 0)) msb = i;
      end
      if (msb <= 15) pre_shift = 0;
      else           pre_shift = msb - 14;
      if (pre_shift == 0) begin
        mag_sh       = mag;
        extra_sticky = 1'b0;
        e_post       = msb;
      end else begin
        mag_sh       = mag >> pre_shift;
        extra_sticky = (|(mag & ((32'd1 << pre_shift) - 32'd1)));
        e_post       = 14;
      end
      biased = e_post + 15;
      if (e_post <= 10) begin
        sh         = 10 - e_post;
        mant_pre   = 11'(mag_sh) << sh;
        guard      = 1'b0;
        sticky     = extra_sticky;
      end else begin
        sh         = e_post - 10;
        mant_pre   = 11'(mag_sh >> sh);
        guard      = mag_sh[sh - 1];
        sticky     = extra_sticky;
        for (int j = 0; j < 32; j++) begin
          if (j < sh - 1 && mag_sh[j]) sticky = 1'b1;
        end
      end
      lsb        = mant_pre[0];
      round_up   = guard && (sticky || lsb);
      mant_round = {1'b0, mant_pre} + 12'(round_up);
      if (mant_round[11]) begin
        biased     = biased + 1;
        mant_final = 10'd0;
      end else begin
        if (mant_round[10] != 1'b1) mant_final = 10'h3FF;
        else                        mant_final = mant_round[9:0];
      end
      if (biased >= 31) fp16 = {s, 15'h7C00};
      else              fp16 = {s, 5'(biased), mant_final};
      return {5'(pre_shift), fp16};
    end
  endfunction

  // Bump the biased-exponent field of `scale` by `shift`, saturated at 30,
  // skipping zero/subnormal/Inf/NaN.
  function automatic logic [15:0] bump_scale(input logic [15:0] scale,
                                             input logic [4:0]  shift);
    logic        s;
    logic [4:0]  e;
    logic [9:0]  f;
    logic [5:0]  e_sum;
    logic [4:0]  e_new;
    begin
      s = scale[15];
      e = scale[14:10];
      f = scale[9:0];
      if (e == 5'd0 || e == 5'd31) return scale;
      e_sum = {1'b0, e} + {1'b0, shift};
      e_new = (e_sum >= 6'd30) ? 5'd30 : e_sum[4:0];
      return {s, e_new, f};
    end
  endfunction

  // ───────────────────── fp16 FMA  a*b + c ─────────────────
  //
  // Unpack each operand into (sign, kind, sig, exp). The value is
  //   (-1)^sign * sig * 2^exp
  // with sig non-negative. We use a 128-bit signed-magnitude accumulator.

  typedef struct packed {
    logic        sign;
    logic [2:0]  kind;   // 0 zero, 1 normal, 2 subnormal, 3 inf, 4 nan
    logic [63:0] sig;    // upper bits unused — gives headroom for shifts
    int          exp;
  } fp16_u_t;

  function automatic fp16_u_t unpack_fp16(input logic [15:0] x);
    fp16_u_t u;
    logic [4:0] bexp;
    logic [9:0] frac;
    begin
      u.sign = x[15];
      bexp   = x[14:10];
      frac   = x[9:0];
      u.sig  = 64'd0;
      u.exp  = 0;
      if (bexp == 5'd0 && frac == 10'd0) begin
        u.kind = 3'd0;
      end else if (bexp == 5'd31 && frac == 10'd0) begin
        u.kind = 3'd3;
      end else if (bexp == 5'd31) begin
        u.kind = 3'd4;
      end else if (bexp == 5'd0) begin
        u.kind = 3'd2;
        // subnormal: value = frac * 2^-24 → sig at bits[22:13], exp=-37.
        u.sig  = {54'd0, frac} << 13;
        u.exp  = -37;
      end else begin
        u.kind = 3'd1;
        // normal: value = (1024+frac) * 2^(bexp-25) → put (1024+frac) at
        // bits[23:13], exp=(bexp-25)-13=bexp-38.
        u.sig  = {53'd0, 1'b1, frac} << 13;
        u.exp  = int'(bexp) - 38;
      end
      return u;
    end
  endfunction

  function automatic logic [15:0] pack_zero(input logic s);
    return {s, 15'd0};
  endfunction
  function automatic logic [15:0] pack_inf(input logic s);
    return {s, 5'b11111, 10'd0};
  endfunction
  function automatic logic [15:0] pack_nan();
    return 16'h7E00;
  endfunction

  function automatic logic [15:0] fma_fp16(
      input logic [15:0] a,
      input logic [15:0] b,
      input logic [15:0] c);
    fp16_u_t      ua, ub, uc;
    logic         sp;
    logic         prod_inf, prod_zero, c_zero, c_inf, any_nan;
    logic [127:0] sigprod;       // 24*24 exact in 48 bits, fits easily
    int           eprod;
    int           anchor;
    logic [127:0] big_p, big_c;
    logic [127:0] mag;
    logic         result_sign;
    int           msb;
    int           biased;
    int           shift;
    int           eff_biased;
    logic [9:0]   mant10;
    logic         guard, round_b, sticky_b, lsb, round_up;
    logic [10:0]  mant_r;
    logic [127:0] below_mask;
    int           shc;
    int           shp;
    begin
      ua = unpack_fp16(a);
      ub = unpack_fp16(b);
      uc = unpack_fp16(c);

      any_nan = (ua.kind == 3'd4) || (ub.kind == 3'd4) || (uc.kind == 3'd4);
      if (any_nan) return pack_nan();

      prod_inf  = (ua.kind == 3'd3) || (ub.kind == 3'd3);
      prod_zero = (ua.kind == 3'd0) || (ub.kind == 3'd0);
      c_zero    = (uc.kind == 3'd0);
      c_inf     = (uc.kind == 3'd3);
      sp        = ua.sign ^ ub.sign;

      if (prod_inf && prod_zero) return pack_nan();
      if (prod_inf && c_inf && (sp != uc.sign)) return pack_nan();
      if (prod_inf) return pack_inf(sp);
      if (c_inf)    return pack_inf(uc.sign);

      // Exact product significand and exponent.
      sigprod = ua.sig * ub.sig;          // ≤ 48 bits
      eprod   = ua.exp + ub.exp;

      if ((prod_zero || (sigprod == 128'd0)) && c_zero) begin
        return pack_zero(1'b0);  // RNE: exact zero → +0
      end

      if (prod_zero || (sigprod == 128'd0)) begin
        anchor = uc.exp;
      end else if (c_zero) begin
        anchor = eprod;
      end else begin
        anchor = (eprod < uc.exp) ? eprod : uc.exp;
      end

      // Place each operand at the common anchor with a positive left-shift.
      // Shift counts are bounded by ~|exp diff| < 90, well within 128.
      if (prod_zero || (sigprod == 128'd0)) begin
        big_p = 128'd0;
      end else begin
        shp = eprod - anchor;
        if (shp < 0) shp = 0;
        big_p = sigprod << shp;
      end
      if (c_zero) begin
        big_c = 128'd0;
      end else begin
        shc = uc.exp - anchor;
        if (shc < 0) shc = 0;
        big_c = {64'd0, uc.sig} << shc;
      end

      if (sp == uc.sign) begin
        mag         = big_p + big_c;
        result_sign = sp;
      end else if (big_p >= big_c) begin
        mag         = big_p - big_c;
        result_sign = sp;
      end else begin
        mag         = big_c - big_p;
        result_sign = uc.sign;
      end

      if (mag == 128'd0) return pack_zero(1'b0);

      // MSB index of mag.
      msb = 0;
      for (int i = 127; i >= 0; i--) begin
        if (mag[i] && (msb == 0)) msb = i;
      end

      biased = anchor + msb + 15;
      if (biased >= 31) return pack_inf(result_sign);

      // Place leading-1 at bit 10.
      shift = msb - 10;
      if (biased <= 0) begin
        shift      = shift + (1 - biased);
        eff_biased = 0;
      end else begin
        eff_biased = biased;
      end

      // Extract mantissa, guard, round, sticky.
      if (shift <= 0) begin
        mant10   = 10'(mag << (-shift)) & 10'h3FF;
        guard    = 1'b0;
        round_b  = 1'b0;
        sticky_b = 1'b0;
      end else if (shift == 1) begin
        mant10   = 10'(mag >> 1) & 10'h3FF;
        guard    = mag[0];
        round_b  = 1'b0;
        sticky_b = 1'b0;
      end else if (shift == 2) begin
        mant10   = 10'(mag >> 2) & 10'h3FF;
        guard    = mag[1];
        round_b  = mag[0];
        sticky_b = 1'b0;
      end else begin
        mant10   = 10'(mag >> shift) & 10'h3FF;
        guard    = mag[shift - 1];
        round_b  = mag[shift - 2];
        below_mask = (128'd1 << (shift - 2)) - 128'd1;
        sticky_b   = ((mag & below_mask) != 128'd0);
      end

      lsb      = mant10[0];
      round_up = guard && ((round_b | sticky_b) || lsb);
      mant_r   = {1'b0, mant10} + 11'(round_up);
      if (mant_r[10]) begin
        eff_biased = eff_biased + 1;
        mant_r     = 11'd0;
      end

      if (eff_biased >= 31) return pack_inf(result_sign);
      if (eff_biased <= 0)  return {result_sign, 5'd0, mant_r[9:0]};
      return {result_sign, 5'(eff_biased), mant_r[9:0]};
    end
  endfunction

  // ───────────────────── fp16 → int8 (saturate) ────────────
  function automatic logic signed [7:0] sat_fp16_i8(input logic [15:0] x);
    logic        s;
    logic [4:0]  bexp;
    logic [9:0]  frac;
    logic [10:0] sig11;
    int          e_unb;
    int          sh;
    logic [7:0]  mag_int;
    logic        guard, sticky, round_up;
    logic [8:0]  mag_rnd;
    logic [10:0] mask;
    begin
      s    = x[15];
      bexp = x[14:10];
      frac = x[9:0];

      // Specials
      if (bexp == 5'h1F && frac != 10'd0) return 8'sd0;             // NaN
      if (bexp == 5'h1F)                  return s ? -8'sd128 : 8'sd127;
      if (bexp == 5'd0)                   return 8'sd0;             // ±0/subnormal

      sig11 = {1'b1, frac};
      e_unb = int'(bexp) - 15;

      if (e_unb >= 7)  return s ? -8'sd128 : 8'sd127;
      if (e_unb <= -2) return 8'sd0;

      // sh in [4..11] for e_unb in [-1..6].
      sh = 10 - e_unb;

      mag_int = 8'(sig11 >> sh);
      guard   = sig11[sh - 1];
      if (sh >= 2) begin
        mask   = (11'd1 << (sh - 1)) - 11'd1;
        sticky = |(sig11 & mask);
      end else begin
        sticky = 1'b0;
      end
      round_up = guard && (sticky || mag_int[0]);
      mag_rnd  = {1'b0, mag_int} + 9'(round_up);

      if (!s && mag_rnd >= 9'd128)        return 8'sd127;
      if ( s && mag_rnd >  9'd128)        return -8'sd128;
      if ( s && mag_rnd == 9'd128)        return -8'sd128;
      if (s) return 8'(-$signed({1'b0, mag_rnd[7:0]}));
      return 8'($signed({1'b0, mag_rnd[7:0]}));
    end
  endfunction

  // ───────────────────── combinational chain ───────────────
  logic [20:0]       cvt_pack;
  logic [15:0]       c1_fp16;
  logic [4:0]        c1_shift;
  logic [15:0]       c_scale_bumped;
  logic [15:0]       c2_fma;
  logic signed [7:0] c3_i8;

  always_comb begin
    cvt_pack       = cvt_i32_fp16(acc_i);
    c1_fp16        = cvt_pack[15:0];
    c1_shift       = cvt_pack[20:16];
    c_scale_bumped = bump_scale(scale_fp16_i, c1_shift);
    c2_fma         = fma_fp16(c1_fp16, c_scale_bumped, bias_fp16_i);
    c3_i8          = sat_fp16_i8(c2_fma);
  end

  // ───────────────────── 4-cycle output align ──────────────
  logic signed [7:0] r1, r2, r3, r4;
  logic [3:0]        v_sr;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      r1   <= 8'sd0;
      r2   <= 8'sd0;
      r3   <= 8'sd0;
      r4   <= 8'sd0;
      v_sr <= 4'b0000;
    end else begin
      r1   <= c3_i8;
      r2   <= r1;
      r3   <= r2;
      r4   <= r3;
      v_sr <= {v_sr[2:0], valid_i};
    end
  end

  assign y_o     = r4;
  assign valid_o = v_sr[3];

endmodule
