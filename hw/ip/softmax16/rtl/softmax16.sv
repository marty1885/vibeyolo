// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// softmax16 — 16-way fp16 softmax for YOLO DFL detection head.
//
//   y[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x))
//
// Interface: parallel-in / parallel-out, 16 fp16 lanes, single
// valid_i / valid_o flowing along the pipeline.
//
// Pipeline (per-stage registered):
//   S0  : capture inputs (x_q, v0)
//   S1  : max tree level 0 (8 pairs)
//   S2  : max tree level 1 (4)
//   S3  : max tree level 2 (2)
//   S4  : max tree level 3 (1)   → m
//   S5  : compute d[i] = m - x[i] (fp16 sub, clamped to [0,16]) ; e[i] = exp_lut(d[i])
//         (exp lookup is combinational after the sub, registered into S5)
//   S6  : sum tree level 0 (8 pairs of fp16 add)
//   S7  : sum tree level 1 (4)
//   S8  : sum tree level 2 (2)
//   S9  : sum tree level 3 (1)   → s
//   S10 : inv_s = recip_lut(s)
//   S11 : per-lane fp16_fma(e[i], inv_s, 0)  → y[i]
//         (fp16_fma itself adds FMA_LAT=5 cycles)
//
// Total latency = 11 (S0..S10 register stages) + FMA_LAT(5) = 16 cycles
// from valid_i to valid_o.
//
// Note: e[i] is computed in S5 and must be carried alongside the sum
// tree for 4 more cycles so it can drive the multiplier. We keep an
// "e_pipe[i]" register chain.

module softmax16 (
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic        valid_i,
  input  logic [15:0] x_i [16],

  output logic        valid_o,
  output logic [15:0] y_o [16]
);

  // ───────────────────────── fp16 helpers ────────────────────────

  // Returns 1 if a > b in real value (sign-magnitude fp16). NaN treated
  // as smaller (DFL inputs are bounded — NaN is not expected, but the
  // helper is defined for completeness).
  function automatic logic fp16_gt(input logic [15:0] a, input logic [15:0] b);
    logic sa, sb;
    logic [4:0] ea, eb;
    logic [9:0] fa, fb;
    logic [14:0] ma, mb;     // magnitude bits
    logic a_zero, b_zero;
    begin
      sa = a[15]; ea = a[14:10]; fa = a[9:0];
      sb = b[15]; eb = b[14:10]; fb = b[9:0];
      a_zero = (ea == 0) && (fa == 0);
      b_zero = (eb == 0) && (fb == 0);
      ma = a[14:0];
      mb = b[14:0];
      // ±0 compare as equal
      if (a_zero && b_zero)      fp16_gt = 1'b0;
      else if (sa != sb)         fp16_gt = !sa;             // positive > negative
      else if (sa == 1'b0)       fp16_gt = (ma > mb);       // both positive
      else                       fp16_gt = (ma < mb);       // both negative
    end
  endfunction

  function automatic logic [15:0] fp16_max(
      input logic [15:0] a, input logic [15:0] b);
    fp16_max = fp16_gt(a, b) ? a : b;
  endfunction

  // Negation: flip sign bit. (Does not touch NaN payload.)
  function automatic logic [15:0] fp16_neg(input logic [15:0] a);
    fp16_neg = {~a[15], a[14:0]};
  endfunction

  // fp16 add: a + b, IEEE-754 binary16, RNE rounding, FTZ on extreme
  // underflow but subnormal outputs preserved. Implemented as a small
  // integer-arithmetic kernel (no use of fp16_fma so the sum tree and
  // the per-lane multiplier are independent kernels).
  //
  // For softmax we know the operands are non-negative finite values
  // bounded by 16 (sum of 16 entries each ≤ 1). The kernel still handles
  // signed operands correctly; it does not have to handle NaN/Inf — if
  // they arrive, behaviour is best-effort (NaN passthrough).
  function automatic logic [15:0] fp16_add(
      input logic [15:0] a, input logic [15:0] b);
    logic        sa, sb;
    logic [4:0]  ea_b, eb_b;
    logic [9:0]  fa, fb;
    logic        a_z, b_z;
    logic        a_inf, b_inf;
    logic        a_nan, b_nan;
    logic [10:0] sig_a, sig_b;    // 11-bit (implicit-1 at bit 10)
    int          ea, eb;
    int          eshift;
    logic [31:0] big_a, big_b;    // 32-bit aligned representations
    int          sh;
    logic        sticky;
    logic [31:0] sum_mag;
    logic        result_sign;
    int          msb;
    int          shift10;
    logic        guard, round_b, st;
    logic [9:0]  mant10;
    logic [10:0] mant_r;
    int          biased_exp;
    int          anchor;
    logic [31:0] tmp;
    begin
      sa = a[15]; ea_b = a[14:10]; fa = a[9:0];
      sb = b[15]; eb_b = b[14:10]; fb = b[9:0];
      a_z = (ea_b == 0) && (fa == 0);
      b_z = (eb_b == 0) && (fb == 0);
      a_inf = (ea_b == 31) && (fa == 0);
      b_inf = (eb_b == 31) && (fb == 0);
      a_nan = (ea_b == 31) && (fa != 0);
      b_nan = (eb_b == 31) && (fb != 0);

      if (a_nan || b_nan) return 16'h7E00;
      if (a_inf && b_inf && (sa != sb)) return 16'h7E00;
      if (a_inf) return a;
      if (b_inf) return b;
      if (a_z && b_z) return 16'h0000;
      if (a_z) return b;
      if (b_z) return a;

      // Build 11-bit significand. Normals get implicit-1; subnormals get 0.
      if (ea_b == 0) begin
        sig_a = {1'b0, fa};
        ea    = -24;
      end else begin
        sig_a = {1'b1, fa};
        ea    = int'(ea_b) - 25;
      end
      if (eb_b == 0) begin
        sig_b = {1'b0, fb};
        eb    = -24;
      end else begin
        sig_b = {1'b1, fb};
        eb    = int'(eb_b) - 25;
      end

      // Align on min(ea, eb). Place each sig at bit 10 of a 32-bit word
      // (i.e., left-shift by 10) then left-shift further by (ex - anchor).
      // 32 bits is plenty for fp16-range operands.
      anchor = (ea < eb) ? ea : eb;
      sticky = 1'b0;

      sh = ea - anchor;
      if (sh > 20) begin
        // would overflow window; but with fp16 range, ea-eb max ≈ 30. Cap
        // at 20 and accumulate dropped bits as sticky implicitly.
        // We instead just keep big_a small + sticky=1.
        tmp = 32'd0;
        sticky = 1'b1;
        big_a  = tmp;
      end else begin
        big_a = {21'd0, sig_a} << sh;
      end

      sh = eb - anchor;
      if (sh > 20) begin
        tmp = 32'd0;
        sticky = 1'b1;
        big_b  = tmp;
      end else begin
        big_b = {21'd0, sig_b} << sh;
      end

      // Combine
      if (sa == sb) begin
        sum_mag = big_a + big_b;
        result_sign = sa;
      end else begin
        if (big_a >= big_b) begin
          sum_mag = big_a - big_b;
          result_sign = sa;
        end else begin
          sum_mag = big_b - big_a;
          result_sign = sb;
        end
      end

      if (sum_mag == 32'd0 && !sticky) return 16'h0000;

      // Find MSB
      msb = 0;
      for (int i = 31; i >= 0; i--) begin
        if (sum_mag[i] && (msb == 0)) msb = i;
      end

      // Unbiased exp of result = anchor + msb. We want MSB at bit 10 →
      // shift right by (msb - 10).
      biased_exp = anchor + msb + 15;
      eshift     = msb - 10;

      if (biased_exp >= 31) return {result_sign, 5'b11111, 10'b0};

      if (biased_exp <= 0) begin
        eshift = eshift + (1 - biased_exp);
        biased_exp = 0;
      end

      shift10 = eshift;
      if (shift10 < 0) begin
        tmp     = sum_mag << (-shift10);
        mant10  = tmp[9:0];
        guard   = 1'b0;
        round_b = 1'b0;
        st      = sticky;
      end else if (shift10 == 0) begin
        mant10  = sum_mag[9:0];
        guard   = 1'b0;
        round_b = 1'b0;
        st      = sticky;
      end else if (shift10 == 1) begin
        mant10  = sum_mag[10:1];
        guard   = sum_mag[0];
        round_b = 1'b0;
        st      = sticky;
      end else if (shift10 == 2) begin
        mant10  = sum_mag[11:2];
        guard   = sum_mag[1];
        round_b = sum_mag[0];
        st      = sticky;
      end else begin
        tmp     = sum_mag >> shift10;
        mant10  = tmp[9:0];
        tmp     = sum_mag >> (shift10 - 1);
        guard   = tmp[0];
        tmp     = sum_mag >> (shift10 - 2);
        round_b = tmp[0];
        st      = sticky;
        for (int j = 0; j < 32; j++) begin
          if (j < shift10 - 2 && sum_mag[j]) st = 1'b1;
        end
      end

      mant_r = {1'b0, mant10} + 11'((guard && ((round_b | st) || mant10[0])) ? 1 : 0);
      if (mant_r[10]) begin
        biased_exp = biased_exp + 1;
        mant_r     = 11'b0;
      end
      if (biased_exp >= 31) return {result_sign, 5'b11111, 10'b0};
      if (biased_exp <= 0)  return {result_sign, 5'd0, mant_r[9:0]};
      return {result_sign, 5'(biased_exp), mant_r[9:0]};
    end
  endfunction

  // ───────────────────────── exp LUT ─────────────────────────────
  // 1024 entries indexed by d = m - x in Q4.6 fixed-point on [0, 16).
  // The index is clamped to [0, 1023]. Entry i stores fp16(exp(-i/64)).
  // exp(-16) ≈ 1.13e-7 — below fp16 subnormal range — so we saturate to
  // 0 outside the LUT range. Step 1/64 keeps the worst-case relative
  // error of exp(-d) at ~(1/128) ≈ 0.78%, i.e. ~8 fp16 ULPs.
  logic [15:0] exp_lut [0:1023];

  // ───────────────────────── recip LUT ───────────────────────────
  // Reciprocal of s, where s = sum exp(x-m). s ∈ [1, 16] always (worst
  // case all 16 entries equal max → s = 16, best case dominant → s ≈ 1).
  // Index by fp16 mantissa (10 bits, 1024 entries) and shift exponent.
  // For fp16 normal value with biased exp e_b and frac f:
  //     s = (1.f) * 2^(e_b - 15)
  //   1/s = (1/(1.f)) * 2^(15 - e_b)
  // Pre-compute recip_mant_lut[f] = round_to_fp16(1.0 / (1 + f/1024)) which
  // is itself a fp16 value with biased exponent 14 (or 15 if f==0). We
  // then adjust the exponent by (15 - e_b) - (14 - 15) = (15 - e_b + 1)
  // — easier: store the recip as a 11-bit mantissa+sign+round artefact,
  // then re-pack with exponent (15 - e_b + 14_adj). For simplicity and
  // accuracy we store full fp16 of (1.0 / 2.0^15 / mantissa_value) and
  // adjust at lookup time by shifting the biased exponent.
  //
  // Implementation: store exp_for_mant[f] = fp16(1/(1+f/1024)) — value
  // in (0.5, 1.0]. Its biased exponent is therefore 14 (or 15 when f=0
  // and result is exactly 1.0). To get 1/s, we do:
  //   m   = mantissa(s); e_b = biased_exp(s)
  //   r   = exp_for_mant[m]   // ≈ 1/(1+m/1024), in (0.5, 1.0]
  //   1/s = r * 2^(15 - e_b) * 2^(-( -? )) — algebraically:
  //     s     = (1 + m/1024) * 2^(e_b - 15)
  //     1/s   = r * 2^(15 - e_b)
  //     fp16(1/s) = fp16(r) with exponent shifted by (15 - e_b - exp_of_r)
  //   r's true value is in (0.5, 1.0], so as fp16 it has biased exp 14
  //   (true exp -1), unless f==0 in which case r=1.0 (biased exp 15).
  // Final packed biased exp = r_biased + (15 - e_b - 15) = r_biased - e_b.
  //                          = (14 or 15) - e_b + 15 - 15 → simplify:
  // Let r_biased = exp_for_mant[f][14:10]. Then
  //     new_biased = r_biased + 15 - e_b
  // because r represents 2^(r_biased - 15) * (1.r_frac), and we want
  // 1/s = r * 2^(15 - e_b) → total exponent factor 2^(r_biased - 15 + 15 - e_b)
  // = 2^(r_biased - e_b). The "+15" bias gives new_biased = r_biased - e_b + 15.
  logic [15:0] recip_mant_lut [0:1023];

  initial begin : g_luts
    real d, ev, mv, rv;
    int  i;
    for (i = 0; i < 1024; i = i + 1) begin
      d  = real'(i) / 64.0;
      ev = $exp(-d);
      exp_lut[i] = real_to_fp16(ev);
    end
    for (i = 0; i < 1024; i = i + 1) begin
      mv = 1.0 + real'(i) / 1024.0;
      rv = 1.0 / mv;
      recip_mant_lut[i] = real_to_fp16(rv);
    end
  end

  // Round-to-nearest-even fp16 from real. Used only in initial blocks.
  function automatic logic [15:0] real_to_fp16(input real v);
    logic        s;
    real         av;
    int          e;
    real         m;
    int          biased;
    logic [9:0]  frac10;
    real         scaled;
    longint      iscaled;
    real         midpoint;
    longint      mant_int;
    begin
      if (v != v) return 16'h7E00;             // NaN
      s = (v < 0.0);
      av = s ? -v : v;
      if (av == 0.0) return {s, 15'd0};
      // Saturation to fp16 max ~65504 → Inf
      if (av >= 65520.0) return {s, 5'b11111, 10'b0};
      // Decompose: find e such that 2^e <= av < 2^(e+1).
      e = 0;
      m = av;
      if (m >= 1.0) begin
        while (m >= 2.0) begin m = m / 2.0; e = e + 1; end
      end else begin
        while (m < 1.0 && e > -30) begin m = m * 2.0; e = e - 1; end
      end
      biased = e + 15;
      if (biased >= 31) return {s, 5'b11111, 10'b0};
      if (biased <= 0) begin
        // subnormal: value = mant_int * 2^-24
        // av = mant_int / 2^24 → mant_int = av * 2^24, round-to-nearest-even
        scaled = av * (1.0 * (1 << 24));
        iscaled = longint'($rtoi(scaled));
        // Round half to even
        if ((scaled - real'(iscaled)) > 0.5) iscaled = iscaled + 1;
        else if ((scaled - real'(iscaled)) == 0.5 && ((iscaled & 64'sd1) != 0)) iscaled = iscaled + 1;
        if (iscaled >= 1024) begin
          // promoted to smallest normal
          return {s, 5'd1, 10'd0};
        end
        return {s, 5'd0, 10'(iscaled[9:0])};
      end
      // Normal. m in [1, 2). Mantissa = (m - 1) * 1024, RNE.
      scaled = (m - 1.0) * 1024.0;
      iscaled = longint'($rtoi(scaled));
      midpoint = real'(iscaled) + 0.5;
      mant_int = iscaled;
      if (scaled > midpoint) mant_int = iscaled + 1;
      else if (scaled == midpoint && ((iscaled & 64'sd1) != 0)) mant_int = iscaled + 1;
      else if (scaled < midpoint) mant_int = iscaled;
      // verilator coverage_off
      if (mant_int >= 1024) begin
        biased = biased + 1;
        mant_int = 0;
        if (biased >= 31) return {s, 5'b11111, 10'b0};
      end
      // verilator coverage_on
      frac10 = 10'(mant_int[9:0]);
      return {s, 5'(biased[4:0]), frac10};
    end
  endfunction

  // ───────────────────────── pipeline ────────────────────────────

  // S0: input register
  logic        v0;
  logic [15:0] x_q [16];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v0 <= 1'b0;
      for (int i = 0; i < 16; i++) x_q[i] <= 16'h0;
    end else begin
      v0 <= valid_i;
      for (int i = 0; i < 16; i++) x_q[i] <= x_i[i];
    end
  end

  // ── max tree (4 stages) ───────────────────────────────────────
  logic [15:0] m1 [8];
  logic [15:0] m2 [4];
  logic [15:0] m3 [2];
  logic [15:0] m4;
  logic        v1, v2, v3, v4;

  // Also carry x through the max tree so we can compute (m - x) at S5.
  logic [15:0] x1 [16], x2 [16], x3 [16], x4 [16];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0;
      for (int i = 0; i < 8;  i++) m1[i] <= 16'h0;
      for (int i = 0; i < 4;  i++) m2[i] <= 16'h0;
      for (int i = 0; i < 2;  i++) m3[i] <= 16'h0;
      m4 <= 16'h0;
      for (int i = 0; i < 16; i++) begin
        x1[i] <= 16'h0; x2[i] <= 16'h0; x3[i] <= 16'h0; x4[i] <= 16'h0;
      end
    end else begin
      v1 <= v0;
      for (int i = 0; i < 8; i++) m1[i] <= fp16_max(x_q[2*i], x_q[2*i+1]);
      for (int i = 0; i < 16; i++) x1[i] <= x_q[i];

      v2 <= v1;
      for (int i = 0; i < 4; i++) m2[i] <= fp16_max(m1[2*i], m1[2*i+1]);
      for (int i = 0; i < 16; i++) x2[i] <= x1[i];

      v3 <= v2;
      for (int i = 0; i < 2; i++) m3[i] <= fp16_max(m2[2*i], m2[2*i+1]);
      for (int i = 0; i < 16; i++) x3[i] <= x2[i];

      v4 <= v3;
      m4 <= fp16_max(m3[0], m3[1]);
      for (int i = 0; i < 16; i++) x4[i] <= x3[i];
    end
  end

  // ── S5: d[i] = m - x[i] (clamped ≥ 0); look up exp(-d). ────────
  // d is exactly ≥ 0 because m = max(x). We compute idx = round(d * 16),
  // clamp to [0, 255], and use exp_lut[idx]. For exact 0 the index is 0,
  // exp_lut[0] = fp16(1.0), which is what we want.
  /* verilator lint_off UNUSEDSIGNAL */
  function automatic logic [9:0] d_to_index(input logic [15:0] d_fp16);
    // d is non-negative fp16. Compute idx ≈ round(d * 64) clamped to
    // [0, 1023]. Uses a 32-bit fixed-point intermediate where 1.0 sits
    // at bit 16; idx = round(scaled / 2^10).
    logic [4:0]  eb;
    logic [9:0]  f;
    int          ev;
    logic [10:0] sig11;
    int          shift;
    logic [31:0] scaled;
    logic [31:0] rounded;
    int          idx_int;
    begin
      eb = d_fp16[14:10];
      f  = d_fp16[9:0];
      if (eb == 5'd0)  return 10'd0;
      if (eb == 5'd31) return 10'd1023;
      ev = int'(eb) - 15;
      sig11 = {1'b1, f};
      // scaled = d * 2^16 ⇒ shift = 16 + ev - 10 = 6 + ev.
      shift = 6 + ev;
      if (shift > 24) return 10'd1023;       // d very large
      if (shift < -16) return 10'd0;         // d very small
      if (shift >= 0) begin
        scaled = {21'd0, sig11} << shift;
      end else begin
        scaled = {21'd0, sig11} >> (-shift);
      end
      // idx = round_to_nearest(d * 64) = round_to_nearest(scaled / 2^10).
      rounded = scaled + 32'd512;            // round-half-up; bias < 1 ULP of idx
      idx_int = int'(rounded >> 10);
      if (idx_int > 1023) idx_int = 1023;
      if (idx_int < 0)    idx_int = 0;
      return 10'(idx_int);
    end
  endfunction
  /* verilator lint_on UNUSEDSIGNAL */

  logic [15:0] e5 [16];
  logic        v5;
  // (m - x) via fp16 add of m + (-x).
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v5 <= 1'b0;
      for (int i = 0; i < 16; i++) e5[i] <= 16'h0;
    end else begin
      v5 <= v4;
      for (int i = 0; i < 16; i++) begin
        logic [15:0] d;
        logic [9:0]  idx;
        d   = fp16_add(m4, fp16_neg(x4[i]));
        // Clamp negative results (numerical noise) to 0 by forcing sign bit clear.
        if (d[15]) d = 16'h0000;
        idx = d_to_index(d);
        e5[i] <= exp_lut[idx];
      end
    end
  end

  // ── sum tree (4 stages) ───────────────────────────────────────
  logic [15:0] s6 [8];
  logic [15:0] s7 [4];
  logic [15:0] s8 [2];
  logic [15:0] s9;
  logic        v6, v7, v8, v9;
  // Carry e along the sum tree.
  logic [15:0] e6 [16], e7 [16], e8 [16], e9 [16];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v6 <= 1'b0; v7 <= 1'b0; v8 <= 1'b0; v9 <= 1'b0;
      for (int i = 0; i < 8;  i++) s6[i] <= 16'h0;
      for (int i = 0; i < 4;  i++) s7[i] <= 16'h0;
      for (int i = 0; i < 2;  i++) s8[i] <= 16'h0;
      s9 <= 16'h0;
      for (int i = 0; i < 16; i++) begin
        e6[i] <= 16'h0; e7[i] <= 16'h0; e8[i] <= 16'h0; e9[i] <= 16'h0;
      end
    end else begin
      v6 <= v5;
      for (int i = 0; i < 8; i++) s6[i] <= fp16_add(e5[2*i], e5[2*i+1]);
      for (int i = 0; i < 16; i++) e6[i] <= e5[i];

      v7 <= v6;
      for (int i = 0; i < 4; i++) s7[i] <= fp16_add(s6[2*i], s6[2*i+1]);
      for (int i = 0; i < 16; i++) e7[i] <= e6[i];

      v8 <= v7;
      for (int i = 0; i < 2; i++) s8[i] <= fp16_add(s7[2*i], s7[2*i+1]);
      for (int i = 0; i < 16; i++) e8[i] <= e7[i];

      v9 <= v8;
      s9 <= fp16_add(s8[0], s8[1]);
      for (int i = 0; i < 16; i++) e9[i] <= e8[i];
    end
  end

  // ── S10: reciprocal lookup ────────────────────────────────────
  logic [15:0] inv_s10;
  logic [15:0] e10 [16];
  logic        v10;

  /* verilator lint_off UNUSEDSIGNAL */
  function automatic logic [15:0] fp16_recip(input logic [15:0] s);
    logic [4:0] eb;
    logic [9:0] f;
    logic [15:0] r;
    int         new_biased;
    int         r_biased;
    logic [9:0] r_frac;
    begin
      eb = s[14:10];
      f  = s[9:0];
      // s should be positive normal (sum of 16 non-negative fp16 numbers
      // each in [0,1]; sum ≥ exp(0)=1 when one entry is the max). If
      // somehow s is 0 or subnormal, saturate inv_s to +Inf → softmax
      // would emit NaN-ish results; documented behaviour.
      if (eb == 5'd0) return 16'h7BFF;        // ~max fp16
      if (eb == 5'd31) return 16'h0000;
      r        = recip_mant_lut[f];
      r_biased = int'(r[14:10]);
      r_frac   = r[9:0];
      // new biased = r_biased + 15 - eb
      new_biased = r_biased + 15 - int'(eb);
      if (new_biased >= 31) return {1'b0, 5'b11111, 10'b0};
      if (new_biased <= 0)  return {1'b0, 5'd0, r_frac};
      return {1'b0, 5'(new_biased[4:0]), r_frac};
    end
  endfunction
  /* verilator lint_on UNUSEDSIGNAL */

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v10     <= 1'b0;
      inv_s10 <= 16'h0;
      for (int i = 0; i < 16; i++) e10[i] <= 16'h0;
    end else begin
      v10     <= v9;
      inv_s10 <= fp16_recip(s9);
      for (int i = 0; i < 16; i++) e10[i] <= e9[i];
    end
  end

  // ── S11: per-lane fp16_fma(e[i], inv_s, 0) (fp16_fma is itself
  //        registered, adding FMA_LAT cycles).
  // Leaf-IP latency — mirror of fp16_lat_pkg (do NOT import it here). The
  // output valid must be a FMA_LAT-deep shift register from v10 so valid_o
  // asserts exactly when y_o (the fp16_fma output) is valid.
  localparam int unsigned FMA_LAT = 5;   // == fp16_lat_pkg::FP16_FMA_LAT
  logic [FMA_LAT-1:0] v11;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) v11 <= '0;
    else         v11 <= {v11[FMA_LAT-2:0], v10};
  end

  generate
    for (genvar gi = 0; gi < 16; gi = gi + 1) begin : g_mul
      fp16_fma u_mul (
        .clk_i  (clk_i),
        .rst_ni (rst_ni),
        .a_i    (e10[gi]),
        .b_i    (inv_s10),
        .c_i    (16'h0000),
        .y_o    (y_o[gi])
      );
    end
  endgenerate

  assign valid_o = v11[FMA_LAT-1];

endmodule
