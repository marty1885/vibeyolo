// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// fp16_fma_ref — behavioural golden for fp16_fma.
//
// Independent algorithm:
//   1. Unpack each fp16 to a pair (sig, exp). Use integer-only form
//      where value = sig * 2^exp, with sig held as a 24-bit unsigned so
//      that fp16 fits with room to spare (subnormals natural).
//   2. The product (24b*24b → 48b) is exact; the smaller of (product, c)
//      is shifted right with sticky tracking until both share a common
//      exponent, then added or subtracted in a 128-bit signed
//      accumulator. (Different anchor strategy from the DUT — DUT
//      anchors on the larger exponent, REF normalises by shifting the
//      smaller down.)
//   3. Normalise the absolute value via a downward exponent scan rather
//      than a leading-zero count.
//   4. Apply RNE rounding via guard+round+sticky.
//   5. Repack, handling special values at entry and exit.

module fp16_fma_ref (
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic [15:0] a_i,
  input  logic [15:0] b_i,
  input  logic [15:0] c_i,

  output logic [15:0] y_o
);

  function automatic logic [15:0] fma16(
      input logic [15:0] a,
      input logic [15:0] b,
      input logic [15:0] c
  );
    // ── unpack helpers (inline) ──────────────────────────────
    logic        sa, sb, sc;
    logic [4:0]  ea_b, eb_b, ec_b;
    logic [9:0]  ma_f, mb_f, mc_f;
    logic        a_z, b_z, c_z;
    logic        a_inf, b_inf, c_inf;
    logic        a_nan, b_nan, c_nan;

    // significand (24-bit, integer form), unbiased exponent
    logic [23:0] siga, sigb, sigc;
    int          ea, eb, ec;

    // product
    logic        sp;
    logic [47:0] sigprod;
    int          eprod;

    // alignment
    logic [127:0] big_p, big_c;
    int           anchor;       // common exponent for both = min(eprod, ec)
    int           shamt;
    logic [127:0] tmp;
    int           sh;

    // add/sub
    logic [127:0] big_p_a, big_c_a;
    logic [127:0] sum_mag;
    logic         result_sign;
    logic         ge;

    // normalisation + extraction
    int           msb_pos;
    int           biased_exp;
    int           shift_to_bit10;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [23:0]  win;
    /* verilator lint_on UNUSEDSIGNAL */
    logic         guard, round_bit, sticky;
    logic [9:0]   mant10;
    logic         lsb;
    logic         round_up;
    logic [10:0]  mant_r;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [127:0] tmp_shr;
    /* verilator lint_on UNUSEDSIGNAL */

    begin
      sa = a[15]; ea_b = a[14:10]; ma_f = a[9:0];
      sb = b[15]; eb_b = b[14:10]; mb_f = b[9:0];
      sc = c[15]; ec_b = c[14:10]; mc_f = c[9:0];

      a_z   = (ea_b == 0) && (ma_f == 0);
      b_z   = (eb_b == 0) && (mb_f == 0);
      c_z   = (ec_b == 0) && (mc_f == 0);
      a_inf = (ea_b == 31) && (ma_f == 0);
      b_inf = (eb_b == 31) && (mb_f == 0);
      c_inf = (ec_b == 31) && (mc_f == 0);
      a_nan = (ea_b == 31) && (ma_f != 0);
      b_nan = (eb_b == 31) && (mb_f != 0);
      c_nan = (ec_b == 31) && (mc_f != 0);

      // ── special-case bail-out ──
      if (a_nan || b_nan || c_nan) return 16'h7E00;
      if ((a_inf && b_z) || (b_inf && a_z)) return 16'h7E00;  // 0*Inf
      if ((a_inf || b_inf) && c_inf && (sa ^ sb) != sc) return 16'h7E00;
      if (a_inf || b_inf) return {(sa ^ sb), 5'b11111, 10'b0};
      if (c_inf)          return {sc,        5'b11111, 10'b0};

      // ── unpack to (sig24, exp), with sig in a 24-bit window so that
      // the implicit-1 (if any) sits at bit 23 for normals, lower for
      // subnormals.
      if (a_z) begin
        siga = 24'd0; ea = 0;
      end else if (ea_b == 0) begin
        // subnormal: shift frac up so the leading frac bit is at bit 23
        // — actually we leave it un-normalised; downstream handles it.
        // Place frac at bits [22:13] (leaving room for "implicit 0" at
        // bit 23). Then exp such that value = sig * 2^exp:
        //   subnormal value = frac * 2^(-24).
        //   With sig = frac << 13, exp = -24 - 13 = -37.
        siga = {1'b0, ma_f, 13'd0};
        ea   = -37;
      end else begin
        // normal: implicit-1 at bit 23 → sig = {1, frac, 13'b0}.
        //   value = (1024+frac)/1024 * 2^(ea_b - 15)
        //         = (1024+frac) * 2^(ea_b - 25)
        //         = sig (which is (1024+frac)<<13) * 2^(ea_b - 38)
        siga = {1'b1, ma_f, 13'd0};
        ea   = int'(ea_b) - 38;
      end

      if (b_z) begin
        sigb = 24'd0; eb = 0;
      end else if (eb_b == 0) begin
        sigb = {1'b0, mb_f, 13'd0};
        eb   = -37;
      end else begin
        sigb = {1'b1, mb_f, 13'd0};
        eb   = int'(eb_b) - 38;
      end

      if (c_z) begin
        sigc = 24'd0; ec = 0;
      end else if (ec_b == 0) begin
        sigc = {1'b0, mc_f, 13'd0};
        ec   = -37;
      end else begin
        sigc = {1'b1, mc_f, 13'd0};
        ec   = int'(ec_b) - 38;
      end

      // ── product ──
      sp      = sa ^ sb;
      sigprod = sigb * siga;     // exact 48-bit
      eprod   = ea + eb;

      // If product underflows to zero (both inputs subnormal and product
      // truly tiny — sigprod==0 captures it).
      // If sigprod == 0, treat product magnitude as 0.

      // ── alignment: common exponent = min(eprod, ec) ──
      // Place big_p = sigprod << (eprod - anchor) in 128 bits.
      // Place big_c = sigc    << (ec    - anchor) in 128 bits.
      // We track sticky for the operand that we shift right (if any).
      // But here we only shift left (since anchor is the min), so no
      // bits are lost; the values are exact integers in 128-bit space.
      // We do still need sticky if shift exceeds 128 — but with our
      // exponent ranges that doesn't happen.
      //
      // Exponent ranges:
      //   eprod = ea + eb. ea in [-37, -8] for normal/subnormal (ea_b
      //   in [1, 30]). Min eprod = -74 (both subnormal). Max eprod = -16
      //   (both normal max). Subnormals shift sig down so the effective
      //   value is still right.
      //   ec in [-37, -8] similarly.
      //   anchor = min(eprod, ec) >= -74.
      //   shamt for product = eprod - anchor in [0, 66]. <<66 of a 48-bit
      //   value yields a 114-bit value — fits in 128.
      //   shamt for c       = ec    - anchor in [0, 66]. <<66 of 24-bit
      //   yields 90 bits — fits.
      //
      // Sticky bits start as 0; we re-introduce sticky concept only at
      // final extraction (since both operands sit exactly in big_*).

      if (sigprod == 0 && sigc == 0) begin
        return 16'h0000;   // both contributions zero → +0
      end

      // Determine anchor without underflowing if one operand is zero.
      if (sigprod == 0)      anchor = ec;
      else if (sigc == 0)    anchor = eprod;
      else if (eprod < ec)   anchor = eprod;
      else                   anchor = ec;

      shamt = eprod - anchor;
      tmp   = {80'd0, sigprod};
      if (sigprod == 0) big_p = 128'd0;
      else begin
        sh = shamt;
        if (sh > 100) sh = 100;
        big_p = tmp << sh;
      end

      shamt = ec - anchor;
      tmp   = {104'd0, sigc};
      if (sigc == 0) big_c = 128'd0;
      else begin
        sh = shamt;
        if (sh > 100) sh = 100;
        big_c = tmp << sh;
      end

      big_p_a = big_p;
      big_c_a = big_c;

      // ── add or subtract magnitudes ──
      if (sp == sc) begin
        sum_mag     = big_p_a + big_c_a;
        result_sign = sp;
      end else begin
        ge = (big_p_a >= big_c_a);
        if (ge) begin
          sum_mag     = big_p_a - big_c_a;
          result_sign = sp;
        end else begin
          sum_mag     = big_c_a - big_p_a;
          result_sign = sc;
        end
      end

      if (sum_mag == 128'd0) return 16'h0000;   // exact zero (RNE → +0)

      // ── normalise via downward scan ──
      // The exact magnitude is sum_mag * 2^anchor. Find MSB position.
      msb_pos = 0;
      for (int i = 127; i >= 0; i--) begin
        if (sum_mag[i] && (msb_pos == 0)) msb_pos = i;
      end
      // (If only bit 0 is set, msb_pos stays 0 → that's correct.)

      // Unbiased exponent of the result = anchor + msb_pos. fp16 bias 15.
      biased_exp = anchor + msb_pos + 15;

      // We want a normal mantissa 1.xxx at bit 10 of a 11-bit window.
      // i.e., place the MSB (the implicit-1) at bit 10. So shift sum_mag
      // right by (msb_pos - 10). Bits below the resulting window are
      // guard+round+sticky.

      if (biased_exp >= 31) begin
        // overflow → ±Inf
        return {result_sign, 5'b11111, 10'b0};
      end

      // If biased_exp <= 0, we have a subnormal output. The leading-1
      // must instead sit lower in the mantissa field; we shift by
      // (msb_pos - 10 + (1 - biased_exp)) to get there.
      if (biased_exp <= 0) begin
        shift_to_bit10 = msb_pos - 10 + (1 - biased_exp);
        biased_exp     = 0;
      end else begin
        shift_to_bit10 = msb_pos - 10;
      end

      if (shift_to_bit10 < 0) begin
        // Should be impossible (msb_pos >= 10 ensures we have ≥11 bits
        // of significand). For paranoia, left-shift to align.
        tmp_shr   = sum_mag << (-shift_to_bit10);
        win       = tmp_shr[23:0];
        guard     = 1'b0;
        round_bit = 1'b0;
        sticky    = 1'b0;
      end else if (shift_to_bit10 == 0) begin
        win       = sum_mag[23:0];
        guard     = 1'b0;
        round_bit = 1'b0;
        sticky    = 1'b0;
      end else if (shift_to_bit10 == 1) begin
        win       = sum_mag[24:1];
        guard     = sum_mag[0];
        round_bit = 1'b0;
        sticky    = 1'b0;
      end else if (shift_to_bit10 == 2) begin
        win       = sum_mag[25:2];
        guard     = sum_mag[1];
        round_bit = sum_mag[0];
        sticky    = 1'b0;
      end else begin
        // shift >= 3: pull 24-bit window, guard, round, and OR-reduce
        // anything below for sticky.
        tmp_shr   = sum_mag >> shift_to_bit10;
        win       = tmp_shr[23:0];
        tmp_shr   = sum_mag >> (shift_to_bit10 - 1);
        guard     = tmp_shr[0];
        tmp_shr   = sum_mag >> (shift_to_bit10 - 2);
        round_bit = tmp_shr[0];
        sticky    = 1'b0;
        for (int j = 0; j < 128; j++) begin
          if (j < shift_to_bit10 - 2 && sum_mag[j]) sticky = 1'b1;
        end
      end

      mant10 = win[9:0];
      lsb    = mant10[0];

      // Effective guard for RNE = `guard`, sticky for RNE = round|sticky.
      round_up = guard && ((round_bit | sticky) || lsb);

      mant_r = {1'b0, mant10} + 11'(round_up);

      // mant_r[10] is the carry out of the mantissa (also: in subnormal
      // path this represents promotion to the smallest normal).
      if (mant_r[10]) begin
        biased_exp = biased_exp + 1;
        mant_r     = 11'b0;
      end

      if (biased_exp >= 31) return {result_sign, 5'b11111, 10'b0};
      if (biased_exp <= 0) begin
        // True subnormal output; biased_exp_field = 0, mantissa = mant_r.
        return {result_sign, 5'd0, mant_r[9:0]};
      end
      return {result_sign, 5'(biased_exp), mant_r[9:0]};
    end
  endfunction

  logic [15:0] y_d;
  assign y_d = fma16(a_i, b_i, c_i);

  // Five registered stages so the behavioral golden matches the DUT's
  // 5-cycle latency (fp16_lat_pkg::FP16_FMA_LAT) cycle-for-cycle. The
  // compute stays fully combinational — only the output is delayed.
  logic [15:0] y_s1, y_s2, y_s3, y_s4;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      y_s1 <= 16'h0000;
      y_s2 <= 16'h0000;
      y_s3 <= 16'h0000;
      y_s4 <= 16'h0000;
      y_o  <= 16'h0000;
    end else begin
      y_s1 <= y_d;
      y_s2 <= y_s1;
      y_s3 <= y_s2;
      y_s4 <= y_s3;
      y_o  <= y_s4;
    end
  end

endmodule
