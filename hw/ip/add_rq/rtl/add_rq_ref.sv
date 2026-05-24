// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// add_rq_ref — independent behavioural golden for add_rq.
//
// Implementation strategy (deliberately different from the DUT):
//
//   * Unpack each fp16 to a SystemVerilog `real` (binary64) via a
//     pure-function fp16→fp64 cast that drives the underlying 64-bit
//     pattern explicitly. binary64 has 52 mantissa bits, far more than
//     fp16 needs, so every intermediate fp16 value fits exactly; the
//     only places where rounding happens are the five fp16-result
//     repacks (one per FMA stage in the DUT). That matches the DUT's
//     five rounding points.
//
//   * Each stage performs the same arithmetic at fp64 precision then
//     repacks to fp16 (RNE, with subnormal + Inf + NaN handling). This
//     is a fundamentally different code path from the DUT (which uses
//     bit-level integer alignment in 100-bit windows) and from the
//     fp16_fma_ref (different anchor + downward scan in 128-bit), so a
//     shared bug across the three is implausible.
//
//   * The final fp16→i8_sat stage uses fp64 RNE then saturation,
//     implemented with integer arithmetic on the unpacked
//     significand+exponent so it does not depend on host floor/ceil.
//
//   * Output is registered to match the DUT's pipeline latency. TOTAL_LAT
//     tracks the DUT: I2F_LAT(2) + FMA_LAT(3) + FMA_LAT(3) + FMA_LAT(3) +
//     SAT_LAT(1) = 12 cycles.

module add_rq_ref (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               valid_i,
  input  logic signed  [7:0] a_i8_i,
  input  logic signed  [7:0] b_i8_i,
  input  logic        [15:0] scale_a_fp16_i,
  input  logic        [15:0] scale_b_fp16_i,
  input  logic        [15:0] inv_out_scale_fp16_i,
  input  logic        [15:0] bias_fp16_i,

  output logic               valid_o,
  output logic signed  [7:0] y_o
);

  // ── fp16 → real (exact) ──────────────────────────────────────
  function automatic real fp16_to_r(input logic [15:0] x);
    logic        s;
    logic [4:0]  e;
    logic [9:0]  m;
    logic [63:0] bits;
    int          unbiased;
    int          lz;
    int          shift;
    logic [9:0]  m_shifted;
    logic [51:0] frac64;
    /* verilator lint_off UNUSEDSIGNAL */
    int          exp64;
    /* verilator lint_on UNUSEDSIGNAL */
    begin
      s = x[15];
      e = x[14:10];
      m = x[9:0];

      if (e == 5'd31) begin
        if (m == 10'd0) begin
          bits = {s, 11'h7FF, 52'd0};
        end else begin
          // qNaN
          bits = {s, 11'h7FF, 1'b1, 51'd0};
        end
        return $bitstoreal(bits);
      end

      if (e == 5'd0 && m == 10'd0) begin
        bits = {s, 63'd0};
        return $bitstoreal(bits);
      end

      if (e == 5'd0) begin
        // subnormal fp16: value = m * 2^-24, normalise.
        lz = 0;
        for (int i = 9; i >= 0; i--) begin
          if (m[i] && (lz == 0) && (i != 9)) lz = 9 - i;
          else if (m[i] && (i == 9)) lz = 0;
        end
        // Recompute lz cleanly:
        lz = 10;
        for (int i = 9; i >= 0; i--) begin
          if (m[i] && (lz == 10)) lz = 9 - i;
        end
        // lz in [0..9]; leading 1 is at bit (9 - lz).
        shift     = lz + 1;
        m_shifted = m << shift;
        unbiased  = -14 - lz;
        exp64     = unbiased + 1023;
        frac64    = {m_shifted[9:0], 42'd0};
      end else begin
        unbiased = int'(e) - 15;
        exp64    = unbiased + 1023;
        frac64   = {m, 42'd0};
      end

      bits = {s, 11'(exp64), frac64};
      return $bitstoreal(bits);
    end
  endfunction

  // ── real → fp16 with RNE, sub/Inf/NaN handling ───────────────
  function automatic logic [15:0] r_to_fp16(input real v);
    logic [63:0] bits;
    logic        s;
    logic [10:0] e11;
    logic [51:0] f52;
    int          unbiased;
    logic [52:0] sig53;        // implicit 1 + 52 frac
    int          biased;
    int          shift;
    logic [9:0]  mant10;
    logic        guard, round_b, sticky;
    logic [10:0] mant_r;
    int          j_lim;
    begin
      bits = $realtobits(v);
      s   = bits[63];
      e11 = bits[62:52];
      f52 = bits[51:0];

      if (e11 == 11'h7FF) begin
        if (f52 == 52'd0) return {s, 5'b11111, 10'b0};
        return 16'h7E00;
      end
      if (e11 == 11'd0 && f52 == 52'd0) return {s, 15'd0};

      if (e11 == 11'd0) begin
        // subnormal fp64 ≪ fp16 subnormal min — flush.
        return {s, 15'd0};
      end

      unbiased = int'(e11) - 1023;
      sig53    = {1'b1, f52};   // implicit 1 at bit 52

      if (unbiased > 15) return {s, 5'b11111, 10'b0};

      if (unbiased >= -14) begin
        shift  = 42;             // bring bit 52 down to bit 10
        biased = unbiased + 15;
      end else begin
        shift  = 42 + (-14 - unbiased);
        biased = 0;
      end

      if (shift >= 53) begin
        mant10 = 10'd0;
        guard  = 1'b0;
        round_b = 1'b0;
        sticky = (sig53 != 53'd0);
      end else begin
        // shift is in [42..52] (worst case is fp16 min subnormal where
        // unbiased = -24 → shift = 42 + 10 = 52). Use general path.
        mant10  = 10'((sig53 >> shift) & 53'h3FF);
        guard   = (shift >= 1) ? sig53[shift-1] : 1'b0;
        round_b = (shift >= 2) ? sig53[shift-2] : 1'b0;
        sticky  = 1'b0;
        j_lim   = shift - 2;
        for (int j = 0; j < 53; j++) begin
          if (j < j_lim && sig53[j]) sticky = 1'b1;
        end
      end

      mant_r = {1'b0, mant10} + 11'(guard && ((round_b | sticky) || mant10[0]));

      if (mant_r[10]) begin
        biased += 1;
        mant_r  = 11'd0;
      end

      if (biased >= 31) return {s, 5'b11111, 10'b0};
      if (biased <= 0)  return {s, 5'd0, mant_r[9:0]};
      return {s, 5'(biased), mant_r[9:0]};
    end
  endfunction

  // ── fp16 → int8 with RNE and saturation ──────────────────────
  // Uses real conversion + integer rounding (different code path
  // from the dedicated fp16_to_i8_sat block).
  function automatic logic signed [7:0] fp16_to_i8_sat_r(input logic [15:0] x);
    logic [4:0]  e;
    logic [9:0]  m;
    logic        s;
    real         v;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [63:0] bits;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [10:0] e11;
    logic [51:0] f52;
    int          unbiased;
    logic [52:0] sig53;
    int          shift;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [53:0] sig_shifted;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [31:0] int_part;
    logic        guard, sticky;
    logic        up;
    int          i_val;
    int          j_lim2;
    begin
      e = x[14:10];
      m = x[9:0];
      s = x[15];

      // Specials
      if (e == 5'd31) begin
        if (m != 10'd0) return 8'sd0;            // NaN → 0
        return s ? 8'sd128 /*=-128*/ : 8'sd127;  // ±Inf → sat
      end
      if (e == 5'd0 && m == 10'd0) return 8'sd0;

      v    = fp16_to_r(x);
      bits = $realtobits(v);
      e11  = bits[62:52];
      f52  = bits[51:0];

      unbiased = int'(e11) - 1023;
      sig53    = {1'b1, f52};

      // value = sig53 * 2^(unbiased - 52). Integer part lies in bits at
      // position >= (52 - unbiased). If unbiased < 0, value < 1, so int
      // part is 0 and we check the half-bit at position (51 - unbiased -? )
      // Simpler: choose shift so the unit bit lands at bit 0.
      //   unit position in sig53 = 52 - unbiased  (when unbiased <= 52)
      //   shift right by that to get integer part.
      if (unbiased >= 52) begin
        // huge values — definitely saturate
        return s ? 8'sd128 : 8'sd127;
      end
      if (unbiased < -1) begin
        // |v| < 0.5 → round to 0.
        return 8'sd0;
      end

      shift = 52 - unbiased;
      // Extract integer part, guard (the bit just below unit), sticky.
      if (shift == 0) begin
        sig_shifted = {1'b0, sig53};
        int_part    = sig_shifted[31:0];
        guard       = 1'b0;
        sticky      = 1'b0;
      end else begin
        int_part = 32'(sig53 >> shift);
        guard    = sig53[shift - 1];
        sticky   = 1'b0;
        j_lim2   = shift - 1;
        for (int j = 0; j < 53; j++) begin
          if (j < j_lim2 && sig53[j]) sticky = 1'b1;
        end
      end

      // RNE: round up if guard && (sticky || lsb)
      up = guard && (sticky || int_part[0]);
      i_val = int'(int_part) + (up ? 1 : 0);
      if (s) i_val = -i_val;

      if (i_val >  127) return 8'sd127;
      if (i_val < -128) return 8'sd128 /*=-128*/;
      return 8'(i_val);
    end
  endfunction

  // ── combinational compute ───────────────────────────────────
  function automatic logic signed [7:0] compute(
      input logic signed [7:0] a8,
      input logic signed [7:0] b8,
      input logic [15:0] sa,
      input logic [15:0] sb,
      input logic [15:0] inv_out,
      input logic [15:0] bias
  );
    real fa, fb, fsa, fsb, finv, fbias;
    logic [15:0] fa16, fb16, ta_fp, tb_fp, sum_fp, fy;
    real ta, tb, sum, fyv;
    int a_int, b_int;
    real a_r, b_r;
    begin
      // Stage 0: i8 → fp16 (a/b are exactly representable in fp16).
      a_int = int'(a8);
      b_int = int'(b8);
      a_r   = real'(a_int);
      b_r   = real'(b_int);
      fa16 = r_to_fp16(a_r);
      fb16 = r_to_fp16(b_r);
      fa = fp16_to_r(fa16);
      fb = fp16_to_r(fb16);

      fsa   = fp16_to_r(sa);
      fsb   = fp16_to_r(sb);
      finv  = fp16_to_r(inv_out);
      fbias = fp16_to_r(bias);

      // Stage 1: ta = fp16(fa * fsa), tb = fp16(fb * fsb)
      ta_fp = r_to_fp16(fa * fsa);
      tb_fp = r_to_fp16(fb * fsb);
      ta = fp16_to_r(ta_fp);
      tb = fp16_to_r(tb_fp);

      // Stage 2: sum = fp16(ta + tb)
      sum_fp = r_to_fp16(ta + tb);
      sum    = fp16_to_r(sum_fp);

      // Stage 3: fy = fp16(sum * inv_out + bias)
      fyv = sum * finv + fbias;
      fy  = r_to_fp16(fyv);

      // Stage 4: fp16 → i8 sat
      return fp16_to_i8_sat_r(fy);
    end
  endfunction

  logic signed [7:0] y_comb;
  assign y_comb = compute(a_i8_i, b_i8_i, scale_a_fp16_i, scale_b_fp16_i,
                          inv_out_scale_fp16_i, bias_fp16_i);

  // ── TOTAL_LAT-cycle pipeline to match DUT latency ───────────
  // == add_rq's I2F_LAT + 3*FMA_LAT + SAT_LAT = 2 + 9 + 1 = 12.
  localparam int unsigned TOTAL_LAT = 12;
  logic signed [7:0]    y_dl [TOTAL_LAT];
  logic [TOTAL_LAT-1:0] v_sr;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < TOTAL_LAT; i++) y_dl[i] <= 8'sd0;
      v_sr <= '0;
    end else begin
      y_dl[0] <= y_comb;
      for (int i = 1; i < TOTAL_LAT; i++) y_dl[i] <= y_dl[i-1];
      v_sr <= {v_sr[TOTAL_LAT-2:0], valid_i};
    end
  end

  assign y_o     = y_dl[TOTAL_LAT-1];
  assign valid_o = v_sr[TOTAL_LAT-1];

endmodule
