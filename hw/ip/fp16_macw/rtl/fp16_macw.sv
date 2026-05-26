// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// fp16_macw — mixed-precision fused multiply-add with a WIDE float
// accumulator: y = a*b + c, where
//   a_i, b_i : IEEE-754 binary16 (fp16)
//   c_i, y_o : a parameterized binary float with ACC_EXP exponent bits and
//              ACC_MANT stored mantissa bits (1 + ACC_EXP + ACC_MANT total).
//
// Purpose: flash_attn (and any long fp16 reduction) loses accuracy when the
// running sum is rounded back to fp16's 10-bit mantissa after every step.
// This block multiplies in fp16 but keeps the accumulate path wide, so a
// reduction only rounds to fp16 once at the very end. The accumulator width
// is configurable (ACC_EXP/ACC_MANT) to trade area against accuracy.
//
// Single rounding (RNE) to the wide format at the output. Subnormal wide
// outputs are emitted (flush only on total underflow). Overflow -> signed
// Inf. NaN propagates (canonical: exp all-ones, msb-frac set). a*b==-0 + +0
// -> +0 (RNE).
//
// LATENCY = 5 cycles, identical to fp16_fma (fp16_lat_pkg::FP16_FMA_LAT).
// Pinning the latency means a consumer that interleaves partial sums for the
// fp16_fma pipeline (e.g. flash_attn) keeps the *same* schedule when it swaps
// in this wide-accumulate cell — only the datatype widens.
//
// Pipeline (5 stages): s1 unpack+product+align(place_op); s2 add/sub; s3
// leading-one detect; s4 normalize barrel-shift; s5 subnormal-shift+round+pack.
// The two extra cuts vs the original 3-stage (after the add, and between the
// LZD and the normalize shift) break the serial leading-zero→shift chain that
// caps Fmax in 7nm; closes 1 GHz on ASAP7 SLVT. See reports/PDK_ASAP7.md.
//
// Datapath (max-anchor, mirrors the proven fp16_fma DUT, generalized):
//   1. Unpack a,b (fp16) and c (wide) to (sign, integer significand, eff_exp)
//      where value = sig * 2^eff_exp.
//   2. Exact product sig_p = sig_a*sig_b (22b), e_p = e_a+e_b.
//   3. Anchor on the operand with the larger top exponent. Place both
//      significands into a WIN-bit window by the uniform rule
//          s = eff_exp - anchor_top + (WIN-2)
//      so window bit b represents 2^(b - (WIN-2) + anchor_top). The dominant
//      MSB lands at bit WIN-2 (bit WIN-1 is the add carry). The lesser
//      operand right-shifts; dropped low bits set a sticky flag.
//   4. Same-sign add / different-sign subtract of the two windows.
//   5. Leading-one detect, normalize, extract ACC_MANT mantissa + guard +
//      round + sticky, RNE round, repack to the wide format.

module fp16_macw #(
  parameter int unsigned ACC_EXP  = 8,
  parameter int unsigned ACC_MANT = 21,
  // Derived — do NOT override at instantiation.
  parameter int unsigned ACC_W    = 1 + ACC_EXP + ACC_MANT
) (
  input  logic clk_i,
  input  logic rst_ni,

  input  logic [15:0]            a_i,
  input  logic [15:0]            b_i,
  input  logic [ACC_W-1:0]       c_i,

  output logic [ACC_W-1:0]       y_o
);

  // ───────────────────────── derived sizes ─────────────────────
  localparam int          ACC_BIAS = (1 << (ACC_EXP-1)) - 1;
  localparam int          PW      = 22;            // fp16*fp16 significand
  localparam int          CW      = int'(ACC_MANT) + 1; // c significand (impl-1 + frac)
  localparam int          MAXW    = (PW > CW) ? PW : CW;
  // Window: dominant significand at the top + a generous sticky/guard region
  // below for the lesser operand's alignment and round/sticky extraction.
  localparam int          WIN     = MAXW + 64;
  localparam int          EXPMAX  = (1 << ACC_EXP) - 1; // all-ones exp field

  // ───────────────────────── unpack fp16 ───────────────────────
  function automatic void unpack_fp16(
      input  logic [15:0] x,
      output logic        s,
      output logic [10:0] sig,
      output int          eff_exp,
      output logic        is_zero,
      output logic        is_inf,
      output logic        is_nan
  );
    logic [4:0] bexp; logic [9:0] frac;
    begin
      s = x[15]; bexp = x[14:10]; frac = x[9:0];
      is_zero = (bexp == 5'd0)  && (frac == 10'd0);
      is_inf  = (bexp == 5'd31) && (frac == 10'd0);
      is_nan  = (bexp == 5'd31) && (frac != 10'd0);
      if (bexp == 5'd0) begin sig = {1'b0, frac}; eff_exp = -24; end
      else              begin sig = {1'b1, frac}; eff_exp = int'(bexp) - 25; end
    end
  endfunction

  // ───────────────────────── unpack wide c ─────────────────────
  function automatic void unpack_wide(
      input  logic [ACC_W-1:0] x,
      output logic             s,
      output logic [CW-1:0]    sig,
      output int               eff_exp,
      output logic             is_zero,
      output logic             is_inf,
      output logic             is_nan
  );
    logic [ACC_EXP-1:0]  bexp;
    logic [ACC_MANT-1:0] frac;
    begin
      s    = x[ACC_W-1];
      bexp = x[ACC_W-2 -: ACC_EXP];
      frac = x[ACC_MANT-1:0];
      is_zero = (bexp == '0)            && (frac == '0);
      is_inf  = (bexp == EXPMAX[ACC_EXP-1:0]) && (frac == '0);
      is_nan  = (bexp == EXPMAX[ACC_EXP-1:0]) && (frac != '0);
      if (bexp == '0) begin
        sig     = {1'b0, frac};
        eff_exp = 1 - ACC_BIAS - int'(ACC_MANT);
      end else begin
        sig     = {1'b1, frac};
        eff_exp = int'(bexp) - ACC_BIAS - int'(ACC_MANT);
      end
    end
  endfunction

  // ───────────────────────── stage 0: unpack + product ─────────
  logic        sa, sb, sc;
  logic [10:0] siga, sigb;
  logic [CW-1:0] sigc;
  int          ea, eb, ec;
  logic        a_zero, a_inf, a_nan, b_zero, b_inf, b_nan, c_zero, c_inf, c_nan;

  always_comb begin
    unpack_fp16(a_i, sa, siga, ea, a_zero, a_inf, a_nan);
    unpack_fp16(b_i, sb, sigb, eb, b_zero, b_inf, b_nan);
    unpack_wide(c_i, sc, sigc, ec, c_zero, c_inf, c_nan);
  end

  logic        sp;
  logic [PW-1:0] sig_p;
  int          ep;
  assign sp    = sa ^ sb;
  assign sig_p = siga * sigb;
  assign ep    = ea + eb;

  logic prod_zero;
  assign prod_zero = a_zero || b_zero || (sig_p == '0);

  // specials
  logic prod_is_inf, prod_is_nan_special;
  assign prod_is_inf         = (a_inf || b_inf) && !(a_zero || b_zero) && !(a_nan || b_nan);
  assign prod_is_nan_special = (a_inf && b_zero) || (b_inf && a_zero);

  logic out_is_nan, out_is_inf, out_inf_sign;
  assign out_is_nan   = a_nan || b_nan || c_nan || prod_is_nan_special ||
                        (prod_is_inf && c_inf && (sp != sc));
  assign out_is_inf   = !out_is_nan && (prod_is_inf || c_inf);
  assign out_inf_sign = (c_inf && !prod_is_inf) ? sc : sp;

  // top exponents (exponent of each operand's MSB)
  int top_p, top_c, anchor_top;
  assign top_p = ep + (PW - 1);
  assign top_c = ec + (CW - 1);
  always_comb begin
    if (prod_zero && c_zero)      anchor_top = 0;
    else if (prod_zero)           anchor_top = top_c;
    else if (c_zero)              anchor_top = top_p;
    else                          anchor_top = (top_c > top_p) ? top_c : top_p;
  end

  // place an integer significand into the WIN-bit window. value = sig*2^e.
  // s = e - anchor_top + (WIN-2). Left-shift never overflows (dominant MSB
  // lands at WIN-2); right-shift captures dropped bits in sticky.
  function automatic void place_op(
      input  logic [WIN-1:0] sigext,    // significand, zero-extended to WIN
      input  int             e,
      input  int             at,
      input  logic           is_zero,
      output logic [WIN-1:0] big,
      output logic           sticky
  );
    int s; logic [WIN-1:0] dropped;
    begin
      big = '0; sticky = 1'b0;
      if (!is_zero) begin
        s = e - at + (WIN - 2);
        if (s >= 0) begin
          if (s < WIN) big = sigext << s;
          else         big = '0;             // (cannot happen for valid ops)
        end else begin
          if (-s < WIN) begin
            big     = sigext >> (-s);
            dropped = sigext << (WIN + s);   // bits shifted out below bit 0
            sticky  = (dropped != '0);
          end else begin
            big    = '0;
            sticky = (sigext != '0);
          end
        end
      end
    end
  endfunction

  logic [WIN-1:0] big_p, big_c;
  logic           stky_p, stky_c;
  always_comb begin
    place_op(WIN'(sig_p), ep, anchor_top, prod_zero, big_p, stky_p);
    place_op(WIN'(sigc),  ec, anchor_top, c_zero,    big_c, stky_c);
  end

  // ───────────────────── stage-1 pipeline register ─────────────
  logic [WIN-1:0] s1_big_p, s1_big_c;
  logic           s1_sp, s1_sc, s1_sticky_pre;
  int             s1_anchor_top;
  logic           s1_out_is_nan, s1_out_is_inf, s1_out_inf_sign;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s1_big_p <= '0; s1_big_c <= '0; s1_sp <= 1'b0; s1_sc <= 1'b0;
      s1_sticky_pre <= 1'b0; s1_anchor_top <= 0;
      s1_out_is_nan <= 1'b0; s1_out_is_inf <= 1'b0; s1_out_inf_sign <= 1'b0;
    end else begin
      s1_big_p <= big_p; s1_big_c <= big_c; s1_sp <= sp; s1_sc <= sc;
      s1_sticky_pre <= stky_p | stky_c; s1_anchor_top <= anchor_top;
      s1_out_is_nan <= out_is_nan; s1_out_is_inf <= out_is_inf;
      s1_out_inf_sign <= out_inf_sign;
    end
  end

  // ───────────────────── add / subtract magnitudes ─────────────
  logic same_sign, ge_pc;
  assign same_sign = (s1_sp == s1_sc);
  assign ge_pc     = (s1_big_p >= s1_big_c);

  logic [WIN:0] mag_raw;          // +1 bit for same-sign carry
  logic         result_sign;
  always_comb begin
    if (same_sign) begin
      mag_raw     = {1'b0, s1_big_p} + {1'b0, s1_big_c};
      result_sign = s1_sp;
    end else begin
      mag_raw     = ge_pc ? {1'b0, (s1_big_p - s1_big_c)}
                          : {1'b0, (s1_big_c - s1_big_p)};
      result_sign = ge_pc ? s1_sp : s1_sc;
    end
  end

  // ── CUT B (after add): register mag_raw before the LZD. LATENCY +1.
  logic [WIN:0]   s1b_mag_raw;
  logic           s1b_result_sign, s1b_sticky_pre;
  int             s1b_anchor_top;
  logic           s1b_out_is_nan, s1b_out_is_inf, s1b_out_inf_sign;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s1b_mag_raw <= '0; s1b_result_sign <= 1'b0; s1b_sticky_pre <= 1'b0;
      s1b_anchor_top <= 0;
      s1b_out_is_nan <= 1'b0; s1b_out_is_inf <= 1'b0; s1b_out_inf_sign <= 1'b0;
    end else begin
      s1b_mag_raw <= mag_raw; s1b_result_sign <= result_sign;
      s1b_sticky_pre <= s1_sticky_pre; s1b_anchor_top <= s1_anchor_top;
      s1b_out_is_nan <= s1_out_is_nan; s1b_out_is_inf <= s1_out_is_inf;
      s1b_out_inf_sign <= s1_out_inf_sign;
    end
  end

  // leading-one detect over the (WIN+1)-bit magnitude
  int  msb_pos; logic any_one;
  always_comb begin
    any_one = 1'b0; msb_pos = 0;
    for (int i = WIN; i >= 0; i--)
      if (s1b_mag_raw[i] && !any_one) begin any_one = 1'b1; msb_pos = i; end
  end

  // ── CUT C (after LZD): register the leading-zero result + mag_raw, BEFORE
  //    the normalize barrel-shift. This breaks the LZD->shift serial chain
  //    that the 7nm timing shows is the wall. LATENCY +1.
  logic [WIN:0]   s1c_mag_raw;
  int             s1c_msb_pos, s1c_anchor_top;
  logic           s1c_any_one, s1c_result_sign, s1c_sticky_pre;
  logic           s1c_out_is_nan, s1c_out_is_inf, s1c_out_inf_sign;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s1c_mag_raw <= '0; s1c_msb_pos <= 0; s1c_anchor_top <= 0;
      s1c_any_one <= 1'b0; s1c_result_sign <= 1'b0; s1c_sticky_pre <= 1'b0;
      s1c_out_is_nan <= 1'b0; s1c_out_is_inf <= 1'b0; s1c_out_inf_sign <= 1'b0;
    end else begin
      s1c_mag_raw <= s1b_mag_raw; s1c_msb_pos <= msb_pos;
      s1c_anchor_top <= s1b_anchor_top; s1c_any_one <= any_one;
      s1c_result_sign <= s1b_result_sign; s1c_sticky_pre <= s1b_sticky_pre;
      s1c_out_is_nan <= s1b_out_is_nan; s1c_out_is_inf <= s1b_out_is_inf;
      s1c_out_inf_sign <= s1b_out_inf_sign;
    end
  end

  // exponent of the leading-1: bit b -> 2^(b - (WIN-2) + anchor_top)
  int e_msb;
  assign e_msb = s1c_msb_pos - (WIN - 2) + s1c_anchor_top;

  // normalize: shift leading-1 up to bit WIN (top of an (WIN+1)-bit reg)
  int shl;
  logic [WIN:0] mag_norm;
  always_comb begin
    shl = WIN - s1c_msb_pos;
    if (!s1c_any_one)    mag_norm = '0;
    else if (shl <= 0)   mag_norm = s1c_mag_raw;
    else                 mag_norm = s1c_mag_raw << shl;
  end

  // wide-normal extraction (leading-1 at bit WIN):
  //   mantissa = bits [WIN-1 -: ACC_MANT], guard/round/sticky below.
  // biased exp for a normal = e_msb + ACC_BIAS.
  int biased_exp_pre;
  assign biased_exp_pre = e_msb + ACC_BIAS;

  // subnormal path: shift right by (1 - biased_exp_pre)
  int sub_shift_s;
  assign sub_shift_s = 1 - biased_exp_pre;
  logic is_sub_path;
  int   sub_shift;
  always_comb begin
    is_sub_path = (biased_exp_pre <= 0);
    if (!is_sub_path)            sub_shift = 0;
    else if (sub_shift_s > WIN)  sub_shift = WIN;
    else                         sub_shift = sub_shift_s;
  end

  // ───────────────────── stage-2 pipeline register ─────────────
  logic [WIN:0] s2_mag_norm;
  int           s2_sub_shift, s2_biased_exp_pre;
  logic         s2_is_sub_path, s2_result_sign, s2_any_one, s2_sticky_pre;
  logic         s2_out_is_nan, s2_out_is_inf, s2_out_inf_sign;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s2_mag_norm <= '0; s2_sub_shift <= 0; s2_biased_exp_pre <= 0;
      s2_is_sub_path <= 1'b0; s2_result_sign <= 1'b0; s2_any_one <= 1'b0;
      s2_sticky_pre <= 1'b0;
      s2_out_is_nan <= 1'b0; s2_out_is_inf <= 1'b0; s2_out_inf_sign <= 1'b0;
    end else begin
      s2_mag_norm <= mag_norm; s2_sub_shift <= sub_shift;
      s2_biased_exp_pre <= biased_exp_pre; s2_is_sub_path <= is_sub_path;
      s2_result_sign <= s1c_result_sign; s2_any_one <= s1c_any_one;
      s2_sticky_pre <= s1c_sticky_pre;
      s2_out_is_nan <= s1c_out_is_nan; s2_out_is_inf <= s1c_out_is_inf;
      s2_out_inf_sign <= s1c_out_inf_sign;
    end
  end

  // apply the subnormal right-shift; remember anything dropped as sticky
  logic [WIN:0] mag_post;
  logic         sub_dropped;
  always_comb begin
    mag_post    = (s2_sub_shift >= WIN+1) ? '0 : (s2_mag_norm >> s2_sub_shift);
    sub_dropped = 1'b0;
    for (int i = 0; i <= WIN; i++)
      if ((i < s2_sub_shift) && s2_mag_norm[i]) sub_dropped = 1'b1;
  end

  // mantissa window: leading-1 at bit WIN, ACC_MANT mantissa bits below it.
  //   mantissa = mag_post[WIN-1 -: ACC_MANT]
  //   guard    = mag_post[WIN-1-ACC_MANT]
  //   round    = mag_post[WIN-2-ACC_MANT]
  //   sticky   = OR(lower) | sub_dropped | s2_sticky_pre
  logic [ACC_MANT-1:0] mant;
  logic                guard_b, round_b, sticky_b;
  always_comb begin
    mant    = mag_post[WIN-1 -: ACC_MANT];
    guard_b = mag_post[WIN-1-ACC_MANT];
    round_b = (WIN-2-ACC_MANT >= 0) ? mag_post[WIN-2-ACC_MANT] : 1'b0;
    sticky_b = s2_sticky_pre | sub_dropped;
    for (int i = 0; i < WIN-2-ACC_MANT; i++)
      if (mag_post[i]) sticky_b = 1'b1;
  end

  logic round_up;
  assign round_up = guard_b && ((round_b | sticky_b) || mant[0]);

  logic [ACC_MANT:0] mant_rounded;       // +1 for round carry
  assign mant_rounded = {1'b0, mant} + (ACC_MANT+1)'(round_up);

  // pack
  logic out_zero_exact;
  assign out_zero_exact = !s2_any_one && !s2_out_is_nan && !s2_out_is_inf;

  int                  biased_exp_pack;
  logic [ACC_MANT-1:0] frac_final;
  always_comb begin
    if (s2_is_sub_path) begin
      if (mant_rounded[ACC_MANT]) begin biased_exp_pack = 1;            frac_final = '0; end
      else                        begin biased_exp_pack = 0;            frac_final = mant_rounded[ACC_MANT-1:0]; end
    end else begin
      if (mant_rounded[ACC_MANT]) begin biased_exp_pack = s2_biased_exp_pre + 1; frac_final = '0; end
      else                        begin biased_exp_pack = s2_biased_exp_pre;     frac_final = mant_rounded[ACC_MANT-1:0]; end
    end
  end

  logic [ACC_W-1:0] y_d;
  always_comb begin
    if (s2_out_is_nan)
      y_d = {1'b0, {ACC_EXP{1'b1}}, 1'b1, {(ACC_MANT-1){1'b0}}};       // canonical qNaN
    else if (s2_out_is_inf)
      y_d = {s2_out_inf_sign, {ACC_EXP{1'b1}}, {ACC_MANT{1'b0}}};
    else if (out_zero_exact)
      y_d = '0;
    else if (biased_exp_pack >= EXPMAX)
      y_d = {s2_result_sign, {ACC_EXP{1'b1}}, {ACC_MANT{1'b0}}};       // overflow -> Inf
    else if (biased_exp_pack <= 0) begin
      if (mant_rounded == '0) y_d = {s2_result_sign, {(ACC_W-1){1'b0}}};
      else                    y_d = {s2_result_sign, {ACC_EXP{1'b0}}, frac_final};
    end else
      y_d = {s2_result_sign, ACC_EXP'(biased_exp_pack), frac_final};
  end

  // ───────────────────── stage-3 register out ──────────────────
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) y_o <= '0;
    else         y_o <= y_d;
  end

endmodule
