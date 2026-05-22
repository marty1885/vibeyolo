// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// fp16_to_i8_sat — IEEE-754 binary16 to signed int8 conversion with
// round-to-nearest-even and saturating clamp to [-128, +127].
//
// Special cases:
//   NaN  -> 0
//   +Inf -> +127
//   -Inf -> -128
//   ±0   -> 0
//   subnormal -> 0   (|x| < 2^-14 << 0.5)
//
// Contract:
//   On posedge clk_i:
//     !rst_ni: y_o <= 8'sd0
//     else   : y_o <= int8_sat( rne( fp16_value(x_i) ) )
//
// Implementation: unpack sign/exp/mant; classify; for finite normals form
// the 11-bit significand {1, mant}, right-shift to an integer with guard
// and sticky retained, apply RNE rounding, then apply the sign and
// saturate to [-128, +127]. Subtle case: +127.5 -> RNE rounds to +128
// -> saturates back to +127. -128.5 -> RNE rounds to -128 which is the
// exact lower bound, no saturation needed.

module fp16_to_i8_sat (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic        [15:0] x_i,
  output logic signed [7:0]  y_o
);

  // ─── unpack ───────────────────────────────────────────────
  logic        sign;
  logic [4:0]  exp_b;
  logic [9:0]  mant;
  assign sign  = x_i[15];
  assign exp_b = x_i[14:10];
  assign mant  = x_i[9:0];

  // ─── classify ─────────────────────────────────────────────
  logic is_special;
  logic is_nan;
  logic is_inf;
  logic is_zero_or_subn;
  assign is_special      = (exp_b == 5'h1F);
  assign is_nan          = is_special && (mant != 10'd0);
  assign is_inf          = is_special && (mant == 10'd0);
  assign is_zero_or_subn = (exp_b == 5'd0);

  // ─── unbiased exponent and significand ────────────────────
  logic signed [6:0] e_unb;    // -15..+16 (only -14..+15 used for normals)
  assign e_unb = $signed({2'b00, exp_b}) - 7'sd15;

  logic [10:0] sig11;
  assign sig11 = {1'b1, mant};

  // Magnitude-range classification:
  //   e_unb >=  7  ->  |value| >= 128, will saturate
  //   e_unb <= -2  ->  |value| < 0.5, rounds to 0
  //   else (e_unb in [-1, 6]) -> shift = 10 - e_unb in [4, 11]
  logic out_of_range_big;
  logic out_of_range_small;
  assign out_of_range_big   = (e_unb >=  7'sd7);
  assign out_of_range_small = (e_unb <= -7'sd2);

  // ─── compute shift_amt (4..11 on the in-range path) ───────
  logic [3:0] shift_amt;
  always_comb begin
    logic signed [7:0] sh;
    sh = 8'sd10 - $signed({e_unb[6], e_unb});
    if (sh < 8'sd0)       shift_amt = 4'd0;
    else if (sh > 8'sd11) shift_amt = 4'd11;
    else                  shift_amt = sh[3:0];
  end

  // ─── integer magnitude (8 bits), guard, sticky ────────────
  logic [7:0]  mag_int;
  logic        guard;
  logic        sticky;
  logic [10:0] sticky_mask;
  always_comb begin
    // Truncate-shift; for shift_amt in [4,11] the result fits in 8 bits
    // (max sig11 is 0x7FF, shifted right by >=4 -> <= 0x7F).
    mag_int = 8'((sig11 >> shift_amt));

    // Guard = bit (shift_amt - 1) of sig11. Falls back to 0 when shift==0.
    guard = (shift_amt == 4'd0) ? 1'b0
                                : sig11[shift_amt - 4'd1];

    // Sticky = OR of bits strictly below the guard, i.e. mask covers
    // bits [shift_amt - 2 : 0]. Empty mask (zero) when shift_amt < 2.
    if (shift_amt >= 4'd2) begin
      sticky_mask = (11'd1 << (shift_amt - 4'd1)) - 11'd1;
    end else begin
      sticky_mask = 11'd0;
    end
    sticky = |(sig11 & sticky_mask);
  end

  // ─── RNE: round_up if guard && (sticky || lsb) ────────────
  logic       round_up;
  logic [8:0] mag_rnd;   // 9 bits: catches 0xFF -> 0x100 carry
  assign round_up = guard && (sticky || mag_int[0]);
  assign mag_rnd  = {1'b0, mag_int} + 9'(round_up);

  // ─── apply sign + clamp to [-128, +127] ───────────────────
  //   positive: saturate when mag_rnd >= 128
  //   negative: saturate (to -128) only when mag_rnd >  128;
  //            mag_rnd == 128 is the exact -128 and is in range.
  // out_of_range_big and out_of_range_small short-circuit.
  logic signed [7:0] y_normal;
  logic signed [8:0] mag_signed;
  assign mag_signed = $signed({1'b0, mag_rnd[7:0]});
  always_comb begin
    if (out_of_range_big) begin
      y_normal = sign ? -8'sd128 : 8'sd127;
    end else if (out_of_range_small) begin
      y_normal = 8'sd0;
    end else if (!sign && mag_rnd >= 9'd128) begin
      y_normal = 8'sd127;
    end else if (sign && mag_rnd > 9'd128) begin
      y_normal = -8'sd128;
    end else if (sign && mag_rnd == 9'd128) begin
      y_normal = -8'sd128;
    end else begin
      // mag_rnd fits in 8 unsigned bits at this point (and as |int8|).
      y_normal = sign ? 8'((-mag_signed)) : 8'(mag_signed);
    end
  end

  // ─── final output mux ─────────────────────────────────────
  logic signed [7:0] y_d;
  always_comb begin
    if (is_nan) begin
      y_d = 8'sd0;
    end else if (is_inf) begin
      y_d = sign ? -8'sd128 : 8'sd127;
    end else if (is_zero_or_subn) begin
      y_d = 8'sd0;
    end else begin
      y_d = y_normal;
    end
  end

  // ─── registered output ────────────────────────────────────
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      y_o <= 8'sd0;
    end else begin
      y_o <= y_d;
    end
  end

endmodule
