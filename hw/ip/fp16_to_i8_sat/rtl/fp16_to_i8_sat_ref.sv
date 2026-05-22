// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// fp16_to_i8_sat_ref — behavioral golden for fp16_to_i8_sat.
//
// Independent style: convert the fp16 value to a signed fixed-point
// number with 14 fractional bits (i.e. value * 2^14), then perform RNE
// rounding to integer using a single add-and-shift, then saturate to
// [-128, +127]. The DUT classifies range first and rounds; this REF
// rounds first (in a wider intermediate, so no fractional precision is
// ever lost) and saturates last, so any off-by-one at boundaries like
// +127.5 / -128.5 would diverge between the two.

module fp16_to_i8_sat_ref (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic        [15:0] x_i,
  output logic signed [7:0]  y_o
);

  function automatic logic signed [7:0] cvt(input logic [15:0] x);
    logic               s;
    logic [4:0]         e;
    logic [9:0]         m;
    logic [10:0]        sig;
    int                 e_unb;
    int                 sh;
    // 32-bit signed accumulator for value * 2^14 with headroom. Max
    // representable fp16 finite (65504) * 2^14 ~ 1.07e9 < 2^31.
    logic signed [31:0] fx;       // value * 2^14
    logic signed [31:0] rnd_q;    // round_to_nearest(fx / 2^14)
    logic               tie;      // exactly halfway
    logic               lsb;
    begin
      s = x[15];
      e = x[14:10];
      m = x[9:0];

      // Special cases first.
      if (e == 5'h1F) begin
        if (m != 10'd0) return 8'sd0;            // NaN -> 0
        return s ? -8'sd128 : 8'sd127;            // ±Inf
      end
      if (e == 5'd0) return 8'sd0;                // zero or subnormal

      // Normal: significand = 1.mant scaled by 2^10 -> integer in [1024, 2047].
      sig   = {1'b1, m};
      e_unb = int'(e) - 15;

      // Build fx = value * 2^14 = sig * 2^(e_unb + 14 - 10) = sig << (e_unb+4).
      // value = sig * 2^(e_unb - 10), so value * 2^14 = sig * 2^(e_unb + 4).
      // The shift right (negative sh) only fires for very small e_unb where
      // the true value is far below 0.5, so any precision loss is below the
      // rounding threshold (fx is much less than 0.5 * 2^14 = 8192).
      sh = e_unb + 4;
      if (sh >= 0) begin
        fx = 32'(signed'({21'd0, sig})) <<< sh;
      end else begin
        fx = 32'(signed'({21'd0, sig})) >>> (-sh);
      end
      if (s) fx = -fx;

      // RNE rounding: round_to_nearest(fx / 2^14). Add 2^13, arith-shift
      // right by 14, then if the discarded magnitude was exactly 2^13
      // (tie) and the result is odd, pull back by one to land on even.
      tie = ((fx >= 0) ? ((fx & 32'sd16383) == 32'sd8192)
                       : (((-fx) & 32'sd16383) == 32'sd8192));

      rnd_q = (fx + 32'sd8192) >>> 14;

      lsb = rnd_q[0];
      if (tie && lsb) begin
        // Halfway, currently rounded up to an odd value — pull back to even.
        rnd_q = rnd_q - 32'sd1;
      end

      // Saturate to [-128, +127].
      if (rnd_q >  32'sd127) return  8'sd127;
      if (rnd_q < -32'sd128) return -8'sd128;
      return rnd_q[7:0];
    end
  endfunction

  logic signed [7:0] y_d;
  assign y_d = cvt(x_i);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      y_o <= 8'sd0;
    end else begin
      y_o <= y_d;
    end
  end

endmodule
