// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// i32_to_fp16_ref — behavioral golden for i32_to_fp16.
//
// Independent implementation: pure-integer, downward-scan style. Scans
// candidate exponents from largest to smallest, building the mantissa by
// right-shifting the magnitude into place. Different code path from the
// DUT (which scans for the leading-1 from LSB up and uses a left-shift
// barrel) so coincidental shared bugs are unlikely.
//
// Now mirrors the DUT's prescale: for |x| ≥ 2^16 the input is shifted
// right before conversion and `shift_o` reports the shift amount.

module i32_to_fp16_ref (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic signed [31:0] x_i,
  output logic        [15:0] y_o,
  output logic        [4:0]  shift_o
);

  function automatic logic [20:0] cvt_and_shift(input logic signed [31:0] x);
    // packed return: {shift[4:0], fp16[15:0]}
    logic               s;
    logic [31:0]        mag;
    int                 e;            // unbiased exponent of |x| (msb position)
    int                 e_post;       // exponent after prescale
    int                 sh;           // mantissa-extraction shift
    int                 pre_shift;    // prescale right-shift applied to |x|
    logic [31:0]        mag_sh;
    logic               extra_sticky;
    logic [10:0]        mant_pre;
    logic [11:0]        mant_round;
    logic               guard, sticky, lsb, round_up;
    int                 biased;
    logic [9:0]         mant_final;
    logic [15:0]        fp16;
    begin
      if (x == 32'sd0) return {5'd0, 16'h0000};

      s   = x[31];
      mag = s ? (~x + 32'd1) : x;

      // Scan for leading-1 from the top.
      e = 0;
      for (int i = 31; i >= 0; i--) begin
        if (mag[i] && (e == 0)) e = i;
      end

      // Prescale: keep zero for the legacy fit-in-fp16 range, otherwise
      // pull the leading-1 down to bit 14 so RNE round-up can't push
      // the fp16 result to ±Inf.
      if (e <= 14) pre_shift = 0;
      else         pre_shift = e - 14;

      if (pre_shift == 0) begin
        mag_sh       = mag;
        extra_sticky = 1'b0;
        e_post       = e;
      end else begin
        mag_sh       = mag >> pre_shift;
        extra_sticky = (|(mag & ((32'd1 << pre_shift) - 32'd1)));
        e_post       = 14;
      end

      biased = e_post + 15;

      // Standard 11-bit-with-implicit-1 extraction from mag_sh.
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


  logic [15:0] y_d;
  logic [4:0]  shift_d;
  logic [20:0] cvt_pack;
  assign cvt_pack = cvt_and_shift(x_i);
  assign y_d      = cvt_pack[15:0];
  assign shift_d  = cvt_pack[20:16];

  // Two registered stages so the behavioral golden matches the DUT's
  // 2-cycle latency (fp16_lat_pkg::I32_TO_FP16_LAT) cycle-for-cycle. The
  // compute itself stays fully combinational — only the output is delayed.
  logic [15:0] y_s1;
  logic [4:0]  shift_s1;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      y_s1    <= 16'h0000;
      shift_s1<= 5'd0;
      y_o     <= 16'h0000;
      shift_o <= 5'd0;
    end else begin
      y_s1    <= y_d;
      shift_s1<= shift_d;
      y_o     <= y_s1;
      shift_o <= shift_s1;
    end
  end

endmodule
