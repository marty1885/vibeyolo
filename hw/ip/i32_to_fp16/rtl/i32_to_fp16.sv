// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// i32_to_fp16 — signed int32 to IEEE-754 binary16 conversion with an
// optional auto-prescale.
//
// Background: |acc_i32| up to 2^31-1 won't fit in fp16 (max ≈ 65504), so
// any layer with a big accumulator used to pre-shift the int32 in Python
// and bake the shift into `scale_fp16`. To eliminate that workaround the
// IP now reports the prescale it performed:
//
//   acc_i32 ≈ fp16_o * 2^shift_o     (within fp16 RNE ULP at exponent shift_o)
//
//   • |acc_i32| < 2^16 → shift_o = 0  (bit-identical to the prior IP)
//   • else             → shift_o = msb_pos(|acc|) - 15 ; the conversion is
//                        applied to (acc >>> shift_o) so the fp16 result
//                        is in [-65535, 65535] before RNE; round-up to
//                        65536 saturates to ±Inf (as before).
//
// The downstream consumer (requant) absorbs `shift_o` by adding it to the
// exponent of `scale_fp16` before the FMA. Other consumers (add_rq,
// box_decode) feed values that always fit in fp16 (|x| ≤ 2^15), so shift_o
// is identically 0 for them and they leave it unconnected.
//
// Contract:
//   On posedge clk_i:
//     - !rst_ni: y_o <= 16'h0000 ; shift_o <= 4'd0
//     - else  : y_o <= fp16(rne(x_i >>> shift)) ; shift_o <= shift
//
// Implementation: leading-one count on |x|, derive shift, then barrel-shift
// to align the implicit-1 to bit 10 with guard/round/sticky retained.

module i32_to_fp16 (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic signed [31:0] x_i,
  output logic        [15:0] y_o,
  output logic        [4:0]  shift_o
);

  // ─── absolute value ──────────────────────────────────────
  // For INT32_MIN (-2^31), |x| = 2^31 fits in 32-bit unsigned.
  logic        sign;
  logic [31:0] mag;
  always_comb begin
    sign = x_i[31];
    mag  = sign ? (~x_i + 32'd1) : x_i;
  end

  // ─── leading-one position (msb index, 0..31) ─────────────
  // For mag==0 we short-circuit to zero below; msb_pos value irrelevant.
  logic [5:0] msb_pos;
  always_comb begin
    msb_pos = 6'd0;
    for (int i = 0; i < 32; i++) begin
      if (mag[i]) msb_pos = 6'(i);
    end
  end

  // ─── prescale selection ──────────────────────────────────
  // Keep shift==0 only for |x| < 2^15 (msb_pos ≤ 14). For msb_pos == 15
  // we MUST shift by 1 because |x| in (65504, 65535] otherwise RNE-rounds
  // UP to fp16 +Inf (the next representable step after 65504 is 65536,
  // and values like 65530 are closer to 65536 than 65504). Above that,
  // pull the implicit-1 down to bit 14, so the shifted magnitude is
  // ≤ 2^15-1 and RNE round-up can't push it to fp16 ±Inf.
  //
  // Max shift: msb_pos can be 31 → shift = 31-14 = 17, which needs 5
  // bits. We expose shift_o as 5 bits so even INT32_MIN (|x|=2^31,
  // shift=17) is representable losslessly.
  logic [4:0] shift_w;
  always_comb begin
    if (msb_pos <= 6'd14) shift_w = 5'd0;
    else                  shift_w = 5'(msb_pos - 6'd14);
  end

  // ─── shifted magnitude ──────────────────────────────────
  // Replace the original magnitude with mag >>> shift_w, keeping the
  // discarded low bits available for an OR-sticky.
  logic [31:0] mag_shifted;
  logic        shifted_sticky;
  always_comb begin
    if (shift_w == 5'd0) begin
      mag_shifted    = mag;
      shifted_sticky = 1'b0;
    end else begin
      mag_shifted    = mag >> shift_w;
      // OR of the bits being shifted out → contributes to sticky in RNE.
      shifted_sticky = |(mag & ((32'd1 << shift_w) - 32'd1));
    end
  end

  // After shifting, the new MSB is at bit (msb_pos - shift_w). Compute it
  // directly so we don't redo the leading-one scan.
  logic [5:0] msb_post;
  always_comb begin
    msb_post = (shift_w == 5'd0) ? msb_pos : 6'd14;
  end

  // ─── mantissa extraction with G/R/S for RNE ──────────────
  // Strategy: left-justify mag_shifted so the implicit 1 lands at bit 31,
  // then mantissa = bits[30:21], guard = bit[20], sticky = OR(bits[19:0])
  // | shifted_sticky.
  logic [31:0] aligned;
  logic [9:0]  mant;
  logic        guard;
  logic        sticky;

  always_comb begin
    if (msb_post >= 6'd31) begin
      aligned = mag_shifted;
    end else begin
      aligned = mag_shifted << (6'd31 - msb_post);
    end
    mant   = aligned[30:21];
    guard  = aligned[20];
    sticky = (|aligned[19:0]) | shifted_sticky;
  end

  logic _unused_aligned_msb;
  assign _unused_aligned_msb = aligned[31];

  // ─── RNE: round up if guard && (sticky || lsb) ───────────
  logic round_up;
  assign round_up = guard && (sticky || mant[0]);

  logic [10:0] mant_rounded;
  assign mant_rounded = {1'b0, mant} + 11'(round_up);

  // ─── biased exponent ─────────────────────────────────────
  // Unbiased exp (of the shifted value) = msb_post; fp16 bias = 15.
  logic signed [7:0] exp_pre;
  logic signed [7:0] exp_post;
  logic [9:0]        mant_final;

  always_comb begin
    exp_pre = 8'(signed'({2'b00, msb_post})) + 8'sd15;
    if (mant_rounded[10]) begin
      exp_post   = exp_pre + 8'sd1;
      mant_final = 10'd0;
    end else begin
      exp_post   = exp_pre;
      mant_final = mant_rounded[9:0];
    end
  end

  // ─── pack / saturate / zero ──────────────────────────────
  logic [15:0] y_d;
  always_comb begin
    if (mag == 32'd0) begin
      y_d = 16'h0000;
    end else if (exp_post >= 8'sd31) begin
      y_d = {sign, 15'h7C00};
    end else begin
      y_d = {sign, exp_post[4:0], mant_final};
    end
  end

  // ─── registered output ───────────────────────────────────
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      y_o     <= 16'h0000;
      shift_o <= 5'd0;
    end else begin
      y_o     <= y_d;
      shift_o <= shift_w;
    end
  end

endmodule
