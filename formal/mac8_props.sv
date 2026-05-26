// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// mac8_props — formal numeric properties for mac8.
//
// mac8 accumulates signed int8xint8 products into a signed int32 with NO
// saturation: it wraps on overflow and relies on "the caller sizing the
// accumulator chain to avoid wrap". This module turns that informal caveat
// into a machine-checked theorem:
//
//   For any run of up to K multiply-accumulates since the last clr/reset,
//   mac8's 32-bit acc_o equals the EXACT integer sum of those products
//   (bit-for-bit) -> the int32 accumulator never overflows for K terms.
//
// Method: carry an exact 48-bit shadow accumulator (acc_ref) and a term
// counter (n_acc) that follow mac8's own update rules, and assert that the
// sign-extended hardware result matches the shadow whenever n_acc <= K.
// A magnitude lemma (|acc_ref| <= n_acc * |prod|_max) bounds the growth and
// makes the exactness assertion inductive (closes at k=1).
//
// Worst-case |signed 8x8 product| = (-128)*(-128) = 16384. The largest K
// with K*16384 <= 2^31-1 is 131071; at K=131072 the sum can reach 2^31 and
// wrap, so the proof passes at K=131071 and (provably) breaks at 131072.

module mac8_props #(
  parameter int K = 131071           // guaranteed-safe accumulation depth
) (
  input  logic              clk_i,
  input  logic              rst_ni,
  input  logic              clr_i,
  input  logic              en_i,
  input  logic signed [7:0] a_i,
  input  logic signed [7:0] b_i
);

  // ---- Device under proof ------------------------------------------------
  logic signed [31:0] acc_o;
  mac8 dut (
    .clk_i (clk_i), .rst_ni (rst_ni),
    .clr_i (clr_i), .en_i (en_i),
    .a_i (a_i), .b_i (b_i), .acc_o (acc_o)
  );

  // ---- Exact shadow model ------------------------------------------------
  localparam int W = 48;
  localparam logic signed [W-1:0] PMAX = 48'sd16384;   // max |signed 8x8|

  logic signed [W-1:0] acc_ref;     // exact, never truncated
  logic        [31:0]  n_acc;       // terms since last clr/reset (capped)
  logic signed [W-1:0] prod_full;

  assign prod_full = $signed(a_i) * $signed(b_i);      // exact product

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      acc_ref <= '0;
      n_acc   <= 32'd0;
    end else if (clr_i) begin
      acc_ref <= prod_full;                            // load: first term
      n_acc   <= 32'd1;
    end else if (en_i) begin
      if (n_acc > K) begin
        acc_ref <= acc_ref;                            // beyond K: freeze
        n_acc   <= n_acc;                              // (don't-care region)
      end else begin
        acc_ref <= acc_ref + prod_full;
        n_acc   <= n_acc + 32'd1;
      end
    end
    // else: hold
  end

  // n_acc * PMAX as a wide signed bound (n_acc is non-negative).
  logic signed [W-1:0] mag_ub;
  assign mag_ub = $signed({16'd0, n_acc}) * PMAX;

  // Sign-extension of the 32-bit hardware accumulator to W bits.
  logic signed [W-1:0] acc_o_sx;
  assign acc_o_sx = {{(W-32){acc_o[31]}}, acc_o};

  // One-cycle snapshot to check clr-over-en priority directly on the DUT.
  logic               clr_q;
  logic signed [31:0] prod32_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      clr_q    <= 1'b0;
      prod32_q <= 32'sd0;
    end else begin
      clr_q    <= clr_i;
      prod32_q <= 32'($signed(a_i) * $signed(b_i));
    end
  end

  // ---- Properties --------------------------------------------------------
  always @(posedge clk_i) begin
    if (rst_ni) begin
      // Lemma (inductive growth bound): |acc_ref| <= n_acc * 16384.
      lemma_mag_hi: assert (acc_ref <=  mag_ub);
      lemma_mag_lo: assert (acc_ref >= -mag_ub);

      // Numeric guarantee: up to K accumulations, the int32 hardware result
      // is the exact integer sum -> no overflow/wrap, and the internal
      // 16-bit product truncation in mac8 is lossless.
      if (n_acc <= K) begin
        no_overflow: assert (acc_o_sx == acc_ref);
      end

      // Contract: synchronous clear takes priority over enable. If clr_i
      // was asserted last cycle, acc_o now holds that cycle's product
      // regardless of en_i (load, not accumulate).
      if (clr_q) begin
        clr_priority: assert (acc_o == prod32_q);
      end
    end
  end

endmodule
