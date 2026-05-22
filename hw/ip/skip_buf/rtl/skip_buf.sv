// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// skip_buf — streaming feature-map skip buffer for FPN/PAN U-turn.
//
// Phased fill-then-drain buffer: the producer writes Depth = H*W*C int8
// bytes in raster order, then once `full_o` asserts the consumer drains
// the same Depth bytes in the same order. Reads and writes are not
// interleaved in this version; a future variant may add a configurable
// read order (e.g. CHW vs HWC traversal).
//
// Implementation:
//   * `mem[Depth]` is a plain 8-bit array; Verilator/synthesis infer BRAM
//     for large Depth.
//   * Two pointers (`wptr_q`, `rptr_q`) advance independently on their
//     respective handshakes. Reads are blocked until `full_o` is set;
//     writes are blocked once `full_o` is set.
//   * `clr_i` (sync) and `!rst_ni` (async-assert / sync-deassert) both
//     return the block to empty.

module skip_buf #(
  parameter int H = 16,
  parameter int W = 16,
  parameter int C = 32
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic              clr_i,

  // Write stream (raster order: c0..C-1 per pixel, pixels row-major).
  input  logic              wvalid_i,
  output logic              wready_o,
  input  logic signed [7:0] wdata_i,

  // Read stream (same raster order in this variant).
  output logic              rvalid_o,
  input  logic              rready_i,
  output logic signed [7:0] rdata_o,

  // Status.
  output logic              full_o,
  output logic              empty_o
);

  localparam int Depth = H * W * C;
  localparam int AW    = (Depth <= 1) ? 1 : $clog2(Depth);
  // One extra bit on the pointers so we can represent the count Depth
  // unambiguously (e.g. a Depth that is a power of two still needs AW+1
  // bits to encode the value Depth itself).
  localparam int PW    = AW + 1;

  logic [PW-1:0] wptr_q, wptr_d;
  logic [PW-1:0] rptr_q, rptr_d;

  logic full_q, full_d;

  logic [7:0] mem [Depth];

  // Handshake firing.
  logic w_fire;
  logic r_fire;

  always_comb begin
    wready_o = ~full_q;
    w_fire   = wvalid_i & wready_o;

    // Reads are blocked until the buffer has been filled, and stop once
    // every entry has been drained.
    rvalid_o = full_q & (rptr_q != PW'(Depth));
    r_fire   = rvalid_o & rready_i;

    wptr_d = wptr_q;
    rptr_d = rptr_q;
    full_d = full_q;

    if (w_fire) begin
      wptr_d = wptr_q + PW'(1);
      if (wptr_q == PW'(Depth - 1)) begin
        full_d = 1'b1;
      end
    end

    if (r_fire) begin
      rptr_d = rptr_q + PW'(1);
    end
  end

  assign full_o  = full_q;
  assign empty_o = ~full_q & (wptr_q == '0);
  // Combinational mem read on the current rptr. Verilator will still
  // infer a RAM for the (write-only) `mem` array; the synthesis target
  // for this block is an SRAM with registered read port that we can swap
  // in later (see README).
  assign rdata_o = (rvalid_o) ? $signed(mem[rptr_q[AW-1:0]]) : 8'sd0;

  // Sequential state.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wptr_q <= '0;
      rptr_q <= '0;
      full_q <= 1'b0;
    end else if (clr_i) begin
      wptr_q <= '0;
      rptr_q <= '0;
      full_q <= 1'b0;
    end else begin
      wptr_q <= wptr_d;
      rptr_q <= rptr_d;
      full_q <= full_d;
    end
  end

  // Memory write.
  always_ff @(posedge clk_i) begin
    if (w_fire) begin
      mem[wptr_q[AW-1:0]] <= wdata_i;
    end
  end

endmodule
