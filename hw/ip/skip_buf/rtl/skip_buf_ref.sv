// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// skip_buf_ref — behavioral golden for skip_buf.
//
// Independently coded reference. Where the DUT uses a `mem[Depth]` array
// with explicit write/read pointers and a `full_q` latch, this REF uses
// a SystemVerilog queue `byte q[$]` with `push_back`/`pop_front` to model
// FIFO semantics, and derives state (full / empty / drained) from counts.
//
// Behavior must be cycle-accurate with the DUT for the same stimulus:
//   * wready_o = 1 until exactly Depth bytes have been written.
//   * After Depth writes, full asserts, wready_o = 0, and rvalid_o = 1.
//   * Reads drain in push order. After Depth reads, rvalid_o = 0 again.
//   * clr_i / !rst_ni return to empty.

module skip_buf_ref #(
  parameter int H = 16,
  parameter int W = 16,
  parameter int C = 32
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic              clr_i,

  input  logic              wvalid_i,
  output logic              wready_o,
  input  logic signed [7:0] wdata_i,

  output logic              rvalid_o,
  input  logic              rready_i,
  output logic signed [7:0] rdata_o,

  output logic              full_o,
  output logic              empty_o
);

  localparam int Depth = H * W * C;

  // Bytes written / bytes read counters as plain integers.
  int unsigned n_written_q, n_written_d;
  int unsigned n_read_q,    n_read_d;

  // Storage queue.
  logic signed [7:0] q [$];

  // Deferred queue actions.
  logic              push_pend;
  logic signed [7:0] push_dat;
  logic              pop_pend;

  // Local derived state from registered counters.
  logic full_now;
  logic drained_now;

  always_comb begin
    full_now    = (n_written_q == Depth);
    drained_now = (n_read_q    == Depth);

    wready_o = ~full_now;
    rvalid_o = full_now & ~drained_now;

    full_o   = full_now;
    empty_o  = (n_written_q == 0) & ~full_now;

    // Read data is the head of the queue when valid. Reading an empty
    // queue is illegal but guarded.
    if (rvalid_o && (q.size() != 0)) begin
      rdata_o = q[0];
    end else begin
      rdata_o = 8'sd0;
    end

    n_written_d = n_written_q;
    n_read_d    = n_read_q;
    push_pend   = 1'b0;
    push_dat    = wdata_i;
    pop_pend    = 1'b0;

    if (wvalid_i && wready_o) begin
      n_written_d = n_written_q + 1;
      push_pend   = 1'b1;
      push_dat    = wdata_i;
    end

    if (rvalid_o && rready_i) begin
      n_read_d = n_read_q + 1;
      pop_pend = 1'b1;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      n_written_q <= 0;
      n_read_q    <= 0;
      q.delete();
    end else if (clr_i) begin
      n_written_q <= 0;
      n_read_q    <= 0;
      q.delete();
    end else begin
      n_written_q <= n_written_d;
      n_read_q    <= n_read_d;
      if (push_pend) q.push_back(push_dat);
      if (pop_pend)  void'(q.pop_front());
    end
  end

endmodule
