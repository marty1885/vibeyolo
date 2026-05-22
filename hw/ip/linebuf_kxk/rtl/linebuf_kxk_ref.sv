// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// linebuf_kxk_ref — behavioral golden for linebuf_kxk.
//
// Different coding style than the DUT:
//   * DUT uses a circular K-row bank + modular indexing + width-typed
//     counters and combinational patch read from per-channel banks.
//   * REF stores the entire H x W frame as a flat unpacked int8 array
//     per channel, uses plain `int` counters, and on each output beat
//     builds the K x K x Channels patch by explicit signed-range checks
//     against H and W with zero fill for out-of-frame cells.
//
// Same handshake contract as the DUT so the TB can compare cycle-by-cycle.

module linebuf_kxk_ref #(
  parameter int K        = 3,
  parameter int W        = 16,
  parameter int H        = 16,
  parameter int Channels = 1
) (
  input  logic                                   clk_i,
  input  logic                                   rst_ni,

  input  logic                                   clr_i,

  input  logic                                   wvalid_i,
  output logic                                   wready_o,
  input  logic signed [Channels-1:0][7:0]        wdata_i,

  output logic                                   rvalid_o,
  input  logic                                   rready_i,
  output logic signed [K*K*Channels-1:0][7:0]    rdata_o
);

  localparam int N     = K * K;
  localparam int P     = (K - 1) / 2;
  localparam int Total = H * W;

  // Flat HxW frame buffer, one per channel. (REF stores the whole image.)
  logic signed [7:0] frame [Channels][H*W];

  // Plain int counters.
  int in_cnt_q,  in_cnt_d;
  int out_cnt_q, out_cnt_d;

  // Decompose out_cnt into (out_row, out_col) for patch indexing.
  int out_row;
  int out_col;
  assign out_row = (W == 0) ? 0 : (out_cnt_q / W);
  assign out_col = (W == 0) ? 0 : (out_cnt_q % W);

  // Required input count (number of pixels that must have arrived before
  // the patch for the current output position can be emitted).
  int req_row;
  int req_col;
  int req_cnt;
  always_comb begin
    req_row = out_row + P;
    if (req_row >= H) req_row = H - 1;
    req_col = out_col + P;
    if (req_col >= W) req_col = W - 1;
    req_cnt = req_row * W + req_col + 1;
  end

  // Input back-pressure model mirrors the DUT's K-row-bank rule so that
  // the two implementations are cycle-aligned for the side-by-side
  // compare.
  int in_row_calc;
  int allow_w;
  assign in_row_calc = (W == 0) ? 0 : (in_cnt_q / W);
  always_comb begin
    if (in_row_calc < K) begin
      allow_w = 1;
    end else if (out_row >= in_row_calc - P) begin
      allow_w = 1;
    end else begin
      allow_w = 0;
    end
  end

  assign wready_o = (in_cnt_q  < Total) && (allow_w != 0);
  assign rvalid_o = (out_cnt_q < Total) && (in_cnt_q >= req_cnt);

  // Build the K x K x Channels patch from the per-channel frames, with
  // explicit zero-pad on OOF.
  always_comb begin
    for (int c = 0; c < Channels; c++) begin
      for (int ky = 0; ky < K; ky++) begin
        for (int kx = 0; kx < K; kx++) begin
          int sr;
          int sc;
          sr = out_row + ky - P;
          sc = out_col + kx - P;
          if (sr < 0 || sr >= H || sc < 0 || sc >= W) begin
            rdata_o[(ky*K + kx)*Channels + c] = 8'sd0;
          end else begin
            rdata_o[(ky*K + kx)*Channels + c] = frame[c][sr * W + sc];
          end
        end
      end
    end
  end

  // Counter update (purely additive — REF doesn't need row/col registers).
  always_comb begin
    in_cnt_d  = in_cnt_q;
    out_cnt_d = out_cnt_q;
    if (wvalid_i && wready_o) in_cnt_d  = in_cnt_q  + 1;
    if (rvalid_o && rready_i) out_cnt_d = out_cnt_q + 1;
  end

  // Sequential.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      in_cnt_q  <= 0;
      out_cnt_q <= 0;
      for (int c = 0; c < Channels; c++) begin
        for (int i = 0; i < H*W; i++) frame[c][i] <= 8'sd0;
      end
    end else if (clr_i) begin
      in_cnt_q  <= 0;
      out_cnt_q <= 0;
      for (int c = 0; c < Channels; c++) begin
        for (int i = 0; i < H*W; i++) frame[c][i] <= 8'sd0;
      end
    end else begin
      in_cnt_q  <= in_cnt_d;
      out_cnt_q <= out_cnt_d;
      if (wvalid_i && wready_o) begin
        for (int c = 0; c < Channels; c++) begin
          frame[c][in_cnt_q] <= wdata_i[c];
        end
      end
    end
  end

  // Avoid unused warnings when N is referenced only in localparam form.
  // verilator lint_off UNUSED
  logic [31:0] unused_n;
  assign unused_n = N;
  // verilator lint_on UNUSED

endmodule
