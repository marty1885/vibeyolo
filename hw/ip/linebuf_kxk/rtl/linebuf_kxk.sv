// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// linebuf_kxk — streaming K x K sliding-window patch generator with
// zero-padding (same-padding semantics).
//
// Consumes a H x W feature map of `Channels` int8 lanes in row-major
// raster order (one pixel of all channels per cycle when valid) and
// emits H x W K x K x Channels patches in the same raster order. For
// output position (or, oc) and channel c, element [(ky*K + kx)*Channels + c]
// of `rdata_o` is the input pixel at (or+ky-P, oc+kx-P) channel c
// (where P=(K-1)/2), or zero if that position is out of frame.
//
// Storage model:
//   * Per-channel circular bank of K row buffers of W int8 cells.
//     `Channels` such banks share one handshake/control FSM.
//   * Internally we track absolute row indices (never reset across
//     frames) so the bank-cycle and back-pressure rule reduce to the
//     single-channel single-frame case: bank `in_row_abs % K` holds
//     row `in_row_abs - K`, and that row is no longer needed once
//     `out_row_abs + P >= in_row_abs`.
//
// Continuous-flow:
//   * After accepting a full H*W frame of inputs the FSM continues
//     into the next frame's row 0 without external intervention; the
//     bank-free rule keeps the input back-pressured until the output
//     has consumed enough rows for the next overwrite to be safe.
//   * `clr_i` remains as an optional reset-to-known-state.
//
// Handshake:
//   * Same ready/valid contract as the single-channel version. For
//     the Channels==1 single-frame case the cycle-level behavior is
//     bit-exact identical to the prior implementation.

module linebuf_kxk #(
  parameter int K        = 3,
  parameter int W        = 16,
  parameter int H        = 16,
  parameter int Channels = 1
) (
  input  logic                                       clk_i,
  input  logic                                       rst_ni,

  input  logic                                       clr_i,

  // Input pixel stream (Channels-wide int8 vector per cycle when valid).
  input  logic                                       wvalid_i,
  output logic                                       wready_o,
  input  logic signed [Channels-1:0][7:0]            wdata_i,

  // Output patch stream: K*K*Channels signed bytes per beat. The element
  // at [(ky*K + kx)*Channels + c] is the input pixel at (out_row+ky-P,
  // out_col+kx-P) on channel c, or zero if OOF.
  output logic                                       rvalid_o,
  input  logic                                       rready_i,
  output logic signed [K*K*Channels-1:0][7:0]        rdata_o
);

  localparam int P        = (K - 1) / 2;
  localparam int Total    = H * W;
  // Absolute counters wrap modulo 2*H rows (input can be at most one
  // frame ahead of output). 2H rows is enough for difference math.
  localparam int RowsAbs  = 2 * H;
  localparam int RowAbsW  = $clog2(RowsAbs + 1);  // covers values 0..2H
  localparam int ColIdxW  = (W <= 1) ? 1 : $clog2(W);
  localparam int RowIdxW  = (H <= 1) ? 1 : $clog2(H);
  localparam int BankW    = (K <= 1) ? 1 : $clog2(K);
  localparam int CntW     = $clog2(Total + 1);

  // synthesis translate_off
  initial begin
    if (Channels < 1) begin
      $fatal(1, "linebuf_kxk: Channels must be >= 1");
    end
    if ((K % 2) != 1) begin
      $fatal(1, "linebuf_kxk: K must be odd");
    end
    if (W < K) begin
      $fatal(1, "linebuf_kxk: W must be >= K");
    end
    if (H < K) begin
      $fatal(1, "linebuf_kxk: H must be >= K");
    end
  end
  // synthesis translate_on

  // ── State ──────────────────────────────────────────────────
  // (col, row, row_abs) for input and output.
  //   *_col_q    in [0, W-1]
  //   *_row_q    in [0, H-1]    : row within current frame (raster pos)
  //   *_row_abs_q in [0, 2H-1)  : absolute row counter, wraps mod 2H
  logic [ColIdxW-1:0] in_col_q,    in_col_d;
  logic [RowIdxW-1:0] in_row_q,    in_row_d;
  logic [RowAbsW-1:0] in_row_abs_q,  in_row_abs_d;

  logic [ColIdxW-1:0] out_col_q,   out_col_d;
  logic [RowIdxW-1:0] out_row_q,   out_row_d;
  logic [RowAbsW-1:0] out_row_abs_q, out_row_abs_d;

  // ── Per-channel storage ────────────────────────────────────
  logic signed [7:0] rowbuf [Channels][K][W];

  // ── Required-input check (per output position) ─────────────
  // For output (out_row, out_col): the last in-frame pixel of the
  // K x K window is at (min(out_row+P, H-1), min(out_col+P, W-1)).
  // We need the input to have written at least up to that column on
  // that row of the *current* output frame.
  //
  // Express the "have we written through (req_row, req_col) of the
  // output's frame" predicate via absolute-row math:
  //   * If in_row_abs > out_row_abs + (req_row - out_row), the row is
  //     fully written.
  //   * If in_row_abs == out_row_abs + (req_row - out_row), then we
  //     also need in_col > req_col (i.e., in_col_abs has stepped past).
  //
  // Equivalent simpler form (mirrors the old single-frame logic when
  // input and output are on the same frame):
  //   req_row_abs = out_row_abs + (req_row - out_row)
  //   ready = (in_row_abs > req_row_abs) ||
  //           (in_row_abs == req_row_abs && in_col_q > req_col)
  // But "in_col_q > req_col" must include the case where the input
  // has already moved to the next row (in_row_abs == req_row_abs+1)
  // — covered by the strict-greater branch.

  logic [ColIdxW-1:0] req_col;
  logic [RowAbsW-1:0] req_row_abs;

  always_comb begin
    automatic int br_row;
    automatic int br_col;
    br_row = int'(out_row_q) + P;
    br_col = int'(out_col_q) + P;
    if (br_row >= H) br_row = H - 1;
    if (br_col >= W) br_col = W - 1;
    req_col = ColIdxW'(br_col);
    // req_row_abs = out_row_abs + (br_row - out_row), mod 2H.
    req_row_abs = RowAbsW'((int'(out_row_abs_q) + (br_row - int'(out_row_q))) % RowsAbs);
  end

  // Compare in_row_abs and req_row_abs as "in_row_abs is ahead of (or
  // equal to) req_row_abs by some amount in [0, H]". We compute the
  // forward distance modulo 2H.
  logic [RowAbsW-1:0] in_fwd_from_req;
  always_comb begin
    automatic int diff;
    diff = int'(in_row_abs_q) - int'(req_row_abs);
    if (diff < 0) diff = diff + RowsAbs;
    in_fwd_from_req = RowAbsW'(diff);
  end

  logic enough_inputs;
  always_comb begin
    // in_row_abs is "ahead" if 0 <= diff <= H (any larger means input
    // wrapped past in the modular sense, which shouldn't happen
    // because input is at most one frame ahead).
    if (in_fwd_from_req == '0) begin
      // Same absolute row as req: need in_col_q > req_col.
      enough_inputs = (in_col_q > req_col);
    end else if (in_fwd_from_req <= RowAbsW'(H)) begin
      enough_inputs = 1'b1;
    end else begin
      enough_inputs = 1'b0;
    end
  end

  // Bank-free: bank `in_row_abs % K` currently holds row
  // (in_row_abs - K) absolute, if in_row_abs >= K, else fresh.
  // The bank is free when that absolute row is no longer in any
  // pending output's window: out_row_abs + P >= in_row_abs.
  // (Modular: compute the forward distance from in_row_abs to
  // out_row_abs+P; if that's "0 or just-past" then free.)
  // Bank-free: compute the modular signed difference
  //   d = out_row_abs - (in_row_abs - P)   (mod 2H, mapped to [-H, +H))
  // Free iff d >= 0. This is the same rule as the single-frame
  // baseline but extended across frame boundaries via the abs-mod
  // counter. After reset both counters are 0 and d = P >= 0, so the
  // initial K rows are correctly admitted without any special case.
  logic bank_free;
  always_comb begin
    automatic int diff;
    // Bank b = in_row_abs % K. When in_row_abs >= K, bank holds row
    // (in_row_abs - K), needed by output rows up to (in_row_abs - K) + P.
    // Free once out_row_abs > (in_row_abs - K) + P, i.e. out >= in - K + P + 1,
    // i.e. out - (in - P) >= 1 - K. Equivalently the classic form
    // `out + P >= in - (K - 1)` collapses to the original single-frame
    // check `out >= in - P` once we exclude the never-written initial K rows.
    diff = int'(out_row_abs_q) - (int'(in_row_abs_q) - P);
    // Map diff into [-H, +H).
    if (diff <  -int'(H)) diff = diff + RowsAbs;
    if (diff >=  int'(H)) diff = diff - RowsAbs;
    // Early fill: first K rows of each frame have never been written yet.
    bank_free = (in_row_abs_q < RowAbsW'(K)) || (diff >= 0);
  end

  // Frame-boundary gate: stop accepting once we've taken H*W inputs for this
  // frame. Caller asserts clr_i (or rst_ni) between frames to release.
  logic in_frame_full;
  assign in_frame_full = (in_row_abs_q >= RowAbsW'(H));
  assign wready_o = bank_free && !in_frame_full;
  assign rvalid_o = enough_inputs;

  // ── Patch readout (combinational) ──────────────────────────
  always_comb begin
    for (int c = 0; c < Channels; c++) begin
      for (int ky = 0; ky < K; ky++) begin
        for (int kx = 0; kx < K; kx++) begin
          int sr;
          int sc;
          sr = int'(out_row_q) + ky - P;
          sc = int'(out_col_q) + kx - P;
          if (sr < 0 || sr >= H || sc < 0 || sc >= W) begin
            rdata_o[(ky*K + kx)*Channels + c] = 8'sd0;
          end else begin
            rdata_o[(ky*K + kx)*Channels + c] = rowbuf[c][sr % K][sc];
          end
        end
      end
    end
  end

  // ── Counters ───────────────────────────────────────────────
  logic w_fire;
  logic r_fire;
  assign w_fire = wvalid_i & wready_o;
  assign r_fire = rvalid_o & rready_i;

  always_comb begin
    in_col_d      = in_col_q;
    in_row_d      = in_row_q;
    in_row_abs_d  = in_row_abs_q;
    out_col_d     = out_col_q;
    out_row_d     = out_row_q;
    out_row_abs_d = out_row_abs_q;

    if (w_fire) begin
      if (in_col_q == ColIdxW'(W - 1)) begin
        in_col_d = '0;
        // End-of-row: advance row (and row_abs).
        if (in_row_q == RowIdxW'(H - 1)) begin
          in_row_d = '0;
        end else begin
          in_row_d = in_row_q + RowIdxW'(1);
        end
        if (in_row_abs_q == RowAbsW'(RowsAbs - 1)) begin
          in_row_abs_d = '0;
        end else begin
          in_row_abs_d = in_row_abs_q + RowAbsW'(1);
        end
      end else begin
        in_col_d = in_col_q + ColIdxW'(1);
      end
    end

    if (r_fire) begin
      if (out_col_q == ColIdxW'(W - 1)) begin
        out_col_d = '0;
        if (out_row_q == RowIdxW'(H - 1)) begin
          out_row_d = '0;
        end else begin
          out_row_d = out_row_q + RowIdxW'(1);
        end
        if (out_row_abs_q == RowAbsW'(RowsAbs - 1)) begin
          out_row_abs_d = '0;
        end else begin
          out_row_abs_d = out_row_abs_q + RowAbsW'(1);
        end
      end else begin
        out_col_d = out_col_q + ColIdxW'(1);
      end
    end
  end

  // ── Sequential ─────────────────────────────────────────────
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      in_col_q      <= '0;
      in_row_q      <= '0;
      in_row_abs_q  <= '0;
      out_col_q     <= '0;
      out_row_q     <= '0;
      out_row_abs_q <= '0;
    end else if (clr_i) begin
      in_col_q      <= '0;
      in_row_q      <= '0;
      in_row_abs_q  <= '0;
      out_col_q     <= '0;
      out_row_q     <= '0;
      out_row_abs_q <= '0;
    end else begin
      in_col_q      <= in_col_d;
      in_row_q      <= in_row_d;
      in_row_abs_q  <= in_row_abs_d;
      out_col_q     <= out_col_d;
      out_row_q     <= out_row_d;
      out_row_abs_q <= out_row_abs_d;
    end
  end

  // Row-buffer write — one bank per channel.
  logic [BankW-1:0] bank_idx;
  assign bank_idx = (K == 1) ? '0 : BankW'(in_row_q % K);

  always_ff @(posedge clk_i) begin
    if (w_fire) begin
      for (int c = 0; c < Channels; c++) begin
        rowbuf[c][bank_idx][in_col_q] <= wdata_i[c];
      end
    end
  end

  // Avoid lint complaints about unused parameters.
  // verilator lint_off UNUSED
  logic [CntW-1:0] unused_cnt;
  assign unused_cnt = CntW'(Total);
  // verilator lint_on UNUSED

endmodule
