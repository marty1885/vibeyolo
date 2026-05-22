// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// upsample2 — streaming nearest-neighbor 2x upsample.
//
// Consumes a HxW int8 feature map in row-major raster order (one pixel per
// cycle when valid) and produces a (2H)x(2W) nearest-neighbor 2x upsampled
// stream. Each input pixel is emitted twice horizontally; each input row is
// emitted twice vertically (the second output row replays from a registered
// row buffer of depth W).
//
// Decoupled ready/valid: a one-deep "current pixel" register `cur_q`/`have_q`
// holds the pixel being emitted, so `wready_o` and `rready_i` are
// independent.
//
// State machine:
//   S_FILL   : consume input row into cur_q (emitted twice horizontally,
//              latched into rowbuf on the second emit).
//   S_REPLAY : block input; emit second output row from rowbuf.
//
// Reset / clr_i empties all state: rvalid_o=0, wready_o=1.
//
// Channels parameter is documentation only; only Channels == 1 is
// implemented and tested. Multi-channel interleaving is future work.

module upsample2 #(
  parameter int W        = 16,
  parameter int Channels = 1
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic              clr_i,

  // Input stream (ready/valid)
  input  logic              wvalid_i,
  output logic              wready_o,
  input  logic signed [7:0] wdata_i,

  // Output stream (ready/valid)
  output logic              rvalid_o,
  input  logic              rready_i,
  output logic signed [7:0] rdata_o
);

  // synthesis translate_off
  initial begin
    if (Channels != 1) begin
      $fatal(1, "upsample2: only Channels==1 is supported");
    end
  end
  // synthesis translate_on

  localparam int ColW = (W <= 1) ? 1 : $clog2(W);

  typedef enum logic {
    S_FILL   = 1'b0,
    S_REPLAY = 1'b1
  } state_e;

  state_e             state_q, state_d;
  logic [ColW-1:0]    col_q,   col_d;
  logic               sub_q,   sub_d;
  logic signed [7:0]  cur_q,   cur_d;
  logic               have_q,  have_d;
  logic signed [7:0]  rowbuf [W];

  // Row buffer write enable / index / data, driven combinationally.
  logic               rb_we;
  logic [ColW-1:0]    rb_wa;
  logic signed [7:0]  rb_wd;

  always_comb begin
    state_d = state_q;
    col_d   = col_q;
    sub_d   = sub_q;
    cur_d   = cur_q;
    have_d  = have_q;

    rvalid_o = 1'b0;
    wready_o = 1'b0;
    rdata_o  = 8'sd0;

    rb_we = 1'b0;
    rb_wa = col_q;
    rb_wd = cur_q;

    unique case (state_q)
      S_FILL: begin
        // Accept a new pixel whenever the holding register is empty.
        wready_o = ~have_q;
        rvalid_o = have_q;
        rdata_o  = cur_q;

        // Input handshake: latch.
        if (wvalid_i && wready_o) begin
          cur_d  = wdata_i;
          have_d = 1'b1;
        end

        // Output handshake: emit the pixel twice (sub 0 then 1).
        if (rvalid_o && rready_i) begin
          if (sub_q == 1'b0) begin
            sub_d = 1'b1;
          end else begin
            sub_d = 1'b0;
            // Latch into rowbuf and drop holding register so the next
            // input pixel can be accepted.
            rb_we  = 1'b1;
            rb_wa  = col_q;
            rb_wd  = cur_q;
            have_d = 1'b0;
            if (col_q == ColW'(W-1)) begin
              col_d   = '0;
              state_d = S_REPLAY;
            end else begin
              col_d = col_q + ColW'(1);
            end
          end
        end
      end

      S_REPLAY: begin
        wready_o = 1'b0;
        rvalid_o = 1'b1;
        rdata_o  = rowbuf[col_q];

        if (rready_i) begin
          if (sub_q == 1'b0) begin
            sub_d = 1'b1;
          end else begin
            sub_d = 1'b0;
            if (col_q == ColW'(W-1)) begin
              col_d   = '0;
              state_d = S_FILL;
            end else begin
              col_d = col_q + ColW'(1);
            end
          end
        end
      end

      default: begin
        state_d = S_FILL;
        col_d   = '0;
        sub_d   = 1'b0;
        cur_d   = 8'sd0;
        have_d  = 1'b0;
      end
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= S_FILL;
      col_q   <= '0;
      sub_q   <= 1'b0;
      cur_q   <= 8'sd0;
      have_q  <= 1'b0;
    end else if (clr_i) begin
      state_q <= S_FILL;
      col_q   <= '0;
      sub_q   <= 1'b0;
      cur_q   <= 8'sd0;
      have_q  <= 1'b0;
    end else begin
      state_q <= state_d;
      col_q   <= col_d;
      sub_q   <= sub_d;
      cur_q   <= cur_d;
      have_q  <= have_d;
    end
  end

  // Row buffer write.
  always_ff @(posedge clk_i) begin
    if (rb_we) begin
      rowbuf[rb_wa] <= rb_wd;
    end
  end

endmodule
