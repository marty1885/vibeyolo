// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// upsample2_ref — behavioral golden for upsample2.
//
// Independently coded reference with the same cycle-accurate handshake
// contract as the DUT but structurally different:
//
//   * The DUT uses an enum FSM (S_FILL / S_REPLAY) + (sub, col) counters
//     + a packed-array rowbuf, with rb_we wired combinationally.
//
//   * This REF uses a single integer phase counter that runs across the
//     entire 4*W output beats of one row-pair, an integer mode (0=fill,
//     1=replay), a captured row stored as a SystemVerilog queue, and a
//     plain (valid, data) register pair for the input holding stage.
//     All updates use non-blocking assigns so the cycle behaviour matches
//     the DUT (which uses next-state regs).

module upsample2_ref #(
  parameter int W        = 16,
  parameter int Channels = 1
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic              clr_i,

  input  logic              wvalid_i,
  output logic              wready_o,
  input  logic signed [7:0] wdata_i,

  output logic              rvalid_o,
  input  logic              rready_i,
  output logic signed [7:0] rdata_o
);

  // synthesis translate_off
  initial begin
    if (Channels != 1) begin
      $fatal(1, "upsample2_ref: only Channels==1 is supported");
    end
  end
  // synthesis translate_on

  // mode_q: 0 = first-row fill+emit, 1 = replay second row.
  logic              mode_q,    mode_d;
  // first/replay pixel counters (0..W-1).
  int                fcnt_q,    fcnt_d;
  int                rcnt_q,    rcnt_d;
  // horizontal sub-pixel index: 0 emits first copy, 1 emits second copy.
  logic              sub_q,     sub_d;
  // holding-stage for the active input pixel (only in fill mode).
  logic              hold_v_q,  hold_v_d;
  logic signed [7:0] hold_d_q,  hold_d_d;
  // captured first-row queue (push on first-emit complete, pop on
  // replay-emit complete).
  logic signed [7:0] row1 [$];
  // We need to schedule queue pushes/pops in the same posedge but as
  // "deferred" actions so combinational outputs read pre-update values.
  logic              row1_push_q;
  logic signed [7:0] row1_push_d_q;
  logic              row1_pop_q;

  always_comb begin
    mode_d   = mode_q;
    fcnt_d   = fcnt_q;
    rcnt_d   = rcnt_q;
    sub_d    = sub_q;
    hold_v_d = hold_v_q;
    hold_d_d = hold_d_q;

    if (mode_q == 1'b0) begin
      // Fill / first-row emit.
      wready_o = ~hold_v_q;
      rvalid_o = hold_v_q;
      rdata_o  = hold_d_q;

      if (wvalid_i && wready_o) begin
        hold_v_d = 1'b1;
        hold_d_d = wdata_i;
      end

      if (rvalid_o && rready_i) begin
        if (sub_q == 1'b0) begin
          sub_d = 1'b1;
        end else begin
          sub_d    = 1'b0;
          hold_v_d = 1'b0;        // drop holding register
          if (fcnt_q == W-1) begin
            fcnt_d = 0;
            mode_d = 1'b1;
          end else begin
            fcnt_d = fcnt_q + 1;
          end
        end
      end
    end else begin
      // Replay second row.
      wready_o = 1'b0;
      // We read the head of row1 as long as a row1 is populated. The
      // queue is guaranteed non-empty in replay because exactly W pixels
      // were pushed during fill.
      rvalid_o = (row1.size() != 0);
      rdata_o  = (row1.size() != 0) ? row1[0] : 8'sd0;

      if (rvalid_o && rready_i) begin
        if (sub_q == 1'b0) begin
          sub_d = 1'b1;
        end else begin
          sub_d  = 1'b0;
          if (rcnt_q == W-1) begin
            rcnt_d = 0;
            mode_d = 1'b0;
          end else begin
            rcnt_d = rcnt_q + 1;
          end
        end
      end
    end
  end

  // Deferred queue actions:
  // * Push to row1 when fill mode completes a pixel (second copy & sub_q==1).
  // * Pop from row1 when replay mode completes a pixel.
  always_comb begin
    row1_push_q   = 1'b0;
    row1_push_d_q = hold_d_q;
    row1_pop_q    = 1'b0;
    if (mode_q == 1'b0) begin
      if (rvalid_o && rready_i && (sub_q == 1'b1)) begin
        row1_push_q = 1'b1;
      end
    end else begin
      if (rvalid_o && rready_i && (sub_q == 1'b1)) begin
        row1_pop_q = 1'b1;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      mode_q   <= 1'b0;
      fcnt_q   <= 0;
      rcnt_q   <= 0;
      sub_q    <= 1'b0;
      hold_v_q <= 1'b0;
      hold_d_q <= 8'sd0;
      row1.delete();
    end else if (clr_i) begin
      mode_q   <= 1'b0;
      fcnt_q   <= 0;
      rcnt_q   <= 0;
      sub_q    <= 1'b0;
      hold_v_q <= 1'b0;
      hold_d_q <= 8'sd0;
      row1.delete();
    end else begin
      mode_q   <= mode_d;
      fcnt_q   <= fcnt_d;
      rcnt_q   <= rcnt_d;
      sub_q    <= sub_d;
      hold_v_q <= hold_v_d;
      hold_d_q <= hold_d_d;
      if (row1_push_q) row1.push_back(row1_push_d_q);
      if (row1_pop_q ) void'(row1.pop_front());
    end
  end

endmodule
