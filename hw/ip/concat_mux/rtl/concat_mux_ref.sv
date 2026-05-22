// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// concat_mux_ref — behavioral golden for concat_mux.
//
// Independently coded reference with the same cycle-accurate handshake
// contract as the DUT but structurally different:
//
//   * The DUT uses a phase bit plus per-stream channel counters and a
//     pixel counter, all next-state regs, with a direct combinational
//     pass-through mux.
//
//   * This REF uses a single integer "beat" counter that advances 0..
//     Ca+Cb-1 within a pixel and 0..Pixels-1 across pixels, plus a one-
//     entry skid representation expressed as integer math. It compares a
//     selected source by deriving `sel = (beat < Ca) ? A : B`.
//
// The two should be cycle-equivalent on identical stimulus.

module concat_mux_ref #(
  parameter int Ca     = 32,
  parameter int Cb     = 32,
  parameter int Pixels = 16
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic              clr_i,

  input  logic              avalid_i,
  output logic              aready_o,
  input  logic signed [7:0] adata_i,

  input  logic              bvalid_i,
  output logic              bready_o,
  input  logic signed [7:0] bdata_i,

  output logic              rvalid_o,
  input  logic              rready_i,
  output logic signed [7:0] rdata_o
);

  int beat_q, beat_d;     // 0..Ca+Cb-1 within current pixel
  int pix_q,  pix_d;      // 0..Pixels-1
  logic sel;              // 0 = pick A, 1 = pick B (derived)

  always_comb begin
    sel = (beat_q < Ca) ? 1'b0 : 1'b1;
  end

  always_comb begin
    beat_d = beat_q;
    pix_d  = pix_q;

    aready_o = 1'b0;
    bready_o = 1'b0;
    rvalid_o = 1'b0;
    rdata_o  = 8'sd0;

    if (sel == 1'b0) begin
      // Pull from A.
      rvalid_o = avalid_i;
      aready_o = rready_i;
      rdata_o  = adata_i;
    end else begin
      // Pull from B.
      rvalid_o = bvalid_i;
      bready_o = rready_i;
      rdata_o  = bdata_i;
    end

    if (rvalid_o && rready_i) begin
      if (beat_q == (Ca + Cb - 1)) begin
        beat_d = 0;
        if (pix_q == Pixels - 1) begin
          pix_d = 0;
        end else begin
          pix_d = pix_q + 1;
        end
      end else begin
        beat_d = beat_q + 1;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      beat_q <= 0;
      pix_q  <= 0;
    end else if (clr_i) begin
      beat_q <= 0;
      pix_q  <= 0;
    end else begin
      beat_q <= beat_d;
      pix_q  <= pix_d;
    end
  end

endmodule
