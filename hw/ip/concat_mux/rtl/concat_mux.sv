// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// concat_mux — channel-dimension concatenation of two int8 streams.
//
// Used in C3k2 / FPN where two feature maps of the same spatial extent
// (Pixels = H*W positions) but with Ca and Cb channels respectively are
// concatenated along the channel axis. Both inputs deliver channel-by-
// channel-interleaved-at-pixel-granularity raster scan: for each pixel
// position, all channels for that pixel are emitted in sequence, then the
// next pixel position.
//
// Output for each pixel position p of Pixels:
//   stream A's Ca channels at p, in order
//   then stream B's Cb channels at p, in order
//
// Implementation: pure pass-through mux with a phase counter. No buffering
// of more than wire data.
//
//   * phase_q == 0 : route A->R, accept Ca samples
//   * phase_q == 1 : route B->R, accept Cb samples
//
// Reset / clr_i: phase=0, ch=0, pix=0 -> aready_o follows rready_i,
// bready_o=0, rvalid_o follows avalid_i.

module concat_mux #(
  parameter int Ca     = 32,
  parameter int Cb     = 32,
  parameter int Pixels = 16
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic              clr_i,

  // Stream A
  input  logic              avalid_i,
  output logic              aready_o,
  input  logic signed [7:0] adata_i,

  // Stream B
  input  logic              bvalid_i,
  output logic              bready_o,
  input  logic signed [7:0] bdata_i,

  // Output
  output logic              rvalid_o,
  input  logic              rready_i,
  output logic signed [7:0] rdata_o
);

  localparam int CaW  = (Ca     <= 1) ? 1 : $clog2(Ca);
  localparam int CbW  = (Cb     <= 1) ? 1 : $clog2(Cb);
  localparam int PixW = (Pixels <= 1) ? 1 : $clog2(Pixels);

  // phase 0 = A active, phase 1 = B active.
  logic            phase_q, phase_d;
  logic [CaW-1:0]  acnt_q,  acnt_d;
  logic [CbW-1:0]  bcnt_q,  bcnt_d;
  logic [PixW-1:0] pix_q,   pix_d;

  always_comb begin
    phase_d = phase_q;
    acnt_d  = acnt_q;
    bcnt_d  = bcnt_q;
    pix_d   = pix_q;

    aready_o = 1'b0;
    bready_o = 1'b0;
    rvalid_o = 1'b0;
    rdata_o  = 8'sd0;

    if (phase_q == 1'b0) begin
      // A is the active source: directly forward the A handshake.
      rvalid_o = avalid_i;
      aready_o = rready_i;
      rdata_o  = adata_i;

      if (avalid_i && rready_i) begin
        if (acnt_q == CaW'(Ca-1)) begin
          acnt_d = '0;
          // After last A channel: if Cb>0 advance to phase B, else this
          // pixel is done; advance pix counter.
          if (Cb == 0) begin
            // Degenerate (no B channels): just advance pixel.
            if (pix_q == PixW'(Pixels-1)) begin
              pix_d = '0;
            end else begin
              pix_d = pix_q + PixW'(1);
            end
          end else begin
            phase_d = 1'b1;
          end
        end else begin
          acnt_d = acnt_q + CaW'(1);
        end
      end
    end else begin
      // B is the active source.
      rvalid_o = bvalid_i;
      bready_o = rready_i;
      rdata_o  = bdata_i;

      if (bvalid_i && rready_i) begin
        if (bcnt_q == CbW'(Cb-1)) begin
          bcnt_d  = '0;
          phase_d = 1'b0;
          if (pix_q == PixW'(Pixels-1)) begin
            pix_d = '0;
          end else begin
            pix_d = pix_q + PixW'(1);
          end
        end else begin
          bcnt_d = bcnt_q + CbW'(1);
        end
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      phase_q <= 1'b0;
      acnt_q  <= '0;
      bcnt_q  <= '0;
      pix_q   <= '0;
    end else if (clr_i) begin
      phase_q <= 1'b0;
      acnt_q  <= '0;
      bcnt_q  <= '0;
      pix_q   <= '0;
    end else begin
      phase_q <= phase_d;
      acnt_q  <= acnt_d;
      bcnt_q  <= bcnt_d;
      pix_q   <= pix_d;
    end
  end

  // Unused signal to keep pix_q live for future debug; lint suppression.
  logic _unused;
  assign _unused = ^{pix_q};

endmodule
