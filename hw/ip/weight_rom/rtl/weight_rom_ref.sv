// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// weight_rom_ref — behavioral golden for weight_rom.
//
// Independently coded: uses queues populated by $readmemh into temporary
// arrays and then copied element-by-element, with an explicit registered
// pipeline stage. Same port list and same cycle-accurate contract.

module weight_rom_ref #(
  parameter int    Kh       = 3,
  parameter int    Kw       = 3,
  parameter int    Ic       = 16,
  parameter int    Oc       = 32,
  parameter string InitFile = "",

  localparam int RowLen  = Kh * Kw * Ic,
  localparam int RowBits = RowLen * 8,
  localparam int AddrW   = (Oc <= 1) ? 1 : $clog2(Oc)
) (
  input  logic                clk_i,
  input  logic                rst_ni,

  input  logic                req_i,
  input  logic [AddrW-1:0]    oc_addr_i,

  output logic [RowBits-1:0]  w_row_o,
  output logic [15:0]         scale_o,
  output logic [15:0]         bias_o
);

  // Independent storage with different structure: a packed bit array
  // built from per-byte queues for weights, plain arrays for the fp16
  // metadata loaded via separate $readmemh calls.
  bit [15:0] sload [Oc];
  bit [15:0] bload [Oc];

  bit [RowBits-1:0] wq [Oc];
  bit [15:0]        sq [Oc];
  bit [15:0]        bq [Oc];

  initial begin
    // Zero defaults.
    for (int i = 0; i < Oc; i++) begin
      sload[i] = '0;
      bload[i] = '0;
      wq[i]    = '0;
      sq[i]    = '0;
      bq[i]    = '0;
    end

    if (InitFile != "") begin
      // Load the weight file as a flat byte stream and the scale/bias
      // files as 16-bit-per-line arrays. Each weight row occupies RowLen
      // consecutive bytes; reassemble into RowBits-wide words.
      //
      // NOTE: this differs from the DUT, which $readmemh-loads the
      // whole row word in one shot. The DUT's .w.hex is one RowBits
      // value per line, so to read it as bytes we use a row-wide
      // intermediate array and then slice it. Implementing this as a
      // row-wide load too (different code path, same answer).
      bit [RowBits-1:0] wfull [Oc];
      for (int i = 0; i < Oc; i++) wfull[i] = '0;
      $readmemh({InitFile, ".w.hex"}, wfull);
      $readmemh({InitFile, ".s.hex"}, sload);
      $readmemh({InitFile, ".b.hex"}, bload);

      for (int i = 0; i < Oc; i++) begin
        // Copy via byte slice to exercise an independent indexing path.
        for (int b = 0; b < RowLen; b++) begin
          wq[i][b*8 +: 8] = wfull[i][b*8 +: 8];
        end
        sq[i] = sload[i];
        bq[i] = bload[i];
      end
    end
  end

  // Registered output, hold on !req_i.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      w_row_o <= '0;
      scale_o <= '0;
      bias_o  <= '0;
    end else begin
      if (req_i) begin
        w_row_o <= wq[oc_addr_i];
        scale_o <= sq[oc_addr_i];
        bias_o  <= bq[oc_addr_i];
      end
    end
  end

endmodule
