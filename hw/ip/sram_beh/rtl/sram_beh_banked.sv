// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// sram_beh_banked — a logical (LWidth × LDepth) 1R1W memory tiled into a bank
// of physical sram_beh macros, exactly as tools/sram_model.py:tile_memory()
// counts them:
//
//   NW = ceil(LWidth / MaxWidth)   macros across  (to make the word wide)
//   ND = ceil(LDepth / MaxDepth)   macros down    (to make it deep)
//   -> NW * ND  physical macros
//
// This is the drop-in replacement for a "magic" wide activation array
// (`logic [LWidth-1:0] arr [LDepth]`): it presents the SAME single-word-per-
// cycle 1R1W port, but its existence makes the bandwidth cost explicit — you
// physically pay NW macros to move LWidth bits in one cycle, and ND macros to
// reach LDepth. Set MaxWidth/MaxDepth to the chosen compiled-macro limits
// (mirror tools/sram_model.py MAX_WIDTH / MAX_DEPTH).
//
// Read latency through the wrapper is the macro ReadLat (the depth-bank output
// mux is combinational on the registered macro outputs).

module sram_beh_banked #(
  parameter int LWidth   = 1024,
  parameter int LDepth   = 6400,
  parameter int MaxWidth = 144,
  parameter int MaxDepth = 4096,
  parameter int ReadLat  = 1,

  localparam int NW    = (LWidth + MaxWidth - 1) / MaxWidth,
  localparam int ND    = (LDepth + MaxDepth - 1) / MaxDepth,
  localparam int AddrW = (LDepth <= 1) ? 1 : $clog2(LDepth),
  localparam int BankW = (ND    <= 1) ? 1 : $clog2(ND)
) (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               w_en_i,
  input  logic [AddrW-1:0]    w_addr_i,
  input  logic [LWidth-1:0]   w_data_i,

  input  logic               r_en_i,
  input  logic [AddrW-1:0]    r_addr_i,
  output logic [LWidth-1:0]   r_data_o
);

  // Per-macro depth (rows actually used in each depth-bank).
  localparam int RowsPerBank = (LDepth + ND - 1) / ND;
  localparam int OffW = (RowsPerBank <= 1) ? 1 : $clog2(RowsPerBank);

  // Decode depth-bank select (high) and in-bank offset (low) for each port.
  logic [BankW-1:0] w_bank, r_bank;
  logic [OffW-1:0]  w_off,  r_off;
  if (ND <= 1) begin : g_one_bank
    assign w_bank = '0; assign r_bank = '0;
    assign w_off  = w_addr_i[OffW-1:0];
    assign r_off  = r_addr_i[OffW-1:0];
  end else begin : g_multi_bank
    assign w_bank = w_addr_i[AddrW-1:OffW];
    assign r_bank = r_addr_i[AddrW-1:OffW];
    assign w_off  = w_addr_i[OffW-1:0];
    assign r_off  = r_addr_i[OffW-1:0];
  end

  // Read-bank select must be delayed by ReadLat to align with macro output.
  logic [BankW-1:0] r_bank_q [ReadLat];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) for (int i = 0; i < ReadLat; i++) r_bank_q[i] <= '0;
    else begin
      r_bank_q[0] <= r_bank;
      for (int i = 1; i < ReadLat; i++) r_bank_q[i] <= r_bank_q[i-1];
    end
  end

  // Macro grid: per depth-bank d, per width-slice w.
  logic [LWidth-1:0] bank_rdata [ND];

  for (genvar d = 0; d < ND; d++) begin : g_depth
    logic                bank_w_en, bank_r_en;
    assign bank_w_en = w_en_i && (BankW'(d) == w_bank);
    assign bank_r_en = r_en_i && (BankW'(d) == r_bank);

    for (genvar w = 0; w < NW; w++) begin : g_width
      localparam int Lo = w * MaxWidth;
      localparam int Wb = (Lo + MaxWidth <= LWidth) ? MaxWidth : (LWidth - Lo);
      sram_beh #(.Width(Wb), .Depth(RowsPerBank), .ReadLat(ReadLat)) u_mac (
        .clk_i, .rst_ni,
        .w_en_i  (bank_w_en),
        .w_addr_i(w_off),
        .w_data_i(w_data_i[Lo +: Wb]),
        .r_en_i  (bank_r_en),
        .r_addr_i(r_off),
        .r_data_o(bank_rdata[d][Lo +: Wb])
      );
    end
  end

  // Output mux: pick the depth-bank that the delayed read addressed.
  assign r_data_o = bank_rdata[r_bank_q[ReadLat-1]];

endmodule
