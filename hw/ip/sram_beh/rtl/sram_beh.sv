// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// sram_beh — behavioral model of a single compiled SRAM macro.
//
// This is the RTL counterpart of the placeholder cost model in
// tools/sram_model.py. It exists so the chip stops assuming "magic" memory
// (full-width, zero-latency, unlimited ports) and instead instantiates objects
// shaped like what a foundry MemoryCompiler actually emits: ONE word per port
// per cycle, a fixed Width, a fixed Depth, and a fixed read latency.
//
// Port flavor: 1R1W (one write port + one independent read port). 1RW is the
// degenerate case (tie r_en/w_en mutually exclusive); 2RW would add a second
// read/write pair. To deliver more than `Width` bits/cycle, instantiate a bank
// of these via sram_beh_banked — you cannot widen a single macro arbitrarily
// (that is exactly the bandwidth limit we are now modeling).
//
// Read-during-write to the same address returns OLD data (read-before-write),
// the common compiler default. ReadLat is the registered read latency
// (1 = data valid the cycle after r_en; 2 = pipelined output reg).
//
// ┌─────────────────── MEMORYCOMPILER SWAP POINT ───────────────────┐
// │ At tape-out, replace the `mem`/read pipeline below with the      │
// │ generated macro instance (e.g. TS1N16*..). The port list here   │
// │ is intentionally a 1R1W superset of the typical compiled pinout.│
// └──────────────────────────────────────────────────────────────────┘

module sram_beh #(
  parameter int Width   = 64,   // bits per word (one macro's max access width)
  parameter int Depth   = 1024, // words
  parameter int ReadLat = 1,    // registered read latency (1 or 2)

  localparam int AddrW = (Depth <= 1) ? 1 : $clog2(Depth)
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  // Write port.
  input  logic              w_en_i,
  input  logic [AddrW-1:0]  w_addr_i,
  input  logic [Width-1:0]  w_data_i,

  // Read port.
  input  logic              r_en_i,
  input  logic [AddrW-1:0]  r_addr_i,
  output logic [Width-1:0]  r_data_o
);

  // ── Storage ────────────────────────────────────────────────────────
  logic [Width-1:0] mem [Depth];

  // ── Write port ─────────────────────────────────────────────────────
  always_ff @(posedge clk_i) begin
    if (w_en_i) mem[w_addr_i] <= w_data_i;
  end

  // ── Read port (read-before-write; ReadLat registered stages) ───────
  logic [Width-1:0] rd_stage1;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)     rd_stage1 <= '0;
    else if (r_en_i) rd_stage1 <= mem[r_addr_i];
  end

  if (ReadLat <= 1) begin : g_lat1
    assign r_data_o = rd_stage1;
  end else begin : g_lat2
    logic [Width-1:0] rd_stage2;
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) rd_stage2 <= '0;
      else         rd_stage2 <= rd_stage1;
    end
    assign r_data_o = rd_stage2;
  end

endmodule
