// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// weight_rom — per-layer int8 weight + fp16 scale + fp16 bias ROM.
//
// Holds three parallel arrays keyed by output-channel index:
//   - Wq:     int8 weights, packed Kh*Kw*Ic bytes per output channel
//   - scale:  fp16 per output channel
//   - bias:   fp16 per output channel
//
// Initialized from $readmemh files (three of them) keyed off InitFile:
//   <InitFile>.w.hex  — weight rows, one Kh*Kw*Ic-byte row per line
//   <InitFile>.s.hex  — fp16 scale,  one 16-bit value per line
//   <InitFile>.b.hex  — fp16 bias,   one 16-bit value per line
// When InitFile is empty (default), the memories stay zero-init (good
// for sim startup); $readmemh is skipped to avoid a missing-file error.
//
// Read-only single port. 1-cycle latency, prim_ram_1p-style: assert req_i
// and oc_addr_i on cycle T → outputs valid on cycle T+1. With req_i=0
// the outputs hold their previous value.

module weight_rom #(
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

  // ── Storage ─────────────────────────────────────────────
  logic [RowBits-1:0] wmem [Oc];
  logic [15:0]        smem [Oc];
  logic [15:0]        bmem [Oc];

  // ── Init ────────────────────────────────────────────────
  // Zero-fill so simulator state is deterministic even when the hex
  // files don't cover every entry (or InitFile is empty).
  initial begin
    for (int i = 0; i < Oc; i++) begin
      wmem[i] = '0;
      smem[i] = '0;
      bmem[i] = '0;
    end
    if (InitFile != "") begin
      $readmemh({InitFile, ".w.hex"}, wmem);
      $readmemh({InitFile, ".s.hex"}, smem);
      $readmemh({InitFile, ".b.hex"}, bmem);
    end
  end

  // ── Read port (1-cycle latency, hold on !req_i) ─────────
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      w_row_o <= '0;
      scale_o <= '0;
      bias_o  <= '0;
    end else if (req_i) begin
      w_row_o <= wmem[oc_addr_i];
      scale_o <= smem[oc_addr_i];
      bias_o  <= bmem[oc_addr_i];
    end
  end

endmodule
