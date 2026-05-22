// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// weight_rom_tb — Verilator TB wrapper that runs DUT and REF in lockstep
// on identical stimulus and exposes scale/bias plus an indexed byte view
// of the wide weight row so the C++ test can sweep it without needing
// to crack Verilator's multi-word packed encoding.

module weight_rom_tb #(
  parameter int    Kh       = 3,
  parameter int    Kw       = 3,
  parameter int    Ic       = 4,
  parameter int    Oc       = 8,
  parameter string InitFile = "",

  localparam int RowLen  = Kh * Kw * Ic,
  localparam int RowBits = RowLen * 8,
  localparam int AddrW   = (Oc <= 1) ? 1 : $clog2(Oc)
) (
  input  logic                clk_i,
  input  logic                rst_ni,

  input  logic                req_i,
  input  logic [AddrW-1:0]    oc_addr_i,

  // Byte-index probe: caller selects byte 0..RowLen-1 of the registered
  // weight row to inspect. Combinational byte slice of the registered
  // RowBits word so it follows the same 1-cycle latency.
  input  logic [31:0]         byte_idx_i,

  output logic [7:0]          w_byte_dut_o,
  output logic [7:0]          w_byte_ref_o,
  output logic [15:0]         scale_dut_o,
  output logic [15:0]         scale_ref_o,
  output logic [15:0]         bias_dut_o,
  output logic [15:0]         bias_ref_o,
  output logic                mismatch_o
);

  logic [RowBits-1:0] w_row_dut, w_row_ref;

  weight_rom #(
    .Kh(Kh), .Kw(Kw), .Ic(Ic), .Oc(Oc), .InitFile(InitFile)
  ) u_dut (
    .clk_i     (clk_i),
    .rst_ni    (rst_ni),
    .req_i     (req_i),
    .oc_addr_i (oc_addr_i),
    .w_row_o   (w_row_dut),
    .scale_o   (scale_dut_o),
    .bias_o    (bias_dut_o)
  );

  weight_rom_ref #(
    .Kh(Kh), .Kw(Kw), .Ic(Ic), .Oc(Oc), .InitFile(InitFile)
  ) u_ref (
    .clk_i     (clk_i),
    .rst_ni    (rst_ni),
    .req_i     (req_i),
    .oc_addr_i (oc_addr_i),
    .w_row_o   (w_row_ref),
    .scale_o   (scale_ref_o),
    .bias_o    (bias_ref_o)
  );

  // Byte selector — clamp to a valid index to keep lint happy.
  logic [31:0] idx;
  assign idx = (byte_idx_i < RowLen) ? byte_idx_i : 32'd0;
  assign w_byte_dut_o = w_row_dut[idx*8 +: 8];
  assign w_byte_ref_o = w_row_ref[idx*8 +: 8];

  assign mismatch_o = (w_row_dut   !== w_row_ref)
                    | (scale_dut_o !== scale_ref_o)
                    | (bias_dut_o  !== bias_ref_o);

endmodule
