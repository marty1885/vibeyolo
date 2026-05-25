// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// fp16_macw_tb — TB wrapper. Exposes the DUT's a/b/c/y so the C++ test can
// cross-check against an independent long-double reference. Width is the
// default (ACC_EXP=8, ACC_MANT=21 → 30-bit accumulator) so the wide ports
// fit a single 32-bit Verilator word.

module fp16_macw_tb #(
  parameter int unsigned ACC_EXP  = 8,
  parameter int unsigned ACC_MANT = 21,
  parameter int unsigned ACC_W    = 1 + ACC_EXP + ACC_MANT
) (
  input  logic              clk_i,
  input  logic              rst_ni,
  input  logic [15:0]       a_i,
  input  logic [15:0]       b_i,
  input  logic [ACC_W-1:0]  c_i,
  output logic [ACC_W-1:0]  y_o
);
  fp16_macw #(.ACC_EXP(ACC_EXP), .ACC_MANT(ACC_MANT)) u_dut (
    .clk_i, .rst_ni, .a_i, .b_i, .c_i, .y_o
  );
endmodule
