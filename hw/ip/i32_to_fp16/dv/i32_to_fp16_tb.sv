// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// i32_to_fp16_tb — Verilator TB wrapper. Runs DUT and REF in lockstep on
// identical stimulus and exposes both outputs (y and shift) plus a
// mismatch flag.

module i32_to_fp16_tb (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic signed [31:0] x_i,

  output logic        [15:0] y_dut_o,
  output logic        [15:0] y_ref_o,
  output logic        [4:0]  shift_dut_o,
  output logic        [4:0]  shift_ref_o,
  output logic               mismatch_o
);

  i32_to_fp16 u_dut (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (x_i),
    .y_o    (y_dut_o),
    .shift_o(shift_dut_o)
  );

  i32_to_fp16_ref u_ref (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (x_i),
    .y_o    (y_ref_o),
    .shift_o(shift_ref_o)
  );

  assign mismatch_o = (y_dut_o !== y_ref_o) || (shift_dut_o !== shift_ref_o);

endmodule
