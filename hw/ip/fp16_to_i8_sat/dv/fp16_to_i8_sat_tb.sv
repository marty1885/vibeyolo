// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// fp16_to_i8_sat_tb — Verilator TB wrapper. Runs DUT and REF in lockstep
// on identical stimulus and exposes both outputs plus a mismatch flag.

module fp16_to_i8_sat_tb (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic        [15:0] x_i,

  output logic signed [7:0]  y_dut_o,
  output logic signed [7:0]  y_ref_o,
  output logic               mismatch_o
);

  fp16_to_i8_sat u_dut (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (x_i),
    .y_o    (y_dut_o)
  );

  fp16_to_i8_sat_ref u_ref (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (x_i),
    .y_o    (y_ref_o)
  );

  assign mismatch_o = (y_dut_o !== y_ref_o);

endmodule
