// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// mac8_tb — Verilator TB wrapper that runs DUT and REF in lockstep on
// identical stimulus and exposes both accumulators plus a mismatch flag.

module mac8_tb (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               clr_i,
  input  logic               en_i,
  input  logic signed [7:0]  a_i,
  input  logic signed [7:0]  b_i,

  output logic signed [31:0] acc_dut_o,
  output logic signed [31:0] acc_ref_o,
  output logic               mismatch_o
);

  mac8 u_dut (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .clr_i  (clr_i),
    .en_i   (en_i),
    .a_i    (a_i),
    .b_i    (b_i),
    .acc_o  (acc_dut_o)
  );

  mac8_ref u_ref (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .clr_i  (clr_i),
    .en_i   (en_i),
    .a_i    (a_i),
    .b_i    (b_i),
    .acc_o  (acc_ref_o)
  );

  assign mismatch_o = (acc_dut_o !== acc_ref_o);

endmodule
