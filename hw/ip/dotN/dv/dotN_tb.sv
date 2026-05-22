// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// dotN_tb — Verilator TB wrapper that instantiates DUT and REF in
// lockstep on identical stimulus and exposes both outputs plus a
// mismatch flag. N is passed through as a module parameter so the
// outer Makefile can build one binary per N via -GN=<n>.

module dotN_tb #(
  parameter int N = 16
) (
  input  logic                       clk_i,
  input  logic                       rst_ni,

  input  logic                       en_i,
  input  logic                       clr_i,
  input  logic signed [N-1:0][7:0]   a_i,
  input  logic signed [N-1:0][7:0]   b_i,

  output logic signed [31:0]         y_dut_o,
  output logic signed [31:0]         y_ref_o,
  output logic                       valid_dut_o,
  output logic                       valid_ref_o,
  output logic                       mismatch_o
);

  dotN #(.N(N)) u_dut (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .en_i    (en_i),
    .clr_i   (clr_i),
    .a_i     (a_i),
    .b_i     (b_i),
    .y_o     (y_dut_o),
    .valid_o (valid_dut_o)
  );

  dotN_ref #(.N(N)) u_ref (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .en_i    (en_i),
    .clr_i   (clr_i),
    .a_i     (a_i),
    .b_i     (b_i),
    .y_o     (y_ref_o),
    .valid_o (valid_ref_o)
  );

  // Compare valid_o always; y_o only when at least one side claims valid
  // (matches integration semantics).
  assign mismatch_o = (valid_dut_o !== valid_ref_o) ||
                      ((valid_dut_o || valid_ref_o) &&
                       (y_dut_o !== y_ref_o));

endmodule
