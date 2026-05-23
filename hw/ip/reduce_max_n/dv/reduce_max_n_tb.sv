// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// reduce_max_n_tb — instantiates DUT and REF in lockstep on identical
// stimulus, exposes both outputs plus a mismatch flag. N is a module
// parameter so the Makefile builds one binary per N via -GN=<n>.

module reduce_max_n_tb #(
  parameter int N = 80
) (
  input  logic                          clk_i,
  input  logic                          rst_ni,

  input  logic                          en_i,
  input  logic signed [N-1:0][7:0]      x_i,

  output logic                          valid_dut_o,
  output logic signed [7:0]             y_dut_o,
  output logic                          valid_ref_o,
  output logic signed [7:0]             y_ref_o,
  output logic                          mismatch_o
);

  reduce_max_n #(.N(N)) u_dut (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .en_i    (en_i),
    .x_i     (x_i),
    .valid_o (valid_dut_o),
    .y_o     (y_dut_o)
  );

  reduce_max_n_ref #(.N(N)) u_ref (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .en_i    (en_i),
    .x_i     (x_i),
    .valid_o (valid_ref_o),
    .y_o     (y_ref_o)
  );

  assign mismatch_o = (y_dut_o !== y_ref_o) || (valid_dut_o !== valid_ref_o);

endmodule
