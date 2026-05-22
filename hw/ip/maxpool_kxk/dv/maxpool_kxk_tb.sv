// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// maxpool_kxk_tb — Verilator TB wrapper that instantiates DUT and REF
// in lockstep on identical stimulus and exposes both outputs plus a
// mismatch flag. K is passed through as a module parameter so the
// outer Makefile can build one binary per K via -GK=<n>.

module maxpool_kxk_tb #(
  parameter int K = 5
) (
  input  logic                              clk_i,
  input  logic                              rst_ni,

  input  logic                              en_i,
  input  logic signed [K*K-1:0][7:0]        x_i,

  output logic signed [7:0]                 y_dut_o,
  output logic signed [7:0]                 y_ref_o,
  output logic                              mismatch_o
);

  maxpool_kxk #(.K(K)) u_dut (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .en_i   (en_i),
    .x_i    (x_i),
    .y_o    (y_dut_o)
  );

  maxpool_kxk_ref #(.K(K)) u_ref (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .en_i   (en_i),
    .x_i    (x_i),
    .y_o    (y_ref_o)
  );

  assign mismatch_o = (y_dut_o !== y_ref_o);

endmodule
