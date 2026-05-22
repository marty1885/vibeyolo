// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// upsample_concat_tb — Verilator wrapper for the model.14/15 upsample
// integration block. Parameters fixed to the P3->detect neck shapes.

module upsample_concat_tb #(
  parameter int H_A = 40,
  parameter int W_A = 40,
  parameter int C_A = 128,
  parameter int H_B = 80,
  parameter int W_B = 80,
  parameter int C_B = 128
) (
  input  logic                              clk_i,
  input  logic                              rst_ni,
  input  logic                              start_i,

  input  logic                              avalid_i,
  output logic                              aready_o,
  input  logic signed [C_A-1:0][7:0]        adata_i,

  input  logic                              bvalid_i,
  output logic                              bready_o,
  input  logic signed [C_B-1:0][7:0]        bdata_i,

  output logic                              ovalid_o,
  input  logic                              oready_i,
  output logic signed [C_A+C_B-1:0][7:0]    odata_o,

  output logic                              done_o
);

  upsample_concat #(
    .H_A(H_A), .W_A(W_A), .C_A(C_A),
    .H_B(H_B), .W_B(W_B), .C_B(C_B)
  ) u_dut (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .start_i (start_i),
    .avalid_i(avalid_i),
    .aready_o(aready_o),
    .adata_i (adata_i),
    .bvalid_i(bvalid_i),
    .bready_o(bready_o),
    .bdata_i (bdata_i),
    .ovalid_o(ovalid_o),
    .oready_i(oready_i),
    .odata_o (odata_o),
    .done_o  (done_o)
  );

endmodule
