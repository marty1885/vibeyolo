// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// sppf_tb — thin Verilator wrapper that exposes the sppf module's
// per-pixel int8 streaming interface to the C++ driver. Uses a small
// W/H to keep build time tight; C is fixed to 128 (matches /model.9).

module sppf_tb #(
  parameter int H = 20,
  parameter int W = 20,
  parameter int C = 128,
  parameter int K = 5
) (
  input  logic                              clk_i,
  input  logic                              rst_ni,
  input  logic                              start_i,

  input  logic                              ivalid_i,
  output logic                              iready_o,
  input  logic signed [C-1:0][7:0]          idata_i,

  output logic                              ovalid_o,
  input  logic                              oready_i,
  output logic signed [4*C-1:0][7:0]        odata_o,

  output logic                              done_o
);

  sppf #(.H(H), .W(W), .C(C), .K(K)) u_dut (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .start_i (start_i),
    .ivalid_i(ivalid_i),
    .iready_o(iready_o),
    .idata_i (idata_i),
    .ovalid_o(ovalid_o),
    .oready_i(oready_i),
    .odata_o (odata_o),
    .done_o  (done_o)
  );

endmodule
