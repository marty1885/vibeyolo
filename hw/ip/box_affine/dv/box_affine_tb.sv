// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// box_affine_tb — DUT + behavioral REF in lockstep on identical stimulus.
// fp16 outputs are compared in the C++ test with a ULP tolerance (the DUT
// chains fp16_fma per step; the ref rounds once from real), so the only
// hard mismatch surfaced here is valid-timing.

module box_affine_tb (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               valid_i,
  input  logic signed [7:0]  l_i,
  input  logic signed [7:0]  t_i,
  input  logic signed [7:0]  r_i,
  input  logic signed [7:0]  b_i,
  input  logic        [15:0] s_box_i,
  input  logic signed [31:0] col_i,
  input  logic signed [31:0] row_i,
  input  logic signed [31:0] stride_i,

  output logic               valid_dut_o,
  output logic        [15:0] cx_dut_o,
  output logic        [15:0] cy_dut_o,
  output logic        [15:0] w_dut_o,
  output logic        [15:0] h_dut_o,
  output logic               valid_ref_o,
  output logic        [15:0] cx_ref_o,
  output logic        [15:0] cy_ref_o,
  output logic        [15:0] w_ref_o,
  output logic        [15:0] h_ref_o,
  output logic               valid_mismatch_o
);

  box_affine u_dut (
    .clk_i, .rst_ni, .valid_i,
    .l_i, .t_i, .r_i, .b_i, .s_box_i, .col_i, .row_i, .stride_i,
    .valid_o(valid_dut_o),
    .cx_o(cx_dut_o), .cy_o(cy_dut_o), .w_o(w_dut_o), .h_o(h_dut_o)
  );

  box_affine_ref u_ref (
    .clk_i, .rst_ni, .valid_i,
    .l_i, .t_i, .r_i, .b_i, .s_box_i, .col_i, .row_i, .stride_i,
    .valid_o(valid_ref_o),
    .cx_o(cx_ref_o), .cy_o(cy_ref_o), .w_o(w_ref_o), .h_o(h_ref_o)
  );

  assign valid_mismatch_o = (valid_dut_o !== valid_ref_o);

endmodule
