// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// box_decode_tb — Verilator TB wrapper. Drives DUT and REF with the same
// 4×16 fp16 probability vectors plus (cx, cy, stride), and exposes both
// xyxy outputs. The C++ test does the ULP tolerance check.

module box_decode_tb (
  input  logic         clk_i,
  input  logic         rst_ni,

  input  logic         valid_i,
  input  logic [255:0] p_l_flat_i,    // 16 × 16-bit lanes, lane 0 in LSBs
  input  logic [255:0] p_t_flat_i,
  input  logic [255:0] p_r_flat_i,
  input  logic [255:0] p_b_flat_i,
  input  logic signed [15:0] cx_i,
  input  logic signed [15:0] cy_i,
  input  logic signed [15:0] stride_i,

  output logic         valid_dut_o,
  output logic         valid_ref_o,
  output logic [15:0]  x1_dut_o,
  output logic [15:0]  y1_dut_o,
  output logic [15:0]  x2_dut_o,
  output logic [15:0]  y2_dut_o,
  output logic [15:0]  x1_ref_o,
  output logic [15:0]  y1_ref_o,
  output logic [15:0]  x2_ref_o,
  output logic [15:0]  y2_ref_o
);

  logic [15:0] p_l [16], p_t [16], p_r [16], p_b [16];

  always_comb begin
    for (int i = 0; i < 16; i++) begin
      p_l[i] = p_l_flat_i[16*i +: 16];
      p_t[i] = p_t_flat_i[16*i +: 16];
      p_r[i] = p_r_flat_i[16*i +: 16];
      p_b[i] = p_b_flat_i[16*i +: 16];
    end
  end

  box_decode u_dut (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .valid_i (valid_i),
    .p_l_i   (p_l),
    .p_t_i   (p_t),
    .p_r_i   (p_r),
    .p_b_i   (p_b),
    .cx_i    (cx_i),
    .cy_i    (cy_i),
    .stride_i(stride_i),
    .valid_o (valid_dut_o),
    .x1_o    (x1_dut_o),
    .y1_o    (y1_dut_o),
    .x2_o    (x2_dut_o),
    .y2_o    (y2_dut_o)
  );

  box_decode_ref u_ref (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .valid_i (valid_i),
    .p_l_i   (p_l),
    .p_t_i   (p_t),
    .p_r_i   (p_r),
    .p_b_i   (p_b),
    .cx_i    (cx_i),
    .cy_i    (cy_i),
    .stride_i(stride_i),
    .valid_o (valid_ref_o),
    .x1_o    (x1_ref_o),
    .y1_o    (y1_ref_o),
    .x2_o    (x2_ref_o),
    .y2_o    (y2_ref_o)
  );

endmodule
