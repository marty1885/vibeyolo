// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_3_m0_cv1_tb — Verilator wrapper. Flattens ports so the C++ test can
// drive 144-lane input windows (3*3*16 bytes = 1152 bits) and read 8-channel
// int8 outputs (64 bits) each cycle.

module layer_3_m0_cv1_tb (
  input  logic                clk_i,
  input  logic                rst_ni,

  input  logic                valid_i,
  // 144 lanes × 8 bits = 1152 bits
  input  logic       [1151:0] x_flat_i,
  // 8 channels × 144 lanes × 8 bits = 9216 bits
  input  logic       [9215:0] w_flat_i,
  // 8 × fp16 = 128 bits
  input  logic        [127:0] scale_flat_i,
  input  logic        [127:0] bias_flat_i,

  output logic                valid_o,
  // 8 × i8 = 64 bits
  output logic         [63:0] y_flat_o
);

  logic signed [143:0][7:0]                 x_w;
  logic signed [7:0][143:0][7:0]            w_w;
  logic        [7:0][15:0]                  scale_w;
  logic        [7:0][15:0]                  bias_w;
  logic signed [7:0][7:0]                   y_w;

  assign x_w     = x_flat_i;
  assign w_w     = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign y_flat_o = y_w;

  layer_3_m0_cv1 u_dut (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .valid_i(valid_i),
    .x_i    (x_w),
    .w_i    (w_w),
    .scale_i(scale_w),
    .bias_i (bias_w),
    .valid_o(valid_o),
    .y_o    (y_w)
  );

endmodule
