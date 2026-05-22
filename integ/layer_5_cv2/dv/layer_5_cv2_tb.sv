// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_5_cv2_tb — Verilator wrapper around layer_5_cv2. Flattens ports so
// the C++ TB can drive 48-lane input vectors (48*8 = 384 bits) and read
// 64-channel int8 outputs (64*8 = 512 bits) each cycle.

module layer_5_cv2_tb (
  input  logic                clk_i,
  input  logic                rst_ni,

  input  logic                valid_i,
  // 48 lanes × 8 bits = 384 bits
  input  logic        [383:0] x_flat_i,
  // 64 channels × 48 lanes × 8 bits = 24576 bits
  input  logic      [24575:0] w_flat_i,
  // 64 × fp16 = 1024 bits
  input  logic       [1023:0] scale_flat_i,
  input  logic       [1023:0] bias_flat_i,

  output logic                valid_o,
  // 64 × i8 = 512 bits
  output logic        [511:0] y_flat_o
);

  logic signed [47:0][7:0]                  x_w;
  logic signed [63:0][47:0][7:0]            w_w;
  logic        [63:0][15:0]                 scale_w;
  logic        [63:0][15:0]                 bias_w;
  logic signed [63:0][7:0]                  y_w;

  assign x_w     = x_flat_i;
  assign w_w     = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign y_flat_o = y_w;

  layer_5_cv2 u_dut (
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
