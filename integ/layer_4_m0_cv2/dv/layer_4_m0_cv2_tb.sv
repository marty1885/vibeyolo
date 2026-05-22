// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_4_m0_cv2_tb — Verilator wrapper. Flattens ports so the C++ test can
// drive 72-lane input windows (3*3*8 = 576 bits) plus 16-channel residual
// samples (128 bits), and read 16-channel int8 outputs (128 bits) each cycle.

module layer_4_m0_cv2_tb (
  input  logic                clk_i,
  input  logic                rst_ni,

  input  logic                valid_i,
  // 72 lanes × 8 bits = 576 bits
  input  logic        [575:0] x_flat_i,
  // 16 residual lanes × 8 bits = 128 bits
  input  logic        [127:0] r_flat_i,
  // 16 channels × 72 lanes × 8 bits = 9216 bits
  input  logic       [9215:0] w_flat_i,
  // 16 × fp16 = 256 bits
  input  logic        [255:0] scale_flat_i,
  input  logic        [255:0] bias_flat_i,

  input  logic        [15:0]  add_scale_a_i,
  input  logic        [15:0]  add_scale_b_i,
  input  logic        [15:0]  add_inv_out_scale_i,
  input  logic        [15:0]  add_bias_i,

  output logic                valid_o,
  // 16 × i8 = 128 bits
  output logic        [127:0] y_flat_o
);

  logic signed [71:0][7:0]                  x_w;
  logic signed [15:0][7:0]                  r_w;
  logic signed [15:0][71:0][7:0]            w_w;
  logic        [15:0][15:0]                 scale_w;
  logic        [15:0][15:0]                 bias_w;
  logic signed [15:0][7:0]                  y_w;

  assign x_w     = x_flat_i;
  assign r_w     = r_flat_i;
  assign w_w     = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign y_flat_o = y_w;

  layer_4_m0_cv2 u_dut (
    .clk_i              (clk_i),
    .rst_ni             (rst_ni),
    .valid_i            (valid_i),
    .x_i                (x_w),
    .r_i                (r_w),
    .w_i                (w_w),
    .scale_i            (scale_w),
    .bias_i             (bias_w),
    .add_scale_a_i      (add_scale_a_i),
    .add_scale_b_i      (add_scale_b_i),
    .add_inv_out_scale_i(add_inv_out_scale_i),
    .add_bias_i         (add_bias_i),
    .valid_o            (valid_o),
    .y_o                (y_w)
  );

endmodule
