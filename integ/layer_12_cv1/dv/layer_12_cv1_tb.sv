// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_12_cv1_tb — Verilator wrapper around layer_12_cv1. Flattens ports
// so the C++ test can drive 4-lane phases (4 bytes = 32 bits) and read
// 128-channel int8 outputs (1024 bits) each cycle.

module layer_12_cv1_tb (
  input  logic                clk_i,
  input  logic                rst_ni,

  input  logic                valid_i,
  input  logic          [4:0] phase_i,         // log2(32) = 5
  // 4 lanes × 8 bits = 32 bits
  input  logic         [31:0] x_flat_i,
  // 128 channels × 4 lanes × 8 bits = 4096 bits
  input  logic       [4095:0] w_flat_i,
  // 128 × fp16 = 2048 bits
  input  logic       [2047:0] scale_flat_i,
  input  logic       [2047:0] bias_flat_i,

  output logic                valid_o,
  // 128 × i8 = 1024 bits
  output logic       [1023:0] y_flat_o
);

  logic signed [3:0][7:0]                    x_w;
  logic signed [127:0][3:0][7:0]             w_w;
  logic        [127:0][15:0]                 scale_w;
  logic        [127:0][15:0]                 bias_w;
  logic signed [127:0][7:0]                  y_w;

  assign x_w     = x_flat_i;
  assign w_w     = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign y_flat_o = y_w;

  layer_12_cv1 u_dut (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .valid_i(valid_i),
    .phase_i(phase_i),
    .x_i    (x_w),
    .w_i    (w_w),
    .scale_i(scale_w),
    .bias_i (bias_w),
    .valid_o(valid_o),
    .y_o    (y_w)
  );

endmodule
