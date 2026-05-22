// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_15_tb — Verilator wrapper around layer_15.
// Geometry: COUT=32, CIN=32, K=3, P_COUT=32, P_CIN=1.
//   x_i  : K*K*P_CIN = 9 i8 (72 bits)
//   w_i  : P_COUT * 9 i8    (32*9*8 = 2304 bits)
//   scale_i / bias_i: 32*16 = 512 bits
//   y_o  : 32 i8 (256 bits)

module layer_15_tb (
  input  logic                  clk_i,
  input  logic                  rst_ni,

  input  logic                  valid_i,
  input  logic                  first_cin_i,
  input  logic                  last_cin_i,
  input  logic       [0:0]      cout_tile_idx_i,   // N_COUT_TILE=1

  input  logic       [71:0]     x_flat_i,
  input  logic       [2303:0]   w_flat_i,
  input  logic       [511:0]    scale_flat_i,
  input  logic       [511:0]    bias_flat_i,

  output logic                  valid_o,
  output logic       [0:0]      cout_tile_idx_o,
  output logic       [255:0]    y_flat_o
);

  logic signed [8:0][7:0]                   x_w;
  logic signed [31:0][8:0][7:0]             w_w;
  logic        [31:0][15:0]                 scale_w;
  logic        [31:0][15:0]                 bias_w;
  logic signed [31:0][7:0]                  y_w;

  assign x_w     = x_flat_i;
  assign w_w     = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign y_flat_o = y_w;

  layer_15 u_dut (
    .clk_i (clk_i), .rst_ni (rst_ni),
    .valid_i (valid_i), .first_cin_i (first_cin_i), .last_cin_i (last_cin_i),
    .cout_tile_idx_i (cout_tile_idx_i),
    .x_i (x_w), .w_i (w_w), .scale_i (scale_w), .bias_i (bias_w),
    .valid_o (valid_o), .cout_tile_idx_o (cout_tile_idx_o), .y_o (y_w)
  );

endmodule
