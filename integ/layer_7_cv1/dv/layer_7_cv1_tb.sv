// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_7_cv1_tb — Verilator wrapper around layer_7_cv1 (tiled conv_layer).
//
// Geometry (scale_pkg_dv): P_COUT=64, P_CIN=8, K=1, CIN=COUT=64.
//   x_i  : K*K*P_CIN = 8   i8 lanes  (64 bits)
//   w_i  : P_COUT * 8 i8           (64 * 8 * 8 = 4096 bits)
//   scale_i / bias_i : P_COUT fp16  (1024 bits each)
//   y_o  : P_COUT i8                (512 bits)
//   N_COUT_TILE = 1

module layer_7_cv1_tb (
  input  logic                  clk_i,
  input  logic                  rst_ni,

  input  logic                  valid_i,
  input  logic                  first_cin_i,
  input  logic                  last_cin_i,
  input  logic       [0:0]      cout_tile_idx_i,

  input  logic       [63:0]     x_flat_i,        // 8 * 8
  input  logic       [4095:0]   w_flat_i,        // 64 * 8 * 8
  input  logic       [1023:0]   scale_flat_i,    // 64 * 16
  input  logic       [1023:0]   bias_flat_i,

  output logic                  valid_o,
  output logic       [0:0]      cout_tile_idx_o,
  output logic       [511:0]    y_flat_o         // 64 * 8
);

  logic signed [7:0][7:0]                 x_w;
  logic signed [63:0][7:0][7:0]           w_w;
  logic        [63:0][15:0]               scale_w;
  logic        [63:0][15:0]               bias_w;
  logic signed [63:0][7:0]                y_w;

  assign x_w     = x_flat_i;
  assign w_w     = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign y_flat_o = y_w;

  layer_7_cv1 u_dut (
    .clk_i           (clk_i),
    .rst_ni          (rst_ni),
    .valid_i         (valid_i),
    .first_cin_i     (first_cin_i),
    .last_cin_i      (last_cin_i),
    .cout_tile_idx_i (cout_tile_idx_i),
    .x_i             (x_w),
    .w_i             (w_w),
    .scale_i         (scale_w),
    .bias_i          (bias_w),
    .valid_o         (valid_o),
    .cout_tile_idx_o (cout_tile_idx_o),
    .y_o             (y_w)
  );

endmodule
