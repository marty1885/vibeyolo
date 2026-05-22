// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_19_tb — Verilator wrapper around layer_19.
// Geometry: COUT=64, CIN=64, K=1, P_COUT=64, P_CIN=2.
//   x_i  : K*K*P_CIN = 2 lanes  (16 bits)
//   w_i  : P_COUT * 2 = 64*2*8 = 1024 bits
//   scale_i / bias_i : 64*16 = 1024 bits
//   y_o  : 64 i8 (512 bits)

module layer_19_tb (
  input  logic                  clk_i,
  input  logic                  rst_ni,

  input  logic                  valid_i,
  input  logic                  first_cin_i,
  input  logic                  last_cin_i,
  input  logic       [0:0]      cout_tile_idx_i,

  input  logic       [15:0]     x_flat_i,
  input  logic       [1023:0]   w_flat_i,
  input  logic       [1023:0]   scale_flat_i,
  input  logic       [1023:0]   bias_flat_i,

  output logic                  valid_o,
  output logic       [0:0]      cout_tile_idx_o,
  output logic       [511:0]    y_flat_o
);

  logic signed [1:0][7:0]                  x_w;
  logic signed [63:0][1:0][7:0]            w_w;
  logic        [63:0][15:0]                scale_w, bias_w;
  logic signed [63:0][7:0]                 y_w;

  assign x_w     = x_flat_i;
  assign w_w     = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign y_flat_o = y_w;

  layer_19 u_dut (
    .clk_i (clk_i), .rst_ni (rst_ni),
    .valid_i (valid_i), .first_cin_i (first_cin_i), .last_cin_i (last_cin_i),
    .cout_tile_idx_i (cout_tile_idx_i),
    .x_i (x_w), .w_i (w_w), .scale_i (scale_w), .bias_i (bias_w),
    .valid_o (valid_o), .cout_tile_idx_o (cout_tile_idx_o), .y_o (y_w)
  );

endmodule
