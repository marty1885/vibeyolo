// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_4_tb — Verilator wrapper around tiled layer_4 (3x3 s1, 8->16, +residual).
// P_COUT=16 (covers all 16 outputs in 1 tile), P_CIN=8 (covers all 8 inputs in 1 tile).
// Therefore N_COUT_TILE = N_CIN_TILE = 1, beats/pix = 1.

module layer_4_tb (
  input  logic                clk_i,
  input  logic                rst_ni,
  input  logic                valid_i,
  input  logic                first_cin_i,
  input  logic                last_cin_i,
  input  logic        [0:0]   cout_tile_idx_i,

  input  logic        [575:0] x_flat_i,           // 9*8 * 8 = 576
  input  logic       [9215:0] w_flat_i,           // 16*72*8 = 9216
  input  logic        [255:0] scale_flat_i,
  input  logic        [255:0] bias_flat_i,

  input  logic        [127:0] r_flat_i,           // 16*8
  input  logic        [255:0] r_scale_flat_i,     // 16*16
  input  logic        [255:0] r_bias_flat_i,      // 16*16

  output logic                valid_o,
  output logic        [0:0]   cout_tile_idx_o,
  output logic        [127:0] y_flat_o
);

  logic signed [71:0][7:0]                 x_w;
  logic signed [15:0][71:0][7:0]           w_w;
  logic        [15:0][15:0]                scale_w;
  logic        [15:0][15:0]                bias_w;
  logic signed [15:0][7:0]                 r_w;
  logic        [15:0][15:0]                r_scale_w;
  logic        [15:0][15:0]                r_bias_w;
  logic signed [15:0][7:0]                 y_w;

  assign x_w       = x_flat_i;
  assign w_w       = w_flat_i;
  assign scale_w   = scale_flat_i;
  assign bias_w    = bias_flat_i;
  assign r_w       = r_flat_i;
  assign r_scale_w = r_scale_flat_i;
  assign r_bias_w  = r_bias_flat_i;
  assign y_flat_o  = y_w;

  layer_4 u_dut (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .valid_i(valid_i), .first_cin_i(first_cin_i), .last_cin_i(last_cin_i),
    .cout_tile_idx_i(cout_tile_idx_i),
    .x_i(x_w), .w_i(w_w), .scale_i(scale_w), .bias_i(bias_w),
    .r_i(r_w), .r_scale_i(r_scale_w), .r_bias_i(r_bias_w),
    .valid_o(valid_o), .cout_tile_idx_o(cout_tile_idx_o), .y_o(y_w)
  );

endmodule
