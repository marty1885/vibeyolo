// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_3_tb — Verilator wrapper around tiled layer_3 (3x3, 16->8).
// P_COUT=16 (covers all 8 outputs in 1 cout tile), P_CIN=8, K=3.

module layer_3_tb (
  input  logic                clk_i,
  input  logic                rst_ni,

  input  logic                valid_i,
  input  logic                first_cin_i,
  input  logic                last_cin_i,
  input  logic        [0:0]   cout_tile_idx_i,    // N_COUT_TILE=1, width forced to 1

  input  logic        [575:0] x_flat_i,           // 9*8 * 8 bits = 576
  input  logic       [9215:0] w_flat_i,           // 16*72*8 = 9216
  input  logic        [255:0] scale_flat_i,
  input  logic        [255:0] bias_flat_i,

  output logic                valid_o,
  output logic        [0:0]   cout_tile_idx_o,
  output logic        [127:0] y_flat_o
);

  logic signed [71:0][7:0]                 x_w;
  logic signed [15:0][71:0][7:0]           w_w;
  logic        [15:0][15:0]                scale_w;
  logic        [15:0][15:0]                bias_w;
  logic signed [15:0][7:0]                 y_w;

  assign x_w     = x_flat_i;
  assign w_w     = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign y_flat_o = y_w;

  layer_3 u_dut (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .valid_i(valid_i), .first_cin_i(first_cin_i), .last_cin_i(last_cin_i),
    .cout_tile_idx_i(cout_tile_idx_i),
    .x_i(x_w), .w_i(w_w),
    .scale_i(scale_w), .bias_i(bias_w),
    .valid_o(valid_o), .cout_tile_idx_o(cout_tile_idx_o), .y_o(y_w)
  );

endmodule
