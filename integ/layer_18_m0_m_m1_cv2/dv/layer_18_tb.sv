// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_18_tb — Verilator wrapper around layer_18 (with residual).

module layer_18_tb (
  input  logic                  clk_i,
  input  logic                  rst_ni,

  input  logic                  valid_i,
  input  logic                  first_cin_i,
  input  logic                  last_cin_i,
  input  logic       [0:0]      cout_tile_idx_i,

  input  logic       [71:0]     x_flat_i,      // 9 lanes
  input  logic       [2303:0]   w_flat_i,      // 32 * 9 i8
  input  logic       [511:0]    scale_flat_i,  // 32 fp16
  input  logic       [511:0]    bias_flat_i,
  input  logic       [255:0]    r_flat_i,      // 32 i8
  input  logic       [511:0]    r_scale_flat_i,
  input  logic       [511:0]    r_bias_flat_i,

  output logic                  valid_o,
  output logic       [0:0]      cout_tile_idx_o,
  output logic       [255:0]    y_flat_o
);

  logic signed [8:0][7:0]                   x_w;
  logic signed [31:0][8:0][7:0]             w_w;
  logic        [31:0][15:0]                 scale_w, bias_w, rs_w, rb_w;
  logic signed [31:0][7:0]                  y_w, r_w;

  assign x_w     = x_flat_i;
  assign w_w     = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign r_w     = r_flat_i;
  assign rs_w    = r_scale_flat_i;
  assign rb_w    = r_bias_flat_i;
  assign y_flat_o = y_w;

  layer_18 u_dut (
    .clk_i (clk_i), .rst_ni (rst_ni),
    .valid_i (valid_i), .first_cin_i (first_cin_i), .last_cin_i (last_cin_i),
    .cout_tile_idx_i (cout_tile_idx_i),
    .x_i (x_w), .w_i (w_w), .scale_i (scale_w), .bias_i (bias_w),
    .r_i (r_w), .r_scale_i (rs_w), .r_bias_i (rb_w),
    .valid_o (valid_o), .cout_tile_idx_o (cout_tile_idx_o), .y_o (y_w)
  );

endmodule
