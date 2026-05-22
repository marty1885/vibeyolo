// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// stem_l0_tb — Verilator wrapper around stem_l0 (tiled architecture).
// P_COUT=16, P_CIN=3, K=3.
//   x_i  : 27 i8 lanes (216 bits)
//   w_i  : 16*27 i8 (3456 bits)
//   scale_i/bias_i : 16 fp16 (256 bits)
//   y_o  : 16 i8 (128 bits)

module stem_l0_tb (
  input  logic                clk_i,
  input  logic                rst_ni,

  input  logic                valid_i,
  input  logic                first_cin_i,
  input  logic                last_cin_i,
  input  logic        [0:0]   cout_tile_idx_i,

  input  logic        [215:0] x_flat_i,
  input  logic       [3455:0] w_flat_i,
  input  logic        [255:0] scale_flat_i,
  input  logic        [255:0] bias_flat_i,

  output logic                valid_o,
  output logic        [0:0]   cout_tile_idx_o,
  output logic        [127:0] y_flat_o
);

  logic signed [26:0][7:0]                  x_w;
  logic signed [15:0][26:0][7:0]            w_w;
  logic        [15:0][15:0]                 scale_w;
  logic        [15:0][15:0]                 bias_w;
  logic signed [15:0][7:0]                  y_w;

  assign x_w     = x_flat_i;
  assign w_w     = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign y_flat_o = y_w;

  stem_l0 u_dut (
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
