// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_9_m0_cv2_tb — Verilator wrapper around layer_9_m0_cv2 (RESIDUAL=1).
//
// Geometry (scale_pkg_dv): P_COUT=32, P_CIN=2, K=3, CIN=16, COUT=32.
//   x_i  : K*K*P_CIN = 18   i8 lanes (144 bits)
//   w_i  : P_COUT * 18 i8           (32 * 18 * 8 = 4608 bits)
//   scale_i / bias_i : P_COUT fp16  (512 bits each)
//   y_o  : P_COUT i8                (256 bits)
//   r_i  : P_COUT i8                (256 bits)
//   r_scale_i / r_bias_i : P_COUT fp16 (512 bits each)
//   N_COUT_TILE = 1

module layer_9_m0_cv2_tb (
  input  logic                  clk_i,
  input  logic                  rst_ni,

  input  logic                  valid_i,
  input  logic                  first_cin_i,
  input  logic                  last_cin_i,
  input  logic       [0:0]      cout_tile_idx_i,

  input  logic       [143:0]    x_flat_i,
  input  logic       [4607:0]   w_flat_i,
  input  logic       [511:0]    scale_flat_i,
  input  logic       [511:0]    bias_flat_i,

  input  logic       [255:0]    r_flat_i,
  input  logic       [511:0]    rs_flat_i,
  input  logic       [511:0]    rb_flat_i,

  output logic                  valid_o,
  output logic       [0:0]      cout_tile_idx_o,
  output logic       [255:0]    y_flat_o
);

  logic signed [17:0][7:0]                 x_w;
  logic signed [31:0][17:0][7:0]           w_w;
  logic        [31:0][15:0]                scale_w;
  logic        [31:0][15:0]                bias_w;
  logic signed [31:0][7:0]                 r_w;
  logic        [31:0][15:0]                rs_w;
  logic        [31:0][15:0]                rb_w;
  logic signed [31:0][7:0]                 y_w;

  assign x_w     = x_flat_i;
  assign w_w     = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign r_w     = r_flat_i;
  assign rs_w    = rs_flat_i;
  assign rb_w    = rb_flat_i;
  assign y_flat_o = y_w;

  layer_9_m0_cv2 u_dut (
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
    .r_i             (r_w),
    .r_scale_i       (rs_w),
    .r_bias_i        (rb_w),
    .valid_o         (valid_o),
    .cout_tile_idx_o (cout_tile_idx_o),
    .y_o             (y_w)
  );

endmodule
