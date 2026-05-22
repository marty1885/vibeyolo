// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_18 — YOLO26n /model.6/m.0/m/m.0/cv2 Conv-BN-SiLU + residual Add
// (3x3, s=1, 32→32). Second conv of the first nested Bottleneck inside the
// C3k=True wrapper /model.6/m.0. Bottleneck.add=True: output = silu(cv2(x)) +
// bottleneck_input, where the residual is /model.6/m.0/cv1/act/Mul_output_0.

// verilator lint_off UNUSEDPARAM
module layer_18
  import scale_pkg::*;
#(
  parameter int  COUT       = LAYER_18_COUT,    // 32
  parameter int  CIN        = LAYER_18_CIN,     // 32
  parameter int  K          = LAYER_18_K,       // 3
  parameter int  STRIDE     = LAYER_18_STRIDE,
  parameter int  PAD        = LAYER_18_PAD,
  parameter int  P_COUT     = LAYER_18_P_COUT,
  parameter int  P_CIN      = LAYER_18_P_CIN,
  // Empirical: pre-SiLU ~[-2.6,2.06], |silu|~1.83, |add|~1.95
  parameter real S_OUT_PRE  = 4.899 / 127.0,
  parameter real S_OUT_SILU = 5.795 / 127.0
) (
  input  logic                                            clk_i,
  input  logic                                            rst_ni,

  input  logic                                            valid_i,
  input  logic                                            first_cin_i,
  input  logic                                            last_cin_i,
  input  logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_i,

  input  logic signed [K*K*P_CIN-1:0][7:0]                x_i,
  input  logic signed [P_COUT-1:0][K*K*P_CIN-1:0][7:0]    w_i,
  input  logic        [P_COUT-1:0][15:0]                  scale_i,
  input  logic        [P_COUT-1:0][15:0]                  bias_i,

  // Residual side-band — driver holds steady across the cin-tile sweep
  // and updates per output pixel.
  input  logic signed [P_COUT-1:0][7:0]                   r_i,
  input  logic        [P_COUT-1:0][15:0]                  r_scale_i,
  input  logic        [P_COUT-1:0][15:0]                  r_bias_i,

  output logic                                            valid_o,
  output logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_o,
  output logic signed [P_COUT-1:0][7:0]                   y_o
);

  // verilator lint_off PINCONNECTEMPTY
  conv_layer #(
    .CIN       (CIN), .COUT (COUT), .K (K), .STRIDE (STRIDE), .PAD (PAD),
    .H_OUT     (LAYER_18_H), .W_OUT (LAYER_18_W),
    .P_COUT    (P_COUT), .P_CIN (P_CIN),
    .RESIDUAL  (1), .SILU (1),
    .S_OUT_PRE (S_OUT_PRE), .S_OUT_SILU(S_OUT_SILU)
  ) u_conv (
    .clk_i (clk_i), .rst_ni (rst_ni),
    .valid_i (valid_i), .ready_o (),
    .first_cin_i (first_cin_i), .last_cin_i (last_cin_i),
    .cout_tile_idx_i (cout_tile_idx_i),
    .x_i (x_i), .w_i (w_i), .scale_i (scale_i), .bias_i (bias_i),
    .r_i (r_i), .r_scale_i (r_scale_i), .r_bias_i (r_bias_i),
    .valid_o (valid_o), .ready_i (1'b1),
    .cout_tile_idx_o (cout_tile_idx_o), .y_o (y_o)
  );
  // verilator lint_on PINCONNECTEMPTY

endmodule
// verilator lint_on UNUSEDPARAM
