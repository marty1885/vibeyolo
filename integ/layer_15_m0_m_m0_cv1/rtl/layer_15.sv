// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_15 — YOLO26n /model.6/m.0/m/m.0/cv1 Conv-BN-SiLU (3x3, s=1, 32→32).
// First conv of the first nested bottleneck inside the C3k=True wrapper
// /model.6/m.0. Input plane is L13 cv1 output (32ch). Non-residual.

// verilator lint_off UNUSEDPARAM
module layer_15
  import scale_pkg::*;
#(
  parameter int  COUT       = LAYER_15_COUT,    // 32
  parameter int  CIN        = LAYER_15_CIN,     // 32
  parameter int  K          = LAYER_15_K,       // 3
  parameter int  STRIDE     = LAYER_15_STRIDE,  // 1
  parameter int  PAD        = LAYER_15_PAD,     // 1
  parameter int  P_COUT     = LAYER_15_P_COUT,  // 32
  parameter int  P_CIN      = LAYER_15_P_CIN,   // 1
  // Empirical: pre-SiLU ~[-4.85, 2.02], post-SiLU |max| ~1.78
  parameter real S_OUT_PRE  = 5.821 / 127.0,
  parameter real S_OUT_SILU = 2.136 / 127.0
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

  output logic                                            valid_o,
  output logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_o,
  output logic signed [P_COUT-1:0][7:0]                   y_o
);

  logic signed [P_COUT-1:0][7:0]  r_tie;
  logic        [P_COUT-1:0][15:0] rs_tie, rb_tie;
  assign r_tie  = '0;
  assign rs_tie = '0;
  assign rb_tie = '0;

  // verilator lint_off PINCONNECTEMPTY
  conv_layer #(
    .CIN       (CIN), .COUT (COUT), .K (K), .STRIDE (STRIDE), .PAD (PAD),
    .H_OUT     (LAYER_15_H), .W_OUT (LAYER_15_W),
    .P_COUT    (P_COUT), .P_CIN (P_CIN),
    .RESIDUAL  (0), .SILU (1),
    .S_OUT_PRE (S_OUT_PRE), .S_OUT_SILU(S_OUT_SILU)
  ) u_conv (
    .clk_i (clk_i), .rst_ni (rst_ni),
    .valid_i (valid_i), .ready_o (),
    .first_cin_i (first_cin_i), .last_cin_i (last_cin_i),
    .cout_tile_idx_i (cout_tile_idx_i),
    .x_i (x_i), .w_i (w_i), .scale_i (scale_i), .bias_i (bias_i),
    .r_i (r_tie), .r_scale_i (rs_tie), .r_bias_i (rb_tie),
    .valid_o (valid_o), .ready_i (1'b1),
    .cout_tile_idx_o (cout_tile_idx_o), .y_o (y_o)
  );
  // verilator lint_on PINCONNECTEMPTY

endmodule
// verilator lint_on UNUSEDPARAM
