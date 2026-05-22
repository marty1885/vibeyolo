// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_4 — YOLO26n /model.2/m.0/cv2 Conv-BN-SiLU + residual add
// (3x3, 8->16, stride 1, pad 1, RESIDUAL=1).
// Thin shim over conv_layer with RESIDUAL=1; residual is /model.2/Slice_1.

// verilator lint_off UNUSEDPARAM
module layer_4
  import scale_pkg::*;
#(
  parameter int  COUT       = LAYER_4_COUT,
  parameter int  CIN        = LAYER_4_CIN,
  parameter int  K          = LAYER_4_K,
  parameter int  STRIDE     = LAYER_4_STRIDE,
  parameter int  PAD        = LAYER_4_PAD,
  parameter int  P_COUT     = 16,
  parameter int  P_CIN      = 8,
  parameter real S_OUT_PRE  = 32.0 / 127.0,
  parameter real S_OUT_SILU = 32.0 / 127.0
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

  // Residual side-band
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
    .CIN(CIN), .COUT(COUT), .K(K), .STRIDE(STRIDE), .PAD(PAD),
    .H_OUT(LAYER_4_H), .W_OUT(LAYER_4_W),
    .P_COUT(P_COUT), .P_CIN(P_CIN),
    .RESIDUAL(1), .SILU(1),
    .S_OUT_PRE(S_OUT_PRE), .S_OUT_SILU(S_OUT_SILU)
  ) u_conv (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .valid_i(valid_i), .ready_o(),
    .first_cin_i(first_cin_i), .last_cin_i(last_cin_i),
    .cout_tile_idx_i(cout_tile_idx_i),
    .x_i(x_i), .w_i(w_i), .scale_i(scale_i), .bias_i(bias_i),
    .r_i(r_i), .r_scale_i(r_scale_i), .r_bias_i(r_bias_i),
    .valid_o(valid_o), .ready_i(1'b1),
    .cout_tile_idx_o(cout_tile_idx_o), .y_o(y_o)
  );
  // verilator lint_on PINCONNECTEMPTY
endmodule
// verilator lint_on UNUSEDPARAM
