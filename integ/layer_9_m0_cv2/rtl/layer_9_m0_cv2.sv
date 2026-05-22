// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_9_m0_cv2 — YOLO26n /model.4/m.0/cv2 Conv-BN-SiLU + residual add
// (3x3, 16->32, stride 1, pad 1). The 32-channel residual is the second half
// of /model.4/cv1 post-SiLU output (slice_1), added AFTER cv2 SiLU.
//
// Thin shim over the reusable conv_layer IP (RESIDUAL=1, SILU=1).

// verilator lint_off UNUSEDPARAM
module layer_9_m0_cv2
  import scale_pkg::*;
#(
  parameter int  COUT       = LAYER_9_COUT,    // 32
  parameter int  CIN        = LAYER_9_CIN,     // 16
  parameter int  K          = LAYER_9_K,       // 3
  parameter int  STRIDE     = LAYER_9_STRIDE,  // 1
  parameter int  PAD        = LAYER_9_PAD,     // 1
  parameter int  P_COUT     = LAYER_9_P_COUT,  // 32
  parameter int  P_CIN      = LAYER_9_P_CIN,   // 2
  parameter real S_OUT_PRE  = 6.5 / 127.0,
  parameter real S_OUT_SILU = 6.5 / 127.0
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

  // Residual side-band (per cout_tile beat; held steady across cin-tile sweep,
  // sampled on last_cin_i).
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
    .CIN       (CIN),
    .COUT      (COUT),
    .K         (K),
    .STRIDE    (STRIDE),
    .PAD       (PAD),
    .H_OUT     (LAYER_9_H / STRIDE),
    .W_OUT     (LAYER_9_W / STRIDE),
    .P_COUT    (P_COUT),
    .P_CIN     (P_CIN),
    .RESIDUAL  (1),
    .SILU      (1),
    .S_OUT_PRE (S_OUT_PRE),
    .S_OUT_SILU(S_OUT_SILU)
  ) u_conv (
    .clk_i           (clk_i),
    .rst_ni          (rst_ni),
    .valid_i         (valid_i),
    .ready_o         (),
    .first_cin_i     (first_cin_i),
    .last_cin_i      (last_cin_i),
    .cout_tile_idx_i (cout_tile_idx_i),
    .x_i             (x_i),
    .w_i             (w_i),
    .scale_i         (scale_i),
    .bias_i          (bias_i),
    .r_i             (r_i),
    .r_scale_i       (r_scale_i),
    .r_bias_i        (r_bias_i),
    .valid_o         (valid_o),
    .ready_i         (1'b1),
    .cout_tile_idx_o (cout_tile_idx_o),
    .y_o             (y_o)
  );
  // verilator lint_on PINCONNECTEMPTY

endmodule
// verilator lint_on UNUSEDPARAM
