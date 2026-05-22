// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// Generated skeleton for L89: /model.22/m.0/m.0.1/attn/pe/conv/Conv_quant  (SILU=0)

module layer_89
  import scale_pkg::*;
#(
  parameter int  COUT       = LAYER_89_COUT,
  parameter int  CIN        = LAYER_89_CIN,
  parameter int  K          = LAYER_89_K,
  parameter int  STRIDE     = LAYER_89_STRIDE,
  parameter int  PAD        = LAYER_89_PAD,
  parameter int  P_COUT     = LAYER_89_P_COUT,
  parameter int  P_CIN      = LAYER_89_P_CIN,
  parameter int  P_PIX      = LAYER_89_P_PIX,
  parameter real S_OUT_PRE  = 8.0 / 127.0,
  parameter real S_OUT_SILU = 8.0 / 127.0
) (
  input  logic                                            clk_i,
  input  logic                                            rst_ni,
  input  logic                                            valid_i,
  input  logic                                            first_cin_i,
  input  logic                                            last_cin_i,
  input  logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_i,
  input  logic signed [P_PIX-1:0][K*K*P_CIN-1:0][7:0]     x_i,
  input  logic signed [P_COUT-1:0][K*K*P_CIN-1:0][7:0]    w_i,
  input  logic        [P_COUT-1:0][15:0]                  scale_i,
  input  logic        [P_COUT-1:0][15:0]                  bias_i,
  output logic                                            valid_o,
  output logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_o,
  output logic signed [P_PIX-1:0][P_COUT-1:0][7:0]        y_o
);

  logic signed [P_COUT-1:0][7:0]  r_tie;
  logic        [P_COUT-1:0][15:0] rs_tie, rb_tie;
  logic        [P_PIX-1:0]        ready_unused;
  logic        [P_PIX-1:0]        valid_lane;
  logic        [P_PIX-1:0][$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                      cout_tile_idx_lane;
  logic        [P_PIX-1:0]        tag_match_lane;
  assign r_tie  = '0;
  assign rs_tie = '0;
  assign rb_tie = '0;
  assign valid_o = (&valid_lane) & (&tag_match_lane);
  assign cout_tile_idx_o = cout_tile_idx_lane[0];

  for (genvar pix = 0; pix < P_PIX; pix++) begin : gen_pix
    assign tag_match_lane[pix] = (cout_tile_idx_lane[pix] == cout_tile_idx_lane[0]);

    conv_layer #(
      .CIN(CIN), .COUT(COUT), .K(K), .STRIDE(STRIDE), .PAD(PAD),
      .H_OUT(LAYER_89_H), .W_OUT(LAYER_89_W),
      .P_COUT(P_COUT), .P_CIN(P_CIN),
      .RESIDUAL(0), .SILU(0),
      .S_OUT_PRE(S_OUT_PRE), .S_OUT_SILU(S_OUT_SILU)
    ) u_conv (
      .clk_i(clk_i), .rst_ni(rst_ni), .valid_i(valid_i), .ready_o(ready_unused[pix]),
      .first_cin_i(first_cin_i), .last_cin_i(last_cin_i),
      .cout_tile_idx_i(cout_tile_idx_i),
      .x_i(x_i[pix]), .w_i(w_i), .scale_i(scale_i), .bias_i(bias_i),
      .r_i(r_tie), .r_scale_i(rs_tie), .r_bias_i(rb_tie),
      .valid_o(valid_lane[pix]), .ready_i(1'b1),
      .cout_tile_idx_o(cout_tile_idx_lane[pix]), .y_o(y_o[pix])
    );
  end

endmodule
