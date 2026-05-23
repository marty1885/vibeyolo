// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// detect_head_tb — wraps detect_head for Verilator. Presents the per-scale
// fp16 scales as flat scalar ports (the DUT takes unpacked [3] arrays which
// are awkward to drive from C++) and passes the packed buses straight
// through.

module detect_head_tb #(
  parameter int N_ANCHOR = 8400,
  parameter int N_CLS    = 80,
  parameter int K        = 300,
  parameter int IDX_W    = 14
) (
  input  logic                          clk_i,
  input  logic                          rst_ni,
  input  logic                          start_i,

  input  logic [15:0]                   s_box0_i, s_box1_i, s_box2_i,
  input  logic [15:0]                   s_cls0_i, s_cls1_i, s_cls2_i,

  output logic                          in_ready_o,
  input  logic                          in_valid_i,
  input  logic signed [3:0][7:0]        box_i,
  input  logic signed [N_CLS-1:0][7:0]  cls_i,

  output logic                          out_valid_o,
  output logic [IDX_W-1:0]              out_anchor_o,
  output logic [3:0][15:0]              out_box_o,
  output logic [N_CLS-1:0][15:0]        out_logits_o,
  output logic                          done_o
);

  logic [15:0] s_box [3];
  logic [15:0] s_cls [3];
  always_comb begin
    s_box[0] = s_box0_i; s_box[1] = s_box1_i; s_box[2] = s_box2_i;
    s_cls[0] = s_cls0_i; s_cls[1] = s_cls1_i; s_cls[2] = s_cls2_i;
  end

  detect_head #(.N_ANCHOR(N_ANCHOR), .N_CLS(N_CLS), .K(K), .IDX_W(IDX_W)) u_dut (
    .clk_i, .rst_ni, .start_i,
    .s_box_i(s_box), .s_cls_i(s_cls),
    .in_ready_o, .in_valid_i, .box_i, .cls_i,
    .out_valid_o, .out_anchor_o, .out_box_o, .out_logits_o, .done_o
  );

endmodule
