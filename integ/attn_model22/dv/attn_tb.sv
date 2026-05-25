// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// attn_tb — Verilator wrapper exposing the attn block streaming interface
// to the C++ driver. The block computes ATTN_OUT = attention(qkv) + pe,
// int8 @ S_AOUT. Per-tensor scales are pinned at elaboration by
// attn_scales_pkg (imported inside attn.sv); no scale parameters on the
// instance.

module attn_tb #(
  parameter int H        = 20,
  parameter int W        = 20,
  parameter int C_QKV    = 256,
  parameter int C_FE     = 128,
  parameter int HEADS    = 2,
  parameter int DIM_Q    = 32,
  parameter int DIM_K    = 32,
  parameter int DIM_V    = 64,
  parameter int unsigned ACC_EXP  = 8,
  parameter int unsigned ACC_MANT = 21
) (
  input  logic                              clk_i,
  input  logic                              rst_ni,
  input  logic                              start_i,

  input  logic                              qkv_valid_i,
  output logic                              qkv_ready_o,
  input  logic signed [C_QKV-1:0][7:0]      qkv_data_i,

  input  logic                              pe_valid_i,
  output logic                              pe_ready_o,
  input  logic signed [C_FE-1:0][7:0]       pe_data_i,

  output logic                              ovalid_o,
  input  logic                              oready_i,
  output logic signed [C_FE-1:0][7:0]       odata_o,

  output logic                              done_o
);

  attn #(
    .H(H), .W(W), .C_QKV(C_QKV), .C_FE(C_FE),
    .HEADS(HEADS), .DIM_Q(DIM_Q), .DIM_K(DIM_K), .DIM_V(DIM_V),
    .ACC_EXP(ACC_EXP), .ACC_MANT(ACC_MANT)
  ) u_dut (
    .clk_i        (clk_i),
    .rst_ni       (rst_ni),
    .start_i      (start_i),
    .qkv_valid_i  (qkv_valid_i),
    .qkv_ready_o  (qkv_ready_o),
    .qkv_data_i   (qkv_data_i),
    .pe_valid_i   (pe_valid_i),
    .pe_ready_o   (pe_ready_o),
    .pe_data_i    (pe_data_i),
    .ovalid_o     (ovalid_o),
    .oready_i     (oready_i),
    .odata_o      (odata_o),
    .done_o       (done_o)
  );

endmodule
