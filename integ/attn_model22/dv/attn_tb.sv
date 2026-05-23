// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// attn_tb — Verilator wrapper exposing the attn block streaming
// interface to the C++ driver. Per-tensor scales are passed in via
// real parameters (read from stim/manifest.json by the C++ driver and
// applied at elaboration through the parameter overrides below — but
// since Verilator doesn't accept runtime real overrides, we instead
// hardcode the scales here at elaboration time by reading a generated
// "stim/s_out_params.sv" -style package... however for this block the
// simpler path is to compile-time-define a few real plugs and
// rebuild per-run is overkill. We instead embed a default set covering
// the rand* samples; the C++ driver loads matching stim/golden which
// were generated with the same defaults).
//
// To avoid the per-sample-scale problem entirely, the extractor pins
// ALL samples to the SAME per-tensor scales (the max over samples).
// See extract.py — this matches how upsample_concat handles its single
// shared S_OUT semantic and is fine for the cosine target.

module attn_tb #(
  parameter int H        = 20,
  parameter int W        = 20,
  parameter int C_QKV    = 256,
  parameter int C_FE     = 128,
  parameter int HEADS    = 2,
  parameter int DIM_Q    = 32,
  parameter int DIM_K    = 32,
  parameter int DIM_V    = 64
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

  input  logic                              proj_valid_i,
  output logic                              proj_ready_o,
  input  logic signed [C_FE-1:0][7:0]       proj_data_i,

  input  logic                              spl1_valid_i,
  output logic                              spl1_ready_o,
  input  logic signed [C_FE-1:0][7:0]       spl1_data_i,

  input  logic                              ffn1_valid_i,
  output logic                              ffn1_ready_o,
  input  logic signed [C_FE-1:0][7:0]       ffn1_data_i,

  output logic                              ovalid_o,
  input  logic                              oready_i,
  output logic signed [C_FE-1:0][7:0]       odata_o,

  output logic                              done_o
);

  // Structural shim pulls fp16 bits directly from attn_scales_pkg via
  // `import` inside attn.sv; no scale parameters needed on the instance.
  attn #(
    .H(H), .W(W), .C_QKV(C_QKV), .C_FE(C_FE),
    .HEADS(HEADS), .DIM_Q(DIM_Q), .DIM_K(DIM_K), .DIM_V(DIM_V)
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
    .proj_valid_i (proj_valid_i),
    .proj_ready_o (proj_ready_o),
    .proj_data_i  (proj_data_i),
    .spl1_valid_i (spl1_valid_i),
    .spl1_ready_o (spl1_ready_o),
    .spl1_data_i  (spl1_data_i),
    .ffn1_valid_i (ffn1_valid_i),
    .ffn1_ready_o (ffn1_ready_o),
    .ffn1_data_i  (ffn1_data_i),
    .ovalid_o     (ovalid_o),
    .oready_i     (oready_i),
    .odata_o      (odata_o),
    .done_o       (done_o)
  );

endmodule
