// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// topk_fp16_tb — Verilator TB wrapper. Instantiates DUT and REF with the
// same parameters and exposes flattened heap output ports to the C++ test.
//
// Two configurations are built (multi-test mode in mk/verilator.mk):
//   • topk_fp16_tb_big   : N=8400, K=300, IDX_W=14
//   • topk_fp16_tb_small : N=32,   K=4,   IDX_W=5

module topk_fp16_tb #(
  parameter int N     = 8400,
  parameter int K     = 300,
  parameter int IDX_W = $clog2(N)
) (
  input  logic                clk_i,
  input  logic                rst_ni,

  input  logic                start_i,
  input  logic                in_valid_i,
  input  logic [15:0]         in_value_i,
  input  logic [IDX_W-1:0]    in_index_i,

  output logic                in_ready_dut_o,
  output logic                in_ready_ref_o,
  output logic                done_dut_o,
  output logic                done_ref_o,

  // Flattened outputs. Lane i lives at bits [16*i +: 16] / [IDX_W*i +: IDX_W]
  output logic [K*16 - 1:0]     out_value_dut_o,
  output logic [K*IDX_W - 1:0]  out_index_dut_o,
  output logic [K*16 - 1:0]     out_value_ref_o,
  output logic [K*IDX_W - 1:0]  out_index_ref_o
);

  logic [15:0]      v_dut [K], v_ref [K];
  logic [IDX_W-1:0] x_dut [K], x_ref [K];

  topk_fp16 #(.N(N), .K(K), .IDX_W(IDX_W)) u_dut (
    .clk_i      (clk_i),
    .rst_ni     (rst_ni),
    .start_i    (start_i),
    .in_valid_i (in_valid_i),
    .in_value_i (in_value_i),
    .in_index_i (in_index_i),
    .in_ready_o (in_ready_dut_o),
    .done_o     (done_dut_o),
    .out_value_o(v_dut),
    .out_index_o(x_dut)
  );

  topk_fp16_ref #(.N(N), .K(K), .IDX_W(IDX_W)) u_ref (
    .clk_i      (clk_i),
    .rst_ni     (rst_ni),
    .start_i    (start_i),
    .in_valid_i (in_valid_i),
    .in_value_i (in_value_i),
    .in_index_i (in_index_i),
    .in_ready_o (in_ready_ref_o),
    .done_o     (done_ref_o),
    .out_value_o(v_ref),
    .out_index_o(x_ref)
  );

  always_comb begin
    for (int i = 0; i < K; i++) begin
      out_value_dut_o[16*i +: 16]         = v_dut[i];
      out_index_dut_o[IDX_W*i +: IDX_W]   = x_dut[i];
      out_value_ref_o[16*i +: 16]         = v_ref[i];
      out_index_ref_o[IDX_W*i +: IDX_W]   = x_ref[i];
    end
  end

endmodule

// Configured wrappers (so Verilator can pick top-module by name).
module topk_fp16_tb_big (
  input  logic clk_i, rst_ni, start_i, in_valid_i,
  input  logic [15:0] in_value_i,
  input  logic [13:0] in_index_i,
  output logic in_ready_dut_o, in_ready_ref_o, done_dut_o, done_ref_o,
  output logic [300*16-1:0] out_value_dut_o, out_value_ref_o,
  output logic [300*14-1:0] out_index_dut_o, out_index_ref_o
);
  topk_fp16_tb #(.N(8400), .K(300), .IDX_W(14)) u (.*);
endmodule

module topk_fp16_tb_small (
  input  logic clk_i, rst_ni, start_i, in_valid_i,
  input  logic [15:0] in_value_i,
  input  logic [4:0]  in_index_i,
  output logic in_ready_dut_o, in_ready_ref_o, done_dut_o, done_ref_o,
  output logic [4*16-1:0] out_value_dut_o, out_value_ref_o,
  output logic [4*5-1:0]  out_index_dut_o, out_index_ref_o
);
  topk_fp16_tb #(.N(32), .K(4), .IDX_W(5)) u (.*);
endmodule
