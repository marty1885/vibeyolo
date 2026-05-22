// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// flash_attn_tb — Verilator TB wrapper. DUT + REF in lockstep,
// instantiated at DV-sized parameters.

module flash_attn_tb #(
  parameter int HEADS = 1,
  parameter int N     = 8,
  parameter int DIM_Q = 4,
  parameter int DIM_V = 4
) (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               q_we_i,
  input  logic        [15:0] q_waddr_i,
  input  logic        [15:0] q_wdata_i,

  input  logic               k_we_i,
  input  logic        [15:0] k_waddr_i,
  input  logic        [15:0] k_wdata_i,

  input  logic               v_we_i,
  input  logic        [15:0] v_waddr_i,
  input  logic        [15:0] v_wdata_i,

  input  logic               start_i,
  output logic               busy_dut_o,
  output logic               busy_ref_o,
  output logic               done_dut_o,
  output logic               done_ref_o,

  input  logic        [15:0] o_raddr_i,
  output logic        [15:0] o_dut_o,
  output logic        [15:0] o_ref_o
);

  flash_attn #(
    .HEADS(HEADS), .N(N), .DIM_Q(DIM_Q), .DIM_V(DIM_V)
  ) u_dut (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .q_we_i (q_we_i), .q_waddr_i(q_waddr_i[$clog2(HEADS*N*DIM_Q):0]), .q_wdata_i(q_wdata_i),
    .k_we_i (k_we_i), .k_waddr_i(k_waddr_i[$clog2(HEADS*N*DIM_Q):0]), .k_wdata_i(k_wdata_i),
    .v_we_i (v_we_i), .v_waddr_i(v_waddr_i[$clog2(HEADS*N*DIM_V):0]), .v_wdata_i(v_wdata_i),
    .start_i(start_i),
    .busy_o (busy_dut_o),
    .done_o (done_dut_o),
    .o_raddr_i(o_raddr_i[$clog2(HEADS*N*DIM_V):0]),
    .o_rdata_o(o_dut_o)
  );

  flash_attn_ref #(
    .HEADS(HEADS), .N(N), .DIM_Q(DIM_Q), .DIM_V(DIM_V)
  ) u_ref (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .q_we_i (q_we_i), .q_waddr_i(q_waddr_i[$clog2(HEADS*N*DIM_Q):0]), .q_wdata_i(q_wdata_i),
    .k_we_i (k_we_i), .k_waddr_i(k_waddr_i[$clog2(HEADS*N*DIM_Q):0]), .k_wdata_i(k_wdata_i),
    .v_we_i (v_we_i), .v_waddr_i(v_waddr_i[$clog2(HEADS*N*DIM_V):0]), .v_wdata_i(v_wdata_i),
    .start_i(start_i),
    .busy_o (busy_ref_o),
    .done_o (done_ref_o),
    .o_raddr_i(o_raddr_i[$clog2(HEADS*N*DIM_V):0]),
    .o_rdata_o(o_ref_o)
  );

endmodule
