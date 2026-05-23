// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
// flash_attn TB — small config: HEADS=1, N=8, DIM_Q=4, DIM_V=4, BR=2, BC=4.

module flash_attn_tb_small (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         start_i,
  output logic         done_dut_o,
  output logic         done_ref_o,
  input  logic [1*8*4*16-1:0]  q_flat_i,
  input  logic [1*8*4*16-1:0]  k_flat_i,
  input  logic [1*8*4*16-1:0]  v_flat_i,
  output logic [1*8*4*16-1:0]  o_dut_flat_o,
  output logic [1*8*4*16-1:0]  o_ref_flat_o
);
  localparam int HEADS = 1;
  localparam int N     = 8;
  localparam int DIM_Q = 4;
  localparam int DIM_V = 4;
  localparam int BR    = 2;
  localparam int BC    = 4;

  flash_attn #(
    .HEADS(HEADS), .N(N), .DIM_Q(DIM_Q), .DIM_V(DIM_V),
    .BR(BR), .BC(BC)
  ) u_dut (
    .clk_i, .rst_ni, .start_i,
    .done_o(done_dut_o),
    .q_flat_i, .k_flat_i, .v_flat_i,
    .o_flat_o(o_dut_flat_o)
  );

  flash_attn_ref #(
    .HEADS(HEADS), .N(N), .DIM_Q(DIM_Q), .DIM_V(DIM_V),
    .BR(BR), .BC(BC)
  ) u_ref (
    .clk_i, .rst_ni, .start_i,
    .done_o(done_ref_o),
    .q_flat_i, .k_flat_i, .v_flat_i,
    .o_flat_o(o_ref_flat_o)
  );
endmodule
