// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
// flash_attn TB — prod: HEADS=2, N=400, DIM_Q=32, DIM_V=64, BR=16, BC=32.

module flash_attn_tb_prod (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         start_i,
  output logic         done_dut_o,
  output logic         done_ref_o,
  input  logic [2*400*32*16-1:0] q_flat_i,
  input  logic [2*400*32*16-1:0] k_flat_i,
  input  logic [2*400*64*16-1:0] v_flat_i,
  output logic [2*400*64*16-1:0] o_dut_flat_o,
  output logic [2*400*64*16-1:0] o_ref_flat_o
);
  localparam int HEADS = 2;
  localparam int N     = 400;
  localparam int DIM_Q = 32;
  localparam int DIM_V = 64;
  localparam int BR    = 16;
  localparam int BC    = 32;
  localparam logic [15:0] TEMP = 16'h31A8;

  flash_attn #(.HEADS(HEADS), .N(N), .DIM_Q(DIM_Q), .DIM_V(DIM_V),
               .BR(BR), .BC(BC), .TEMP_FP16(TEMP)) u_dut (
    .clk_i, .rst_ni, .start_i,
    .done_o(done_dut_o),
    .q_flat_i, .k_flat_i, .v_flat_i,
    .o_flat_o(o_dut_flat_o)
  );
  flash_attn_ref #(.HEADS(HEADS), .N(N), .DIM_Q(DIM_Q), .DIM_V(DIM_V),
                   .BR(BR), .BC(BC), .TEMP_FP16(TEMP)) u_ref (
    .clk_i, .rst_ni, .start_i,
    .done_o(done_ref_o),
    .q_flat_i, .k_flat_i, .v_flat_i,
    .o_flat_o(o_ref_flat_o)
  );
endmodule
