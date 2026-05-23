// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
// flash_attn TB — mid1 config: HEADS=2, N=64, DIM_Q=16, DIM_V=32, BR=8, BC=16.

module flash_attn_tb_mid1 (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         start_i,
  output logic         done_dut_o,
  output logic         done_ref_o,
  input  logic [2*64*16*16-1:0] q_flat_i,
  input  logic [2*64*16*16-1:0] k_flat_i,
  input  logic [2*64*32*16-1:0] v_flat_i,
  output logic [2*64*32*16-1:0] o_dut_flat_o,
  output logic [2*64*32*16-1:0] o_ref_flat_o
);
  localparam int HEADS = 2;
  localparam int N     = 64;
  localparam int DIM_Q = 16;
  localparam int DIM_V = 32;
  localparam int BR    = 8;
  localparam int BC    = 16;
  // TEMP for DIM_Q=16: 1/sqrt(16) = 0.25 = fp16 0x3400
  localparam logic [15:0] TEMP = 16'h3400;

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
