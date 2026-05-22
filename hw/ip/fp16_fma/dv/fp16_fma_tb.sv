// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// fp16_fma_tb — Verilator TB wrapper. Runs DUT and REF in lockstep on
// identical stimulus and exposes both outputs plus a mismatch flag.

module fp16_fma_tb (
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic [15:0] a_i,
  input  logic [15:0] b_i,
  input  logic [15:0] c_i,

  output logic [15:0] y_dut_o,
  output logic [15:0] y_ref_o,
  output logic        mismatch_o
);

  fp16_fma u_dut (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (a_i),
    .b_i    (b_i),
    .c_i    (c_i),
    .y_o    (y_dut_o)
  );

  fp16_fma_ref u_ref (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (a_i),
    .b_i    (b_i),
    .c_i    (c_i),
    .y_o    (y_ref_o)
  );

  assign mismatch_o = (y_dut_o !== y_ref_o);

endmodule
