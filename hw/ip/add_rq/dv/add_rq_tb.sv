// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// add_rq_tb — Verilator TB wrapper. Runs DUT and REF in lockstep on
// identical stimulus and exposes both outputs plus a mismatch flag.

module add_rq_tb (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               valid_i,
  input  logic signed  [7:0] a_i8_i,
  input  logic signed  [7:0] b_i8_i,
  input  logic        [15:0] scale_a_fp16_i,
  input  logic        [15:0] scale_b_fp16_i,
  input  logic        [15:0] inv_out_scale_fp16_i,
  input  logic        [15:0] bias_fp16_i,

  output logic               valid_dut_o,
  output logic signed  [7:0] y_dut_o,
  output logic               valid_ref_o,
  output logic signed  [7:0] y_ref_o,
  output logic               mismatch_o
);

  add_rq u_dut (
    .clk_i               (clk_i),
    .rst_ni              (rst_ni),
    .valid_i             (valid_i),
    .a_i8_i              (a_i8_i),
    .b_i8_i              (b_i8_i),
    .scale_a_fp16_i      (scale_a_fp16_i),
    .scale_b_fp16_i      (scale_b_fp16_i),
    .inv_out_scale_fp16_i(inv_out_scale_fp16_i),
    .bias_fp16_i         (bias_fp16_i),
    .valid_o             (valid_dut_o),
    .y_o                 (y_dut_o)
  );

  add_rq_ref u_ref (
    .clk_i               (clk_i),
    .rst_ni              (rst_ni),
    .valid_i             (valid_i),
    .a_i8_i              (a_i8_i),
    .b_i8_i              (b_i8_i),
    .scale_a_fp16_i      (scale_a_fp16_i),
    .scale_b_fp16_i      (scale_b_fp16_i),
    .inv_out_scale_fp16_i(inv_out_scale_fp16_i),
    .bias_fp16_i         (bias_fp16_i),
    .valid_o             (valid_ref_o),
    .y_o                 (y_ref_o)
  );

  // Mismatch only counts when both DUTs report valid. The DUT pipeline
  // delivers valid_o aligned with y_o; same for REF.
  assign mismatch_o = (valid_dut_o !== valid_ref_o) ||
                      (valid_dut_o && (y_dut_o !== y_ref_o));

endmodule
