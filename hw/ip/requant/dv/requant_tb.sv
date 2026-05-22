// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// requant_tb — Verilator TB wrapper. Drives DUT (`requant`, which composes
// the three sub-IPs) and REF (an independent flat behavioral model) with
// identical stimulus and exposes both outputs plus a per-cycle mismatch
// flag aligned on the valid bit.

module requant_tb (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               valid_i,
  input  logic signed [31:0] acc_i,
  input  logic        [15:0] scale_fp16_i,
  input  logic        [15:0] bias_fp16_i,

  output logic               valid_dut_o,
  output logic               valid_ref_o,
  output logic signed  [7:0] y_dut_o,
  output logic signed  [7:0] y_ref_o,
  output logic               mismatch_o
);

  requant u_dut (
    .clk_i        (clk_i),
    .rst_ni       (rst_ni),
    .valid_i      (valid_i),
    .acc_i        (acc_i),
    .scale_fp16_i (scale_fp16_i),
    .bias_fp16_i  (bias_fp16_i),
    .valid_o      (valid_dut_o),
    .y_o          (y_dut_o)
  );

  requant_ref u_ref (
    .clk_i        (clk_i),
    .rst_ni       (rst_ni),
    .valid_i      (valid_i),
    .acc_i        (acc_i),
    .scale_fp16_i (scale_fp16_i),
    .bias_fp16_i  (bias_fp16_i),
    .valid_o      (valid_ref_o),
    .y_o          (y_ref_o)
  );

  // Only meaningful when valid_dut_o is high (and by construction the
  // shift-registers in DUT and REF stay in lockstep, so valid_ref_o ==
  // valid_dut_o each cycle as well).
  assign mismatch_o = (y_dut_o !== y_ref_o) || (valid_dut_o !== valid_ref_o);

endmodule
