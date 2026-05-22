// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// softmax16_tb — Verilator TB wrapper. Drives DUT and REF with the
// same 16-lane fp16 vector and exposes both outputs plus a per-lane
// fp16-ULP-difference signal. The C++ test does the tolerance check.

module softmax16_tb (
  input  logic         clk_i,
  input  logic         rst_ni,

  input  logic         valid_i,
  input  logic [255:0] x_flat_i,        // 16 × 16-bit lanes, lane 0 in LSBs

  output logic         valid_dut_o,
  output logic         valid_ref_o,
  output logic [255:0] y_dut_flat_o,
  output logic [255:0] y_ref_flat_o
);

  logic [15:0] x_lanes [16];
  logic [15:0] y_dut   [16];
  logic [15:0] y_ref   [16];

  always_comb begin
    for (int i = 0; i < 16; i++) x_lanes[i] = x_flat_i[16*i +: 16];
  end

  softmax16 u_dut (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .valid_i (valid_i),
    .x_i     (x_lanes),
    .valid_o (valid_dut_o),
    .y_o     (y_dut)
  );

  softmax16_ref u_ref (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .valid_i (valid_i),
    .x_i     (x_lanes),
    .valid_o (valid_ref_o),
    .y_o     (y_ref)
  );

  always_comb begin
    for (int i = 0; i < 16; i++) begin
      y_dut_flat_o[16*i +: 16] = y_dut[i];
      y_ref_flat_o[16*i +: 16] = y_ref[i];
    end
  end

endmodule
