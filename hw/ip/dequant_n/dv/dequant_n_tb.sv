// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// dequant_n_tb — DUT + behavioral REF in lockstep. Outputs are expected
// bit-identical (fp16(int8) is exact), so mismatch_o is a hard per-cycle
// equality check across all N lanes and valid.

module dequant_n_tb #(
  parameter int N = 80
) (
  input  logic                      clk_i,
  input  logic                      rst_ni,

  input  logic                      en_i,
  input  logic signed [N-1:0][7:0]  x_i,
  input  logic        [15:0]        scale_i,

  output logic                      valid_dut_o,
  output logic        [N-1:0][15:0] y_dut_o,
  output logic                      valid_ref_o,
  output logic        [N-1:0][15:0] y_ref_o,
  output logic                      mismatch_o
);

  dequant_n #(.N(N)) u_dut (
    .clk_i, .rst_ni, .en_i, .x_i, .scale_i,
    .valid_o(valid_dut_o), .y_o(y_dut_o)
  );
  dequant_n_ref #(.N(N)) u_ref (
    .clk_i, .rst_ni, .en_i, .x_i, .scale_i,
    .valid_o(valid_ref_o), .y_o(y_ref_o)
  );

  logic ymism;
  always_comb begin
    ymism = (valid_dut_o !== valid_ref_o);
    for (int i = 0; i < N; i++)
      if (y_dut_o[i] !== y_ref_o[i]) ymism = 1'b1;
  end
  assign mismatch_o = ymism;

endmodule
