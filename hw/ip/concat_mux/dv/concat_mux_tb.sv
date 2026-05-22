// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// concat_mux_tb — Verilator TB wrapper. Instantiates DUT and REF on
// identical stimulus and exposes both output streams plus a sampled
// mismatch flag.
//
// Multiple parameterized wrappers are produced (one per (Ca,Cb,Pixels)
// triple) to support multi-test mode.

module concat_mux_tb #(
  parameter int Ca     = 2,
  parameter int Cb     = 3,
  parameter int Pixels = 4
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic              clr_i,

  input  logic              avalid_i,
  input  logic signed [7:0] adata_i,
  input  logic              bvalid_i,
  input  logic signed [7:0] bdata_i,
  input  logic              rready_i,

  // DUT
  output logic              aready_dut_o,
  output logic              bready_dut_o,
  output logic              rvalid_dut_o,
  output logic signed [7:0] rdata_dut_o,
  // REF
  output logic              aready_ref_o,
  output logic              bready_ref_o,
  output logic              rvalid_ref_o,
  output logic signed [7:0] rdata_ref_o,

  output logic              mismatch_o
);

  concat_mux #(.Ca(Ca), .Cb(Cb), .Pixels(Pixels)) u_dut (
    .clk_i    (clk_i),
    .rst_ni   (rst_ni),
    .clr_i    (clr_i),
    .avalid_i (avalid_i),
    .aready_o (aready_dut_o),
    .adata_i  (adata_i),
    .bvalid_i (bvalid_i),
    .bready_o (bready_dut_o),
    .bdata_i  (bdata_i),
    .rvalid_o (rvalid_dut_o),
    .rready_i (rready_i),
    .rdata_o  (rdata_dut_o)
  );

  concat_mux_ref #(.Ca(Ca), .Cb(Cb), .Pixels(Pixels)) u_ref (
    .clk_i    (clk_i),
    .rst_ni   (rst_ni),
    .clr_i    (clr_i),
    .avalid_i (avalid_i),
    .aready_o (aready_ref_o),
    .adata_i  (adata_i),
    .bvalid_i (bvalid_i),
    .bready_o (bready_ref_o),
    .bdata_i  (bdata_i),
    .rvalid_o (rvalid_ref_o),
    .rready_i (rready_i),
    .rdata_o  (rdata_ref_o)
  );

  assign mismatch_o =
      (rvalid_dut_o && rvalid_ref_o && (rdata_dut_o !== rdata_ref_o)) ||
      (rvalid_dut_o !== rvalid_ref_o) ||
      (aready_dut_o !== aready_ref_o) ||
      (bready_dut_o !== bready_ref_o);

endmodule

/* verilator lint_off DECLFILENAME */
module concat_mux_tb_c2_3_p4 (
  input  logic              clk_i,
  input  logic              rst_ni,
  input  logic              clr_i,
  input  logic              avalid_i,
  input  logic signed [7:0] adata_i,
  input  logic              bvalid_i,
  input  logic signed [7:0] bdata_i,
  input  logic              rready_i,
  output logic              aready_dut_o,
  output logic              bready_dut_o,
  output logic              rvalid_dut_o,
  output logic signed [7:0] rdata_dut_o,
  output logic              aready_ref_o,
  output logic              bready_ref_o,
  output logic              rvalid_ref_o,
  output logic signed [7:0] rdata_ref_o,
  output logic              mismatch_o
);
  concat_mux_tb #(.Ca(2), .Cb(3), .Pixels(4)) u (.*);
endmodule

module concat_mux_tb_c8_16_p8 (
  input  logic              clk_i,
  input  logic              rst_ni,
  input  logic              clr_i,
  input  logic              avalid_i,
  input  logic signed [7:0] adata_i,
  input  logic              bvalid_i,
  input  logic signed [7:0] bdata_i,
  input  logic              rready_i,
  output logic              aready_dut_o,
  output logic              bready_dut_o,
  output logic              rvalid_dut_o,
  output logic signed [7:0] rdata_dut_o,
  output logic              aready_ref_o,
  output logic              bready_ref_o,
  output logic              rvalid_ref_o,
  output logic signed [7:0] rdata_ref_o,
  output logic              mismatch_o
);
  concat_mux_tb #(.Ca(8), .Cb(16), .Pixels(8)) u (.*);
endmodule
/* verilator lint_on DECLFILENAME */
