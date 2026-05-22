// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// skip_buf_tb — Verilator TB wrapper. Instantiates DUT and REF on
// identical stimulus and exposes both output streams plus a sampled
// mismatch flag.

module skip_buf_tb #(
  parameter int H = 4,
  parameter int W = 4,
  parameter int C = 4
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic              clr_i,

  input  logic              wvalid_i,
  input  logic signed [7:0] wdata_i,
  input  logic              rready_i,

  // DUT
  output logic              wready_dut_o,
  output logic              rvalid_dut_o,
  output logic signed [7:0] rdata_dut_o,
  output logic              full_dut_o,
  output logic              empty_dut_o,
  // REF
  output logic              wready_ref_o,
  output logic              rvalid_ref_o,
  output logic signed [7:0] rdata_ref_o,
  output logic              full_ref_o,
  output logic              empty_ref_o,

  output logic              mismatch_o
);

  skip_buf #(.H(H), .W(W), .C(C)) u_dut (
    .clk_i    (clk_i),
    .rst_ni   (rst_ni),
    .clr_i    (clr_i),
    .wvalid_i (wvalid_i),
    .wready_o (wready_dut_o),
    .wdata_i  (wdata_i),
    .rvalid_o (rvalid_dut_o),
    .rready_i (rready_i),
    .rdata_o  (rdata_dut_o),
    .full_o   (full_dut_o),
    .empty_o  (empty_dut_o)
  );

  skip_buf_ref #(.H(H), .W(W), .C(C)) u_ref (
    .clk_i    (clk_i),
    .rst_ni   (rst_ni),
    .clr_i    (clr_i),
    .wvalid_i (wvalid_i),
    .wready_o (wready_ref_o),
    .wdata_i  (wdata_i),
    .rvalid_o (rvalid_ref_o),
    .rready_i (rready_i),
    .rdata_o  (rdata_ref_o),
    .full_o   (full_ref_o),
    .empty_o  (empty_ref_o)
  );

  assign mismatch_o =
      (wready_dut_o !== wready_ref_o) ||
      (rvalid_dut_o !== rvalid_ref_o) ||
      (full_dut_o   !== full_ref_o)   ||
      (empty_dut_o  !== empty_ref_o)  ||
      (rvalid_dut_o && rvalid_ref_o && (rdata_dut_o !== rdata_ref_o));

endmodule

// Concrete wrappers for multi-test build (H=W=C=4 → Depth=64).
/* verilator lint_off DECLFILENAME */
module skip_buf_tb_small (
  input  logic              clk_i,
  input  logic              rst_ni,
  input  logic              clr_i,
  input  logic              wvalid_i,
  input  logic signed [7:0] wdata_i,
  input  logic              rready_i,
  output logic              wready_dut_o,
  output logic              rvalid_dut_o,
  output logic signed [7:0] rdata_dut_o,
  output logic              full_dut_o,
  output logic              empty_dut_o,
  output logic              wready_ref_o,
  output logic              rvalid_ref_o,
  output logic signed [7:0] rdata_ref_o,
  output logic              full_ref_o,
  output logic              empty_ref_o,
  output logic              mismatch_o
);
  skip_buf_tb #(.H(4), .W(4), .C(4)) u (.*);
endmodule
/* verilator lint_on DECLFILENAME */

// H=W=C=8 → Depth=512 (stress for wider address decode).
/* verilator lint_off DECLFILENAME */
module skip_buf_tb_large (
  input  logic              clk_i,
  input  logic              rst_ni,
  input  logic              clr_i,
  input  logic              wvalid_i,
  input  logic signed [7:0] wdata_i,
  input  logic              rready_i,
  output logic              wready_dut_o,
  output logic              rvalid_dut_o,
  output logic signed [7:0] rdata_dut_o,
  output logic              full_dut_o,
  output logic              empty_dut_o,
  output logic              wready_ref_o,
  output logic              rvalid_ref_o,
  output logic signed [7:0] rdata_ref_o,
  output logic              full_ref_o,
  output logic              empty_ref_o,
  output logic              mismatch_o
);
  skip_buf_tb #(.H(8), .W(8), .C(8)) u (.*);
endmodule
/* verilator lint_on DECLFILENAME */
