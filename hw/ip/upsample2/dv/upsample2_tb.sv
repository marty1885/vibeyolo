// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// upsample2_tb — Verilator TB wrapper. Instantiates DUT and REF on
// identical stimulus and exposes both output streams plus a sampled
// mismatch flag (only meaningful when both rvalid_o are high).
//
// A separate parameterized wrapper (upsample2_tb_w<W>) is produced per
// width to support multi-test mode without recompiling the same module.

module upsample2_tb #(
  parameter int W = 4
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
  // REF
  output logic              wready_ref_o,
  output logic              rvalid_ref_o,
  output logic signed [7:0] rdata_ref_o,

  output logic              mismatch_o
);

  upsample2 #(.W(W), .Channels(1)) u_dut (
    .clk_i    (clk_i),
    .rst_ni   (rst_ni),
    .clr_i    (clr_i),
    .wvalid_i (wvalid_i),
    .wready_o (wready_dut_o),
    .wdata_i  (wdata_i),
    .rvalid_o (rvalid_dut_o),
    .rready_i (rready_i),
    .rdata_o  (rdata_dut_o)
  );

  upsample2_ref #(.W(W), .Channels(1)) u_ref (
    .clk_i    (clk_i),
    .rst_ni   (rst_ni),
    .clr_i    (clr_i),
    .wvalid_i (wvalid_i),
    .wready_o (wready_ref_o),
    .wdata_i  (wdata_i),
    .rvalid_o (rvalid_ref_o),
    .rready_i (rready_i),
    .rdata_o  (rdata_ref_o)
  );

  // Sampled mismatch: only meaningful when both sides have a valid output
  // beat. Because DUT and REF share identical back-pressure semantics in
  // this design they should be cycle-aligned in practice; the gating here
  // is defensive.
  assign mismatch_o =
      (rvalid_dut_o && rvalid_ref_o && (rdata_dut_o !== rdata_ref_o)) ||
      (rvalid_dut_o !== rvalid_ref_o) ||
      (wready_dut_o !== wready_ref_o);

endmodule

// Concrete wrapper for W=4 multi-test build.
/* verilator lint_off DECLFILENAME */
module upsample2_tb_w4 (
  input  logic              clk_i,
  input  logic              rst_ni,
  input  logic              clr_i,
  input  logic              wvalid_i,
  input  logic signed [7:0] wdata_i,
  input  logic              rready_i,
  output logic              wready_dut_o,
  output logic              rvalid_dut_o,
  output logic signed [7:0] rdata_dut_o,
  output logic              wready_ref_o,
  output logic              rvalid_ref_o,
  output logic signed [7:0] rdata_ref_o,
  output logic              mismatch_o
);
  upsample2_tb #(.W(4)) u (.*);
endmodule
/* verilator lint_on DECLFILENAME */

// Concrete wrapper for W=8 multi-test build.
/* verilator lint_off DECLFILENAME */
module upsample2_tb_w8 (
  input  logic              clk_i,
  input  logic              rst_ni,
  input  logic              clr_i,
  input  logic              wvalid_i,
  input  logic signed [7:0] wdata_i,
  input  logic              rready_i,
  output logic              wready_dut_o,
  output logic              rvalid_dut_o,
  output logic signed [7:0] rdata_dut_o,
  output logic              wready_ref_o,
  output logic              rvalid_ref_o,
  output logic signed [7:0] rdata_ref_o,
  output logic              mismatch_o
);
  upsample2_tb #(.W(8)) u (.*);
endmodule
/* verilator lint_on DECLFILENAME */
