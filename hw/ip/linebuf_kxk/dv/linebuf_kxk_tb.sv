// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// linebuf_kxk_tb — Verilator TB wrapper. Instantiates DUT and REF on
// identical stimulus and exposes both output streams plus a sampled
// mismatch flag.
//
// Concrete wrappers are emitted per (K, H, W, Channels) configuration to
// support multi-test mode without recompiling the same module.

module linebuf_kxk_tb #(
  parameter int K        = 3,
  parameter int W        = 4,
  parameter int H        = 4,
  parameter int Channels = 1
) (
  input  logic                                clk_i,
  input  logic                                rst_ni,
  input  logic                                clr_i,

  input  logic                                wvalid_i,
  input  logic signed [Channels-1:0][7:0]     wdata_i,
  input  logic                                rready_i,

  // DUT
  output logic                                wready_dut_o,
  output logic                                rvalid_dut_o,
  output logic signed [K*K*Channels-1:0][7:0] rdata_dut_o,
  // REF
  output logic                                wready_ref_o,
  output logic                                rvalid_ref_o,
  output logic signed [K*K*Channels-1:0][7:0] rdata_ref_o,

  output logic                                mismatch_o
);

  linebuf_kxk #(.K(K), .W(W), .H(H), .Channels(Channels)) u_dut (
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

  linebuf_kxk_ref #(.K(K), .W(W), .H(H), .Channels(Channels)) u_ref (
    .clk_i    (clk_i),
    .rst_ni   (rst_ni),
    .clr_i    (clr_i),
    .wvalid_i (wvalid_i),
    .wready_o (wready_ref_o),
    .wdata_i  (wdata_i),
    .rvalid_o (rvalid_ref_o),
    .rdata_o  (rdata_ref_o),
    .rready_i (rready_i)
  );

  // Mismatch only meaningful when both have valid output beats. DUT and
  // REF share identical handshake semantics so they should be
  // cycle-aligned; gating is defensive.
  assign mismatch_o =
      (rvalid_dut_o && rvalid_ref_o && (rdata_dut_o !== rdata_ref_o)) ||
      (rvalid_dut_o !== rvalid_ref_o) ||
      (wready_dut_o !== wready_ref_o);

endmodule

// Concrete wrapper: K=3 W=4 H=4 Channels=1.
/* verilator lint_off DECLFILENAME */
module linebuf_kxk_tb_k3w4 (
  input  logic              clk_i,
  input  logic              rst_ni,
  input  logic              clr_i,
  input  logic              wvalid_i,
  input  logic signed [7:0] wdata_i,
  input  logic              rready_i,
  output logic              wready_dut_o,
  output logic              rvalid_dut_o,
  output logic signed [8:0][7:0] rdata_dut_o,
  output logic              wready_ref_o,
  output logic              rvalid_ref_o,
  output logic signed [8:0][7:0] rdata_ref_o,
  output logic              mismatch_o
);
  linebuf_kxk_tb #(.K(3), .W(4), .H(4), .Channels(1)) u (.*);
endmodule
/* verilator lint_on DECLFILENAME */

// Concrete wrapper: K=5 W=8 H=8 Channels=1.
/* verilator lint_off DECLFILENAME */
module linebuf_kxk_tb_k5w8 (
  input  logic              clk_i,
  input  logic              rst_ni,
  input  logic              clr_i,
  input  logic              wvalid_i,
  input  logic signed [7:0] wdata_i,
  input  logic              rready_i,
  output logic              wready_dut_o,
  output logic              rvalid_dut_o,
  output logic signed [24:0][7:0] rdata_dut_o,
  output logic              wready_ref_o,
  output logic              rvalid_ref_o,
  output logic signed [24:0][7:0] rdata_ref_o,
  output logic              mismatch_o
);
  linebuf_kxk_tb #(.K(5), .W(8), .H(8), .Channels(1)) u (.*);
endmodule
/* verilator lint_on DECLFILENAME */

// Concrete wrapper: K=3 W=8 H=8 Channels=16.
//   wdata width  = 16*8  = 128b
//   rdata width  = 9*16*8 = 1152b (9 patch cells * 16 channels * 8b)
/* verilator lint_off DECLFILENAME */
module linebuf_kxk_tb_k3w8c16 (
  input  logic                   clk_i,
  input  logic                   rst_ni,
  input  logic                   clr_i,
  input  logic                   wvalid_i,
  input  logic signed [15:0][7:0] wdata_i,
  input  logic                   rready_i,
  output logic                   wready_dut_o,
  output logic                   rvalid_dut_o,
  output logic signed [9*16-1:0][7:0] rdata_dut_o,
  output logic                   wready_ref_o,
  output logic                   rvalid_ref_o,
  output logic signed [9*16-1:0][7:0] rdata_ref_o,
  output logic                   mismatch_o
);
  linebuf_kxk_tb #(.K(3), .W(8), .H(8), .Channels(16)) u (.*);
endmodule
/* verilator lint_on DECLFILENAME */

// Concrete wrapper: K=3 W=8 H=8 Channels=64.
//   wdata width  = 64*8  = 512b
//   rdata width  = 9*64*8 = 4608b
/* verilator lint_off DECLFILENAME */
module linebuf_kxk_tb_k3w8c64 (
  input  logic                   clk_i,
  input  logic                   rst_ni,
  input  logic                   clr_i,
  input  logic                   wvalid_i,
  input  logic signed [63:0][7:0] wdata_i,
  input  logic                   rready_i,
  output logic                   wready_dut_o,
  output logic                   rvalid_dut_o,
  output logic signed [9*64-1:0][7:0] rdata_dut_o,
  output logic                   wready_ref_o,
  output logic                   rvalid_ref_o,
  output logic signed [9*64-1:0][7:0] rdata_ref_o,
  output logic                   mismatch_o
);
  linebuf_kxk_tb #(.K(3), .W(8), .H(8), .Channels(64)) u (.*);
endmodule
/* verilator lint_on DECLFILENAME */
