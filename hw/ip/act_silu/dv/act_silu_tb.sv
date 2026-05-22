// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// act_silu_tb — Verilator TB wrapper. Instantiates two parameterisations
// of act_silu (and matching act_silu_ref) so a single binary exercises
// both the "default" scales and a tighter scale pair that forces
// saturation at the int8 output. The C++ test drives identical stimulus
// into all four DUT/REF instances and compares cycle-by-cycle.

module act_silu_tb (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic signed [7:0] x_i,

  // Variant A: balanced scales, no saturation expected.
  output logic signed [7:0] yA_dut_o,
  output logic signed [7:0] yA_ref_o,
  output logic              mismatchA_o,

  // Variant B: large input scale + small output scale → forces saturation.
  output logic signed [7:0] yB_dut_o,
  output logic signed [7:0] yB_ref_o,
  output logic              mismatchB_o
);

  // ── Variant A: InScale=1/16, OutScale=1/16 ─────────
  act_silu #(
    .InScale  (1.0/16.0),
    .OutScale (1.0/16.0)
  ) u_dut_a (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (x_i),
    .y_o    (yA_dut_o)
  );

  act_silu_ref #(
    .InScale  (1.0/16.0),
    .OutScale (1.0/16.0)
  ) u_ref_a (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (x_i),
    .y_o    (yA_ref_o)
  );

  assign mismatchA_o = (yA_dut_o !== yA_ref_o);

  // ── Variant B: InScale=1/4, OutScale=1/64 — saturating ──
  act_silu #(
    .InScale  (1.0/4.0),
    .OutScale (1.0/512.0)
  ) u_dut_b (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (x_i),
    .y_o    (yB_dut_o)
  );

  act_silu_ref #(
    .InScale  (1.0/4.0),
    .OutScale (1.0/512.0)
  ) u_ref_b (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (x_i),
    .y_o    (yB_ref_o)
  );

  assign mismatchB_o = (yB_dut_o !== yB_ref_o);

endmodule
