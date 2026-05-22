// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// act_sigmoid_tb — Verilator TB wrapper. Instantiates two
// parameterisations of act_sigmoid (and matching act_sigmoid_ref) so a
// single binary exercises both a non-saturating scale pair and a tighter
// pair that forces upper-rail saturation. The C++ test drives identical
// stimulus into all four DUT/REF instances and compares cycle-by-cycle.

module act_sigmoid_tb (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic signed [7:0] x_i,

  // Variant A: InScale=1/16, OutScale=1/128 — non-saturating; sigmoid
  // range [0,1] maps to quantised [0,127].
  output logic signed [7:0] yA_dut_o,
  output logic signed [7:0] yA_ref_o,
  output logic              mismatchA_o,

  // Variant B: InScale=1/16, OutScale=1/256 — upper-rail saturating;
  // sigmoid values above 127/256 ≈ 0.496 clip to +127.
  output logic signed [7:0] yB_dut_o,
  output logic signed [7:0] yB_ref_o,
  output logic              mismatchB_o
);

  // ── Variant A: InScale=1/16, OutScale=1/128 — non-saturating ─────
  act_sigmoid #(
    .InScale  (1.0/16.0),
    .OutScale (1.0/128.0)
  ) u_dut_a (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (x_i),
    .y_o    (yA_dut_o)
  );

  act_sigmoid_ref #(
    .InScale  (1.0/16.0),
    .OutScale (1.0/128.0)
  ) u_ref_a (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (x_i),
    .y_o    (yA_ref_o)
  );

  assign mismatchA_o = (yA_dut_o !== yA_ref_o);

  // ── Variant B: InScale=1/16, OutScale=1/256 — upper-rail saturating ──
  act_sigmoid #(
    .InScale  (1.0/16.0),
    .OutScale (1.0/256.0)
  ) u_dut_b (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (x_i),
    .y_o    (yB_dut_o)
  );

  act_sigmoid_ref #(
    .InScale  (1.0/16.0),
    .OutScale (1.0/256.0)
  ) u_ref_b (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (x_i),
    .y_o    (yB_ref_o)
  );

  assign mismatchB_o = (yB_dut_o !== yB_ref_o);

endmodule
