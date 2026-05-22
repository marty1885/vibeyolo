// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// prim_clk_gate — integrated clock-gate cell wrapper.
//
// Behavioral RTL. PD: replace this body with the tech-library ICG (e.g.
// CKLNQD*). Do NOT change the port list. `test_en_i` bypasses the gate
// during scan to keep the clock free-running.

module prim_clk_gate (
  input  logic clk_i,
  input  logic en_i,
  input  logic test_en_i,
  output logic clk_o
);

  logic en_latch_q;

  // active-low latch around the enable
  always_latch begin
    if (!clk_i) en_latch_q = en_i | test_en_i;
  end

  assign clk_o = clk_i & en_latch_q;

endmodule
