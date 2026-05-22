// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// prim_rst_sync — async-assert / sync-deassert reset synchronizer.
//
// Behavioral RTL. PD: replace this body with the tech-library synchronizer
// cell (typically two cascaded flops with anti-meta attributes, or a
// foundry-provided RST_SYNC macro). Do NOT change the port list.

module prim_rst_sync #(
  parameter int unsigned STAGES = 2
) (
  input  logic clk_i,
  input  logic rst_ni,    // async assert
  output logic rst_no     // sync-deassert, async-assert
);

  // Reset synchronizer intentionally uses rst_ni both async (in flop control)
  // and sync (in the shift). SYNCASYNCNET is expected and silenced.
  /* verilator lint_off SYNCASYNCNET */
  (* async_reg = "true" *) logic [STAGES-1:0] sync_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) sync_q <= '0;
    else         sync_q <= {sync_q[STAGES-2:0], 1'b1};
  end
  /* verilator lint_on SYNCASYNCNET */

  assign rst_no = sync_q[STAGES-1];

endmodule
