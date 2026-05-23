// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// reduce_max_n_ref — behavioral golden for reduce_max_n.
//
// Independently coded in a deliberately different style from the DUT: a
// flat linear scan tracking a running max in an always_comb for-loop,
// initialised to the int8 minimum. Same port list and cycle-accurate
// contract as reduce_max_n.

module reduce_max_n_ref #(
  parameter int N = 80
) (
  input  logic                          clk_i,
  input  logic                          rst_ni,

  input  logic                          en_i,
  input  logic signed [N-1:0][7:0]      x_i,

  output logic                          valid_o,
  output logic signed [7:0]             y_o
);

  logic signed [7:0] running_max;
  logic signed [7:0] cand;

  always_comb begin
    running_max = 8'sh80;  // -128, identity for signed int8 max
    for (int i = 0; i < N; i++) begin
      cand = x_i[i];
      if (cand > running_max) begin
        running_max = cand;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      y_o     <= '0;
      valid_o <= 1'b0;
    end else begin
      valid_o <= en_i;
      if (en_i) begin
        y_o <= running_max;
      end
    end
  end

endmodule
