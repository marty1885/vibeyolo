// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// maxpool_kxk_ref — behavioral golden for maxpool_kxk.
//
// Independently coded in a deliberately different style from the DUT: a
// flat linear scan that tracks a running max in an `always_comb` for-
// loop, initialised to int8 minimum. Same port list and same cycle-
// accurate contract as maxpool_kxk.

module maxpool_kxk_ref #(
  parameter int K = 5
) (
  input  logic                              clk_i,
  input  logic                              rst_ni,

  input  logic                              en_i,
  input  logic signed [K*K-1:0][7:0]        x_i,

  output logic signed [7:0]                 y_o
);

  localparam int N = K * K;

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
      y_o <= '0;
    end else if (en_i) begin
      y_o <= running_max;
    end
  end

endmodule
