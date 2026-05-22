// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// dotN_ref — behavioral golden for dotN.
//
// Independently coded in a deliberately different style from the DUT:
//   - DUT instantiates N mac8 multipliers and builds an explicit
//     generate-loop pipelined adder tree.
//   - REF uses a single combinational reduction (linear summation over
//     `int` accumulator) followed by a LATENCY-deep shift register of
//     32-bit values, so we walk a flat shift-line instead of a tree.
//     If the DUT tree had a structural bug the REF won't share it.
//
// Same port list and same cycle-accurate latency contract: LATENCY =
// 1 + ceil(log2(N)) cycles from en_i to valid_o.

module dotN_ref #(
  parameter int N = 16
) (
  input  logic                       clk_i,
  input  logic                       rst_ni,

  input  logic                       en_i,
  input  logic                       clr_i,
  input  logic signed [N-1:0][7:0]   a_i,
  input  logic signed [N-1:0][7:0]   b_i,

  output logic signed [31:0]         y_o,
  output logic                       valid_o
);

  localparam int LEVELS  = (N <= 1) ? 0 : $clog2(N);
  localparam int LATENCY = 1 + LEVELS;

  // Combinational sum of all N signed products using `int` arithmetic.
  // When en_i is low or clr_i is high we feed 0 into the pipeline so
  // the staged result follows the same gating as the DUT.
  int sum_comb;
  always_comb begin
    sum_comb = 0;
    if (en_i && !clr_i) begin
      for (int i = 0; i < N; i++) begin
        sum_comb = sum_comb +
            (int'(signed'(a_i[i])) * int'(signed'(b_i[i])));
      end
    end
  end

  // Shift register of 32-bit accumulators, LATENCY deep.
  logic signed [31:0] sum_pipe [LATENCY];
  logic               vld_pipe [LATENCY];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < LATENCY; i++) begin
        sum_pipe[i] <= 32'sd0;
        vld_pipe[i] <= 1'b0;
      end
    end else if (clr_i) begin
      for (int i = 0; i < LATENCY; i++) begin
        sum_pipe[i] <= 32'sd0;
        vld_pipe[i] <= 1'b0;
      end
    end else begin
      sum_pipe[0] <= 32'(sum_comb);
      vld_pipe[0] <= en_i;
      for (int i = 1; i < LATENCY; i++) begin
        sum_pipe[i] <= sum_pipe[i-1];
        vld_pipe[i] <= vld_pipe[i-1];
      end
    end
  end

  assign y_o     = sum_pipe[LATENCY-1];
  assign valid_o = vld_pipe[LATENCY-1];

endmodule
