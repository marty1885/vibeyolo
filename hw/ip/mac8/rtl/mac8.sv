// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// mac8 — signed int8 × int8 multiply-accumulate.
//
// Atomic compute leaf. One signed 8×8 multiply per cycle, accumulated into
// a signed int32. Synchronous clear; enable gates accumulation. Wraps on
// int32 overflow (no saturation — caller sizes the accumulator chain to
// avoid wrap in normal use).
//
// Contract:
//   - On posedge clk_i with !rst_ni: acc_o ← 0.
//   - Else if clr_i:                acc_o ← signed(a_i) * signed(b_i)
//                                   (load with current product; enables
//                                    chained "first-tap" semantics).
//   - Else if en_i:                 acc_o ← acc_o + signed(a_i)*signed(b_i)
//   - Else:                         acc_o holds.

module mac8 (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               clr_i,      // synchronous clear-and-load
  input  logic               en_i,       // accumulate enable
  input  logic signed [7:0]  a_i,
  input  logic signed [7:0]  b_i,

  output logic signed [31:0] acc_o
);

  logic signed [15:0] prod;
  assign prod = a_i * b_i;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      acc_o <= '0;
    end else if (clr_i) begin
      acc_o <= 32'(prod);
    end else if (en_i) begin
      acc_o <= acc_o + 32'(prod);
    end
  end

endmodule
