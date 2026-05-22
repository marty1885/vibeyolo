// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// mac8_ref — behavioral golden for mac8.
//
// Independently coded reference using SystemVerilog `int` arithmetic so a
// coding error in mac8.sv won't be silently mirrored here. Same port list
// and same cycle-accurate contract as mac8.

module mac8_ref (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               clr_i,
  input  logic               en_i,
  input  logic signed [7:0]  a_i,
  input  logic signed [7:0]  b_i,

  output logic signed [31:0] acc_o
);

  int acc;
  int prod;

  always_comb begin
    // sign-extend through `int` multiply
    prod = int'(signed'(a_i)) * int'(signed'(b_i));
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      acc <= 0;
    end else if (clr_i) begin
      acc <= prod;
    end else if (en_i) begin
      acc <= acc + prod;
    end
  end

  assign acc_o = acc;

endmodule
