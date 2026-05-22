// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// act_sigmoid_ref — behavioral golden for act_sigmoid.
//
// Independently coded: instead of pre-baking a ROM at elaboration, this
// reference recomputes sigmoid(x*InScale)/OutScale in `real` arithmetic
// on every cycle and then rounds. Same port list, same 1-cycle
// registered latency, same reset behaviour. Cycle-accurate equality with
// the DUT must hold for every codepoint and every `InScale`/`OutScale`
// pair.

module act_sigmoid_ref #(
  parameter real InScale  = 1.0/16.0,
  parameter real OutScale = 1.0/128.0
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic signed [7:0] x_i,
  output logic signed [7:0] y_o
);

  // Compute sigmoid(codepoint * InScale) / OutScale and quantise with
  // round-half-to-even + int8 saturation. Coded differently from the
  // DUT's function on purpose: works in scaled-output units, splits sign
  // and magnitude before rounding. Sigmoid output is non-negative so the
  // `neg` branch is unreachable in practice for any reasonable scale,
  // but it is kept for parity with the SiLU reference style.
  function automatic logic signed [7:0] sigmoid_q(input int codepoint);
    real    f;
    real    ef;
    real    s;
    real    scaled;
    real    mag;
    real    floor_mag;
    real    frac;
    longint integer_part;
    longint q;
    bit     neg;
    f = real'(codepoint) * InScale;
    if (f >= 0.0) begin
      s = 1.0 / (1.0 + $exp(-f));
    end else begin
      ef = $exp(f);
      s  = ef / (1.0 + ef);
    end
    scaled = s / OutScale;
    neg = (scaled < 0.0);
    mag = neg ? -scaled : scaled;
    floor_mag = real'($rtoi(mag));         // mag ≥ 0 so $rtoi == floor
    frac = mag - floor_mag;
    integer_part = longint'($rtoi(mag));
    // Round-half-to-even (see note in act_sigmoid.sv): no float sigmoid
    // input produces an exact-0.5 tie here, so the tie sub-case is
    // defensively present but never exercised.
    // verilator coverage_off
    if (frac == 0.5 && ((integer_part & 64'sd1) == 0)) begin
      q = integer_part;
    end else
    // verilator coverage_on
    if (frac >= 0.5) begin
      q = integer_part + 1;
    end else begin
      q = integer_part;
    end
    // verilator coverage_off
    if (neg) q = -q;
    // verilator coverage_on
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    sigmoid_q = q[7:0];
  endfunction

  logic signed [7:0] next_y;
  int                cp;

  always_comb begin
    cp     = int'(signed'(x_i));
    next_y = sigmoid_q(cp);
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      y_o <= '0;
    end else begin
      y_o <= next_y;
    end
  end

endmodule
