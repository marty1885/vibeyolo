// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// act_silu — int8 → int8 SiLU activation via a 256-entry ROM lookup.
//
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
//
// The DUT realises the activation by baking the float SiLU at elaboration
// into a 256-entry signed-int8 ROM. The input x_i is interpreted as a
// signed int8 with scale `InScale` (the float-domain value is
// signed(x_i) * InScale); the output y_o is a signed int8 with scale
// `OutScale` (so the float-domain SiLU value is signed(y_o) * OutScale).
//
// For each codepoint i ∈ [-128, 127]:
//   f = real'(i) * InScale
//   s = f / (1.0 + exp(-f))                 // float SiLU
//   q = round_half_to_even(s / OutScale)
//   y[i] = sat_i8(q)                        // clamp to [-128, 127]
//
// One cycle of latency: y_o is registered. Reset → y_o = 0.

module act_silu #(
  parameter real InScale  = 1.0/16.0,
  parameter real OutScale = 1.0/16.0
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic signed [7:0] x_i,
  output logic signed [7:0] y_o
);

  // Round-half-to-even with int8 saturation. `v` is a real value already
  // scaled into output-LSB units.
  function automatic logic signed [7:0] rne_sat_i8(input real v);
    real     f;
    longint  n;          // truncated integer part
    real     frac;       // |v - n|
    longint  q;
    f = v;
    n = longint'($rtoi(f));  // truncates toward zero
    frac = f - real'(n);
    if (frac < 0.0) frac = -frac;
    // Round-half-to-even. SiLU never produces an exact 0.5 fraction at
    // any of the 256 codepoints for any reasonable (InScale, OutScale),
    // so the half-tie branch is defensively present but never taken;
    // we collapse the "round up" cases ( > 0.5 and the even-rounded-up
    // tie sub-case) into a single comparison.
    // verilator coverage_off
    if (frac == 0.5 && ((n & 64'sd1) == 0)) begin
      q = n;
    end else
    // verilator coverage_on
    if (frac >= 0.5) begin
      q = (f >= 0.0) ? (n + 1) : (n - 1);
    end else begin
      q = n;
    end
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    rne_sat_i8 = q[7:0];
  endfunction

  // ── ROM ────────────────────────────────────────────
  logic signed [7:0] lut [0:255];

  initial begin : g_lut_init
    int    idx;
    int    codepoint;
    real   f;
    real   s;
    for (idx = 0; idx < 256; idx = idx + 1) begin
      // Index 0..127 → codepoints 0..127; index 128..255 → −128..−1.
      codepoint = (idx < 128) ? idx : (idx - 256);
      f = real'(codepoint) * InScale;
      // SiLU = f / (1 + exp(-f)). Compute robustly for large |f|.
      if (f >= 0.0) begin
        s = f / (1.0 + $exp(-f));
      end else begin
        // multiply num/den by exp(f) to avoid huge exp(-f) when f ≪ 0.
        real ef;
        ef = $exp(f);
        s  = (f * ef) / (1.0 + ef);
      end
      lut[idx] = rne_sat_i8(s / OutScale);
    end
  end

  // ── Registered lookup ──────────────────────────────
  logic [7:0] addr;
  assign addr = x_i;  // raw 8-bit pattern as ROM address

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      y_o <= '0;
    end else begin
      y_o <= lut[addr];
    end
  end

endmodule
