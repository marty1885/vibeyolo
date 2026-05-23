// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// dequant_n — N-lane int8 → fp16 dequantization with a shared fp16 scale.
//
//   y[i] = fp16( int8(x[i]) * scale )      (single fused rounding)
//
// Used by the detect head to dequantize the 80 gathered class logits of a
// selected anchor to fp16 (N=80, scale = that anchor's per-tensor cls
// S_OUT). Each lane is i32_to_fp16 (the int8 widens losslessly) feeding a
// fp16_fma that multiplies by the shared scale (c=0). Because fp16(int8) is
// exact, the fused product equals fp16(exact_i8 × scale_fp16) — so the DV
// golden matches BIT-EXACTLY (0 ULP), unlike the fp16-accumulating box path.
//
// LATENCY = 2 cycles (i32_to_fp16, then fma). Throughput 1 vector/cycle.

module dequant_n #(
  parameter int N = 80
) (
  input  logic                      clk_i,
  input  logic                      rst_ni,

  input  logic                      en_i,
  input  logic signed [N-1:0][7:0]  x_i,
  input  logic        [15:0]        scale_i,   // fp16, sampled with en_i

  output logic                      valid_o,
  output logic        [N-1:0][15:0] y_o
);

  // valid + scale alignment pipeline (2 stages).
  logic [1:0]  vq;
  logic [15:0] scale_s1;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      vq <= 2'b0; scale_s1 <= 16'h0;
    end else begin
      vq       <= {vq[0], en_i};
      scale_s1 <= scale_i;   // align with i32_to_fp16 (S1) output
    end
  end
  assign valid_o = vq[1];

  // verilator lint_off UNUSEDSIGNAL
  logic [4:0] sh [N];   // i32_to_fp16 auto-shift unused (int8 fits exactly)
  // verilator lint_on UNUSEDSIGNAL

  generate
    for (genvar gi = 0; gi < N; gi = gi + 1) begin : g_lane
      logic [15:0] xf_s1;
      i32_to_fp16 u_cvt (
        .clk_i, .rst_ni,
        .x_i    (32'(signed'(x_i[gi]))),
        .y_o    (xf_s1),
        .shift_o(sh[gi])
      );
      fp16_fma u_mul (
        .clk_i, .rst_ni,
        .a_i (xf_s1),
        .b_i (scale_s1),
        .c_i (16'h0000),
        .y_o (y_o[gi])
      );
    end
  endgenerate

endmodule
