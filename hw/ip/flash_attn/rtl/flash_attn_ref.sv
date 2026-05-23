// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// flash_attn_ref — independent behavioural SV golden for flash_attn.
//
// Uses SV `real` arithmetic to compute the reference output. It
// implements ordinary (non-tiled) softmax-attention with the same
// TEMP scale and produces fp16 outputs. Because the DUT uses tiled
// online softmax with fp16 throughout, exact bit-match is not
// expected; the test tolerates a per-output fp16-ULP budget.
//
// Interface mirrors flash_attn.sv. Behavioral model runs whenever
// `start_i` pulses; produces `done_o` one cycle later with the
// answer registered into o_flat_o.
//
// This is a *behavioral* reference — not for synthesis. The `real`
// arithmetic is gated to a single combinational task fired in an
// always_ff so it is not part of the synthesizable datapath.

module flash_attn_ref #(
  parameter int HEADS = 1,
  parameter int N     = 8,
  parameter int DIM_Q = 4,
  parameter int DIM_V = 4,
  parameter int BR    = 4,
  parameter int BC    = 4,
  parameter logic [15:0] TEMP_FP16 = 16'h31A8,
  parameter int MAX_CYC_HINT = 100000
) (
  input  logic clk_i,
  input  logic rst_ni,

  input  logic start_i,
  output logic done_o,

  input  logic [HEADS*N*DIM_Q*16-1:0] q_flat_i,
  input  logic [HEADS*N*DIM_Q*16-1:0] k_flat_i,
  input  logic [HEADS*N*DIM_V*16-1:0] v_flat_i,
  output logic [HEADS*N*DIM_V*16-1:0] o_flat_o
);

  // unused param suppressions
  /* verilator lint_off UNUSEDPARAM */
  localparam int _U_BR = BR;
  localparam int _U_BC = BC;
  localparam int _U_MCH = MAX_CYC_HINT;
  /* verilator lint_on UNUSEDPARAM */

  function automatic real fp16_to_real(input logic [15:0] x);
    int s, e, f;
    real v;
    begin
      s = (x >> 15) & 1;
      e = (x >> 10) & 31;
      f = x & 1023;
      if (e == 31) begin
        // best-effort: treat as 0 (not expected in tests)
        v = 0.0;
      end else if (e == 0) begin
        v = real'(f) * (1.0 / (1 << 24));
      end else begin
        v = (1.0 + real'(f) / 1024.0) * (2.0 ** (e - 15));
      end
      return s ? -v : v;
    end
  endfunction

  function automatic logic [15:0] real_to_fp16(input real v);
    logic        s;
    real         av, m, scaled, midpoint;
    int          e, biased;
    longint      iscaled, mant_int;
    logic [9:0]  frac10;
    begin
      if (v != v) return 16'h7E00;
      s = (v < 0.0);
      av = s ? -v : v;
      if (av == 0.0) return {s, 15'd0};
      if (av >= 65520.0) return {s, 5'b11111, 10'b0};
      e = 0; m = av;
      if (m >= 1.0) begin while (m >= 2.0) begin m = m / 2.0; e = e + 1; end end
      else          begin while (m < 1.0 && e > -30) begin m = m * 2.0; e = e - 1; end end
      biased = e + 15;
      if (biased >= 31) return {s, 5'b11111, 10'b0};
      if (biased <= 0) begin
        scaled = av * (1.0 * (1 << 24));
        iscaled = longint'($rtoi(scaled));
        if ((scaled - real'(iscaled)) > 0.5) iscaled = iscaled + 1;
        else if ((scaled - real'(iscaled)) == 0.5 && ((iscaled & 64'sd1) != 0)) iscaled = iscaled + 1;
        if (iscaled >= 1024) return {s, 5'd1, 10'd0};
        return {s, 5'd0, 10'(iscaled[9:0])};
      end
      scaled = (m - 1.0) * 1024.0;
      iscaled = longint'($rtoi(scaled));
      midpoint = real'(iscaled) + 0.5;
      mant_int = iscaled;
      if (scaled > midpoint) mant_int = iscaled + 1;
      else if (scaled == midpoint && ((iscaled & 64'sd1) != 0)) mant_int = iscaled + 1;
      // verilator coverage_off
      if (mant_int >= 1024) begin
        biased = biased + 1; mant_int = 0;
        if (biased >= 31) return {s, 5'b11111, 10'b0};
      end
      // verilator coverage_on
      frac10 = 10'(mant_int[9:0]);
      return {s, 5'(biased[4:0]), frac10};
    end
  endfunction

  logic done_q;
  logic [HEADS*N*DIM_V*16-1:0] o_q;
  assign done_o   = done_q;
  assign o_flat_o = o_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      done_q <= 1'b0;
      o_q    <= '0;
    end else begin
      done_q <= 1'b0;
      if (start_i) begin
        // Compute everything in `real` for golden.
        real Qr [HEADS][N][DIM_Q];
        real Kr [HEADS][N][DIM_Q];
        real Vr [HEADS][N][DIM_V];
        real S  [N];
        real P  [N];
        real Or [DIM_V];
        real mx, sum;
        real temp_r;
        temp_r = fp16_to_real(TEMP_FP16);
        for (int h = 0; h < HEADS; h++) begin
          for (int r = 0; r < N; r++) begin
            for (int k = 0; k < DIM_Q; k++) begin
              Qr[h][r][k] = fp16_to_real(q_flat_i[16*(h*N*DIM_Q + r*DIM_Q + k) +: 16]);
              Kr[h][r][k] = fp16_to_real(k_flat_i[16*(h*N*DIM_Q + r*DIM_Q + k) +: 16]);
            end
            for (int d = 0; d < DIM_V; d++)
              Vr[h][r][d] = fp16_to_real(v_flat_i[16*(h*N*DIM_V + r*DIM_V + d) +: 16]);
          end
        end
        for (int h = 0; h < HEADS; h++) begin
          for (int r = 0; r < N; r++) begin
            // S[j] = (Q[r] · K[j]) * temp
            for (int j = 0; j < N; j++) begin
              real acc;
              acc = 0.0;
              for (int k = 0; k < DIM_Q; k++) acc = acc + Qr[h][r][k] * Kr[h][j][k];
              S[j] = acc * temp_r;
            end
            // P = softmax(S)
            mx = S[0];
            for (int j = 1; j < N; j++) if (S[j] > mx) mx = S[j];
            sum = 0.0;
            for (int j = 0; j < N; j++) begin
              P[j] = $exp(S[j] - mx);
              sum  = sum + P[j];
            end
            for (int j = 0; j < N; j++) P[j] = P[j] / sum;
            // O[r][d] = sum_j P[j] * V[j][d]
            for (int d = 0; d < DIM_V; d++) begin
              real acc;
              acc = 0.0;
              for (int j = 0; j < N; j++) acc = acc + P[j] * Vr[h][j][d];
              Or[d] = acc;
            end
            for (int d = 0; d < DIM_V; d++)
              o_q[16*(h*N*DIM_V + r*DIM_V + d) +: 16] <= real_to_fp16(Or[d]);
          end
        end
        done_q <= 1'b1;
      end
    end
  end

endmodule
