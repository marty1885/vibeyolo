// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// dequant_n_ref — independent behavioral golden for dequant_n.
//
// Computes each lane in real arithmetic and rounds once to fp16 (RNE):
//   y[i] = real_to_fp16( real(int8 x[i]) * fp16_to_real(scale) )
// Since fp16(int8) is exact, this is bit-identical to the DUT's fused
// fma(fp16(x), scale, 0) — the TB asserts 0 ULP. Latency matched (5).

module dequant_n_ref #(
  parameter int N = 80
) (
  input  logic                      clk_i,
  input  logic                      rst_ni,

  input  logic                      en_i,
  input  logic signed [N-1:0][7:0]  x_i,
  input  logic        [15:0]        scale_i,

  output logic                      valid_o,
  output logic        [N-1:0][15:0] y_o
);

  localparam int LATENCY = 5;   // == DUT I2F_LAT(2) + FMA_LAT(3)

  function automatic real fp16_to_real(input logic [15:0] x);
    logic s; logic [4:0] eb; logic [9:0] f; real v, m; int e, k;
    begin
      s = x[15]; eb = x[14:10]; f = x[9:0];
      if (eb == 5'd31)      v = (f == 0) ? 1.0e30 : 0.0;
      else if (eb == 5'd0)  v = real'(f) * (1.0 / 16777216.0);
      else begin
        m = 1.0 + real'(f) / 1024.0; e = int'(eb) - 15;
        if (e >= 0) for (k = 0; k < e;  k = k + 1) m = m * 2.0;
        else        for (k = 0; k < -e; k = k + 1) m = m / 2.0;
        v = m;
      end
      return s ? -v : v;
    end
  endfunction

  function automatic logic [15:0] real_to_fp16(input real v);
    logic s; real av, m, scaled, frac_part; int e, biased;
    longint iscaled, mant_int;
    begin
      if (v != v) return 16'h7E00;
      s = (v < 0.0); av = s ? -v : v;
      if (av == 0.0) return {s, 15'd0};
      if (av >= 65520.0) return {s, 5'b11111, 10'b0};
      e = 0; m = av;
      if (m >= 1.0) while (m >= 2.0) begin m = m / 2.0; e = e + 1; end
      else while (m < 1.0) begin m = m * 2.0; e = e - 1; if (e < -30) break; end
      biased = e + 15;
      if (biased >= 31) return {s, 5'b11111, 10'b0};
      if (biased <= 0) begin
        scaled = av * (1.0 * (1 << 24));
        iscaled = longint'($rtoi(scaled));
        frac_part = scaled - real'(iscaled);
        if (frac_part > 0.5)                                      iscaled = iscaled + 1;
        else if ((frac_part == 0.5) && ((iscaled & 64'sd1) != 0)) iscaled = iscaled + 1;
        if (iscaled >= 1024) return {s, 5'd1, 10'd0};
        return {s, 5'd0, 10'(iscaled[9:0])};
      end
      scaled = (m - 1.0) * 1024.0;
      iscaled = longint'($rtoi(scaled));
      frac_part = scaled - real'(iscaled);
      mant_int = iscaled;
      if (frac_part > 0.5)                                      mant_int = iscaled + 1;
      else if ((frac_part == 0.5) && ((iscaled & 64'sd1) != 0)) mant_int = iscaled + 1;
      // verilator coverage_off
      if (mant_int >= 1024) begin
        biased = biased + 1; mant_int = 0;
        if (biased >= 31) return {s, 5'b11111, 10'b0};
      end
      // verilator coverage_on
      return {s, 5'(biased[4:0]), 10'(mant_int[9:0])};
    end
  endfunction

  logic [15:0] y_comb [N];
  always_comb begin
    real sc;
    sc = fp16_to_real(scale_i);
    for (int i = 0; i < N; i++)
      y_comb[i] = real_to_fp16(real'($signed(x_i[i])) * sc);
  end

  logic        v_pipe [0:LATENCY-1];
  logic [15:0] y_pipe [0:LATENCY-1][N];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int k = 0; k < LATENCY; k++) begin
        v_pipe[k] <= 1'b0;
        for (int i = 0; i < N; i++) y_pipe[k][i] <= 16'h0;
      end
    end else begin
      v_pipe[0] <= en_i;
      for (int i = 0; i < N; i++) y_pipe[0][i] <= y_comb[i];
      for (int k = 1; k < LATENCY; k++) begin
        v_pipe[k] <= v_pipe[k-1];
        for (int i = 0; i < N; i++) y_pipe[k][i] <= y_pipe[k-1][i];
      end
    end
  end

  assign valid_o = v_pipe[LATENCY-1];
  always_comb for (int i = 0; i < N; i++) y_o[i] = y_pipe[LATENCY-1][i];

endmodule
