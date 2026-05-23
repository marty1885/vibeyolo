// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// box_affine_ref — independent SV behavioral golden for box_affine.
//
// Computes the entire decode in real arithmetic in the natural
// (anchor - d)*stride formulation, then rounds each output to fp16 (RNE).
// The DUT chains the same math through fp16_fma with per-step rounding and
// a folded ×(1/1280) for centers, so the TB applies a fp16-ULP tolerance.
// Latency matched to the DUT (7 cycles) for cycle-by-cycle comparison.

module box_affine_ref (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               valid_i,
  input  logic signed [7:0]  l_i,
  input  logic signed [7:0]  t_i,
  input  logic signed [7:0]  r_i,
  input  logic signed [7:0]  b_i,
  input  logic        [15:0] s_box_i,
  input  logic signed [31:0] col_i,
  input  logic signed [31:0] row_i,
  input  logic signed [31:0] stride_i,

  output logic               valid_o,
  output logic        [15:0] cx_o,
  output logic        [15:0] cy_o,
  output logic        [15:0] w_o,
  output logic        [15:0] h_o
);

  localparam int LATENCY = 7;

  function automatic real fp16_to_real(input logic [15:0] x);
    logic s; logic [4:0] eb; logic [9:0] f; real v, m; int e, k;
    begin
      s = x[15]; eb = x[14:10]; f = x[9:0];
      if (eb == 5'd31) begin
        v = (f == 0) ? 1.0e30 : 0.0;
      end else if (eb == 5'd0) begin
        v = real'(f) * (1.0 / 16777216.0);
      end else begin
        m = 1.0 + real'(f) / 1024.0;
        e = int'(eb) - 15;
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
      if (m >= 1.0) begin
        while (m >= 2.0) begin m = m / 2.0; e = e + 1; end
      end else begin
        while (m < 1.0) begin m = m * 2.0; e = e - 1; if (e < -30) break; end
      end
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

  // ─────── combinational decode in real ───────
  logic [15:0] cx_d, cy_d, w_d, h_d;
  always_comb begin
    real sbox, dl, dt, dr, db, ax, ay, sr, x1, y1, x2, y2;
    sbox = fp16_to_real(s_box_i);
    dl = real'(l_i) * sbox; dt = real'(t_i) * sbox;
    dr = real'(r_i) * sbox; db = real'(b_i) * sbox;
    ax = real'(col_i) + 0.5; ay = real'(row_i) + 0.5;
    sr = real'(stride_i);
    x1 = (ax - dl) * sr; x2 = (ax + dr) * sr;
    y1 = (ay - dt) * sr; y2 = (ay + db) * sr;
    cx_d = real_to_fp16(((x1 + x2) / 2.0) / 640.0);
    cy_d = real_to_fp16(((y1 + y2) / 2.0) / 640.0);
    w_d  = real_to_fp16((x2 - x1) / 640.0);
    h_d  = real_to_fp16((y2 - y1) / 640.0);
  end

  // ─────── latency-matching pipeline ───────
  logic        v_pipe  [0:LATENCY-1];
  logic [15:0] cx_pipe [0:LATENCY-1];
  logic [15:0] cy_pipe [0:LATENCY-1];
  logic [15:0] w_pipe  [0:LATENCY-1];
  logic [15:0] h_pipe  [0:LATENCY-1];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int k = 0; k < LATENCY; k++) begin
        v_pipe[k]<=1'b0; cx_pipe[k]<=16'h0; cy_pipe[k]<=16'h0;
        w_pipe[k]<=16'h0; h_pipe[k]<=16'h0;
      end
    end else begin
      v_pipe[0]<=valid_i; cx_pipe[0]<=cx_d; cy_pipe[0]<=cy_d;
      w_pipe[0]<=w_d; h_pipe[0]<=h_d;
      for (int k = 1; k < LATENCY; k++) begin
        v_pipe[k]<=v_pipe[k-1]; cx_pipe[k]<=cx_pipe[k-1]; cy_pipe[k]<=cy_pipe[k-1];
        w_pipe[k]<=w_pipe[k-1]; h_pipe[k]<=h_pipe[k-1];
      end
    end
  end

  assign valid_o = v_pipe[LATENCY-1];
  assign cx_o    = cx_pipe[LATENCY-1];
  assign cy_o    = cy_pipe[LATENCY-1];
  assign w_o     = w_pipe[LATENCY-1];
  assign h_o     = h_pipe[LATENCY-1];

endmodule
