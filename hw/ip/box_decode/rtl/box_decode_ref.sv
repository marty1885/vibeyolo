// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// box_decode_ref — independent SV behavioral golden for box_decode.
//
// Uses real-valued arithmetic for the entire computation, then rounds
// each of x1,y1,x2,y2 to fp16 (RNE) at the output. Pipeline latency is
// matched to the DUT (19 cycles) so the TB can compare cycle-by-cycle.
//
// This is an "ideal real" reference; the DUT does the same computation
// chained through fp16_fma at every step, so per-step fp16 rounding
// error accumulates. The TB applies a ULP tolerance.

module box_decode_ref (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               valid_i,
  input  logic        [15:0] p_l_i [16],
  input  logic        [15:0] p_t_i [16],
  input  logic        [15:0] p_r_i [16],
  input  logic        [15:0] p_b_i [16],
  input  logic signed [15:0] cx_i,
  input  logic signed [15:0] cy_i,
  input  logic signed [15:0] stride_i,

  output logic               valid_o,
  output logic        [15:0] x1_o,
  output logic        [15:0] y1_o,
  output logic        [15:0] x2_o,
  output logic        [15:0] y2_o
);

  localparam int LATENCY = 19;

  function automatic real fp16_to_real(input logic [15:0] x);
    logic       s;
    logic [4:0] eb;
    logic [9:0] f;
    real        v;
    real        m;
    int         e, k;
    begin
      s  = x[15];
      eb = x[14:10];
      f  = x[9:0];
      if (eb == 5'd31) begin
        v = (f == 0) ? 1.0e30 : 0.0;
      end else if (eb == 5'd0) begin
        v = real'(f) * (1.0 / 16777216.0);
      end else begin
        m = 1.0 + real'(f) / 1024.0;
        e = int'(eb) - 15;
        if (e >= 0) begin
          for (k = 0; k < e;  k = k + 1) m = m * 2.0;
        end else begin
          for (k = 0; k < -e; k = k + 1) m = m / 2.0;
        end
        v = m;
      end
      return s ? -v : v;
    end
  endfunction

  function automatic logic [15:0] real_to_fp16(input real v);
    logic       s;
    real        av, m, scaled, frac_part;
    int         e, biased;
    longint     iscaled, mant_int;
    begin
      if (v != v) return 16'h7E00;
      s  = (v < 0.0);
      av = s ? -v : v;
      if (av == 0.0) return {s, 15'd0};
      if (av >= 65520.0) return {s, 5'b11111, 10'b0};
      e = 0;
      m = av;
      if (m >= 1.0) begin
        while (m >= 2.0) begin m = m / 2.0; e = e + 1; end
      end else begin
        while (m < 1.0) begin
          m = m * 2.0;
          e = e - 1;
          if (e < -30) break;
        end
      end
      biased = e + 15;
      if (biased >= 31) return {s, 5'b11111, 10'b0};
      if (biased <= 0) begin
        scaled    = av * (1.0 * (1 << 24));
        iscaled   = longint'($rtoi(scaled));
        frac_part = scaled - real'(iscaled);
        if (frac_part > 0.5)                                        iscaled = iscaled + 1;
        else if ((frac_part == 0.5) && ((iscaled & 64'sd1) != 0))   iscaled = iscaled + 1;
        if (iscaled >= 1024) return {s, 5'd1, 10'd0};
        return {s, 5'd0, 10'(iscaled[9:0])};
      end
      scaled    = (m - 1.0) * 1024.0;
      iscaled   = longint'($rtoi(scaled));
      frac_part = scaled - real'(iscaled);
      mant_int  = iscaled;
      if (frac_part > 0.5)                                       mant_int = iscaled + 1;
      else if ((frac_part == 0.5) && ((iscaled & 64'sd1) != 0))  mant_int = iscaled + 1;
      // verilator coverage_off
      if (mant_int >= 1024) begin
        biased   = biased + 1;
        mant_int = 0;
        if (biased >= 31) return {s, 5'b11111, 10'b0};
      end
      // verilator coverage_on
      return {s, 5'(biased[4:0]), 10'(mant_int[9:0])};
    end
  endfunction

  // ─────── combinational decode in real ───────
  logic [15:0] x1_d, y1_d, x2_d, y2_d;
  always_comb begin
    real p_real [4][16];
    real d [4];
    real cxr, cyr, sr, cx_c, cy_c;
    int  s, i;
    for (s = 0; s < 4; s = s + 1)
      for (i = 0; i < 16; i = i + 1) begin
        case (s)
          0: p_real[s][i] = fp16_to_real(p_l_i[i]);
          1: p_real[s][i] = fp16_to_real(p_t_i[i]);
          2: p_real[s][i] = fp16_to_real(p_r_i[i]);
          default: p_real[s][i] = fp16_to_real(p_b_i[i]);
        endcase
      end
    for (s = 0; s < 4; s = s + 1) begin
      d[s] = 0.0;
      for (i = 0; i < 16; i = i + 1)
        d[s] = d[s] + p_real[s][i] * real'(i);
    end
    cxr  = real'(cx_i);
    cyr  = real'(cy_i);
    sr   = real'(stride_i);
    cx_c = (cxr + 0.5) * sr;
    cy_c = (cyr + 0.5) * sr;
    x1_d = real_to_fp16(cx_c - d[0] * sr);
    y1_d = real_to_fp16(cy_c - d[1] * sr);
    x2_d = real_to_fp16(cx_c + d[2] * sr);
    y2_d = real_to_fp16(cy_c + d[3] * sr);
  end

  // ─────── latency-matching pipeline ───────
  logic        v_pipe   [0:LATENCY-1];
  logic [15:0] x1_pipe  [0:LATENCY-1];
  logic [15:0] y1_pipe  [0:LATENCY-1];
  logic [15:0] x2_pipe  [0:LATENCY-1];
  logic [15:0] y2_pipe  [0:LATENCY-1];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int k = 0; k < LATENCY; k++) begin
        v_pipe[k]  <= 1'b0;
        x1_pipe[k] <= 16'h0;
        y1_pipe[k] <= 16'h0;
        x2_pipe[k] <= 16'h0;
        y2_pipe[k] <= 16'h0;
      end
    end else begin
      v_pipe[0]  <= valid_i;
      x1_pipe[0] <= x1_d;
      y1_pipe[0] <= y1_d;
      x2_pipe[0] <= x2_d;
      y2_pipe[0] <= y2_d;
      for (int k = 1; k < LATENCY; k++) begin
        v_pipe[k]  <= v_pipe[k-1];
        x1_pipe[k] <= x1_pipe[k-1];
        y1_pipe[k] <= y1_pipe[k-1];
        x2_pipe[k] <= x2_pipe[k-1];
        y2_pipe[k] <= y2_pipe[k-1];
      end
    end
  end

  assign valid_o = v_pipe[LATENCY-1];
  assign x1_o    = x1_pipe[LATENCY-1];
  assign y1_o    = y1_pipe[LATENCY-1];
  assign x2_o    = x2_pipe[LATENCY-1];
  assign y2_o    = y2_pipe[LATENCY-1];

endmodule
