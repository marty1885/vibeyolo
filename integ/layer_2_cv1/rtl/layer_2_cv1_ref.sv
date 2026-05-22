// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_2_cv1_ref — behavioral golden for layer_2_cv1.
//
// Independently coded reference: a 1-cycle clocked model that computes the
// full int8/fp16 pipeline (dotN + 2-phase accumulator + requant + SiLU) in
// `real` arithmetic, with the same port semantics as the DUT.
//
// Used as an independent SV cross-check (the C++ TB already compares the
// DUT against the Python-derived fp32 HW-ref); this reference is shipped
// for parity with other IPs in hw/ip/*.

module layer_2_cv1_ref #(
  parameter int  NCH_OUT = 32,
  parameter int  N_LANE  = 16,
  parameter real S_OUT_PRE  = 80.0 / 127.0,
  parameter real S_OUT_SILU = 80.0 / 127.0
) (
  input  logic                                          clk_i,
  input  logic                                          rst_ni,

  input  logic                                          valid_i,
  input  logic                                          phase_i,
  input  logic signed [N_LANE-1:0][7:0]                 x_i,
  input  logic signed [NCH_OUT-1:0][N_LANE-1:0][7:0]    w_i,
  input  logic        [NCH_OUT-1:0][15:0]               scale_fp16_i,
  input  logic        [NCH_OUT-1:0][15:0]               bias_fp16_i,

  output logic                                          valid_o,
  output logic signed [NCH_OUT-1:0][7:0]                y_o
);

  // ── Helpers ──
  function automatic real fp16_to_real(input logic [15:0] h);
    int unsigned sign_b, exp_f, mant;
    real         val, mantv;
    begin
      sign_b = h[15];
      exp_f  = h[14:10];
      mant   = h[9:0];
      if (exp_f == 0) begin
        mantv = real'(mant) / 1024.0;
        val   = mantv * (2.0 ** -14);
      end else if (exp_f == 31) begin
        val = 0.0;
      end else begin
        mantv = 1.0 + real'(mant) / 1024.0;
        val   = mantv * (2.0 ** (int'(exp_f) - 15));
      end
      if (sign_b != 0) val = -val;
      fp16_to_real = val;
    end
  endfunction

  function automatic logic signed [7:0] rne_sat_i8(input real v);
    longint n, q;
    real    f, frac;
    begin
      f = v;
      n = longint'($rtoi(f));
      frac = f - real'(n);
      if (frac < 0.0) frac = -frac;
      if (frac >= 0.5) q = (f >= 0.0) ? (n + 1) : (n - 1);
      else             q = n;
      if (q >  127) q =  127;
      if (q < -128) q = -128;
      rne_sat_i8 = q[7:0];
    end
  endfunction

  function automatic real silu_real(input real f);
    real ef;
    begin
      if (f >= 0.0) silu_real = f / (1.0 + $exp(-f));
      else begin
        ef = $exp(f);
        silu_real = (f * ef) / (1.0 + ef);
      end
    end
  endfunction

  // ── Per-channel phase-0 partial accumulator (clocked) ──
  integer partial_q [NCH_OUT];
  logic                                  valid_q;
  logic signed [NCH_OUT-1:0][7:0]        y_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int c = 0; c < NCH_OUT; c++) begin
        partial_q[c] <= 0;
        y_q[c]       <= 8'sd0;
      end
      valid_q <= 1'b0;
    end else begin
      valid_q <= 1'b0;
      // For each output channel, compute the phase dot.
      for (int c = 0; c < NCH_OUT; c++) begin
        int dot;
        dot = 0;
        for (int k = 0; k < N_LANE; k++) begin
          dot += $signed(x_i[k]) * $signed(w_i[c][k]);
        end
        if (valid_i) begin
          if (phase_i == 1'b0) begin
            partial_q[c] <= dot;
          end else begin
            real pre, silu;
            real scale_r, bias_r;
            int  q_pre;
            scale_r = fp16_to_real(scale_fp16_i[c]);
            bias_r  = fp16_to_real(bias_fp16_i[c]);
            pre = real'(partial_q[c] + dot) * scale_r + bias_r;
            q_pre = $signed({24'd0, rne_sat_i8(pre)});  // signed extend
            // Re-derive signed value (rne_sat_i8 returns [7:0]).
            if (q_pre >= 128) q_pre = q_pre - 256;
            // SiLU on pre-domain (i8 * S_OUT_PRE)
            silu = silu_real(real'(q_pre) * S_OUT_PRE) / S_OUT_SILU;
            y_q[c] <= rne_sat_i8(silu);
          end
        end
      end
      if (valid_i && phase_i == 1'b1) valid_q <= 1'b1;
    end
  end

  assign valid_o = valid_q;
  assign y_o     = y_q;

endmodule
