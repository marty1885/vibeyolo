// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// softmax16_ref — behavioural golden for softmax16.
//
// Computes softmax in `real` arithmetic and casts each output back to
// fp16 (RNE). Matches the DUT's pipeline latency (14 cycles) so the TB
// can compare cycle-by-cycle.

module softmax16_ref (
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic        valid_i,
  input  logic [15:0] x_i [16],

  output logic        valid_o,
  output logic [15:0] y_o [16]
);

  localparam int LATENCY = 16;   // == DUT 11 register stages + FMA_LAT(5)

  function automatic real fp16_to_real(input logic [15:0] x);
    logic        s;
    logic [4:0]  eb;
    logic [9:0]  f;
    real         v;
    begin
      s  = x[15];
      eb = x[14:10];
      f  = x[9:0];
      if (eb == 5'd31) begin
        if (f == 0) v = 1.0e30;        // Inf surrogate (DFL inputs bounded)
        else        v = 0.0;           // NaN → treat as 0 here (not used in real flow)
      end else if (eb == 5'd0) begin
        v = real'(f) * (1.0 / 16777216.0);    // f * 2^-24
      end else begin
        // (1 + f/1024) * 2^(eb-15)
        real m;
        int  e;
        int  k;
        m = 1.0 + real'(f) / 1024.0;
        e = int'(eb) - 15;
        if (e >= 0) begin
          for (k = 0; k < e; k = k + 1) m = m * 2.0;
        end else begin
          for (k = 0; k < -e; k = k + 1) m = m / 2.0;
        end
        v = m;
      end
      return s ? -v : v;
    end
  endfunction

  // Round-to-nearest-even real → fp16.
  function automatic logic [15:0] real_to_fp16(input real v);
    logic        s;
    real         av;
    int          e;
    real         m;
    real         scaled;
    longint      iscaled;
    real         frac_part;
    int          biased;
    longint      mant_int;
    begin
      // NaN passthrough — but softmax of finite inputs can't produce NaN.
      if (v != v) return 16'h7E00;
      s = (v < 0.0);
      av = s ? -v : v;
      if (av == 0.0) return {s, 15'd0};
      if (av >= 65520.0) return {s, 5'b11111, 10'b0};
      // Find e: 2^e <= av < 2^(e+1)
      e = 0;
      m = av;
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
        if (frac_part > 0.5)                                  iscaled = iscaled + 1;
        else if ((frac_part == 0.5) && ((iscaled & 64'sd1) != 0)) iscaled = iscaled + 1;
        if (iscaled >= 1024) return {s, 5'd1, 10'd0};
        return {s, 5'd0, 10'(iscaled[9:0])};
      end
      scaled = (m - 1.0) * 1024.0;
      iscaled = longint'($rtoi(scaled));
      frac_part = scaled - real'(iscaled);
      mant_int = iscaled;
      if (frac_part > 0.5)                                       mant_int = iscaled + 1;
      else if ((frac_part == 0.5) && ((iscaled & 64'sd1) != 0))  mant_int = iscaled + 1;
      // verilator coverage_off
      if (mant_int >= 1024) begin
        biased = biased + 1;
        mant_int = 0;
        if (biased >= 31) return {s, 5'b11111, 10'b0};
      end
      // verilator coverage_on
      return {s, 5'(biased[4:0]), 10'(mant_int[9:0])};
    end
  endfunction

  // Combinational softmax in real.
  logic [15:0] y_d [16];
  always_comb begin
    real xv [16];
    real mx;
    real e_arr [16];
    real s;
    int  i;
    for (i = 0; i < 16; i = i + 1) xv[i] = fp16_to_real(x_i[i]);
    mx = xv[0];
    for (i = 1; i < 16; i = i + 1) if (xv[i] > mx) mx = xv[i];
    s = 0.0;
    for (i = 0; i < 16; i = i + 1) begin
      e_arr[i] = $exp(xv[i] - mx);
      s = s + e_arr[i];
    end
    for (i = 0; i < 16; i = i + 1) y_d[i] = real_to_fp16(e_arr[i] / s);
  end

  // Latency-matching pipeline registers (12 stages).
  logic        v_pipe [0:LATENCY-1];
  logic [15:0] y_pipe [0:LATENCY-1][16];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int k = 0; k < LATENCY; k++) begin
        v_pipe[k] <= 1'b0;
        for (int i = 0; i < 16; i++) y_pipe[k][i] <= 16'h0;
      end
    end else begin
      v_pipe[0] <= valid_i;
      for (int i = 0; i < 16; i++) y_pipe[0][i] <= y_d[i];
      for (int k = 1; k < LATENCY; k++) begin
        v_pipe[k] <= v_pipe[k-1];
        for (int i = 0; i < 16; i++) y_pipe[k][i] <= y_pipe[k-1][i];
      end
    end
  end

  assign valid_o = v_pipe[LATENCY-1];
  always_comb for (int i = 0; i < 16; i++) y_o[i] = y_pipe[LATENCY-1][i];

endmodule
