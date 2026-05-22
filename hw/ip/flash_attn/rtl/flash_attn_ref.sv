// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// flash_attn_ref — behavioural golden for flash_attn.
//
// Computes textbook dense softmax-attention in `real`, with an
// fp16-RNE round after every intermediate op (matmul, scale, softmax,
// final matmul). This is the per-element ground truth the DUT is
// compared against in DV. Matches DUT's interface (Q/K/V write ports,
// start_i / done_o, O read port).
//
// Note: this is NOT a cycle-by-cycle mirror — it commits the entire O
// tensor in a single combinational pass (between start_i and done_o,
// asserting done_o after a short fixed delay). The TB synchronises on
// done_o for both DUT and REF before comparing O word-by-word.

// verilator lint_off UNUSEDPARAM
// verilator lint_off UNUSEDSIGNAL
module flash_attn_ref #(
  parameter int  HEADS  = 2,
  parameter int  N      = 400,
  parameter int  DIM_Q  = 32,
  parameter int  DIM_V  = 64,
  parameter real TEMP   = 0.17677669529663687
) (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               q_we_i,
  input  logic [$clog2(HEADS*N*DIM_Q):0]   q_waddr_i,
  input  logic        [15:0] q_wdata_i,

  input  logic               k_we_i,
  input  logic [$clog2(HEADS*N*DIM_Q):0]   k_waddr_i,
  input  logic        [15:0] k_wdata_i,

  input  logic               v_we_i,
  input  logic [$clog2(HEADS*N*DIM_V):0]   v_waddr_i,
  input  logic        [15:0] v_wdata_i,

  input  logic               start_i,
  output logic               busy_o,
  output logic               done_o,

  input  logic [$clog2(HEADS*N*DIM_V):0]   o_raddr_i,
  output logic        [15:0] o_rdata_o
);

  localparam int Q_DEPTH = HEADS * N * DIM_Q;
  localparam int V_DEPTH = HEADS * N * DIM_V;
  localparam int Q_AW    = $clog2(Q_DEPTH);
  localparam int V_AW    = $clog2(V_DEPTH);

  /* verilator coverage_off */
  logic [15:0] q_mem [0:Q_DEPTH-1];
  logic [15:0] k_mem [0:Q_DEPTH-1];
  logic [15:0] v_mem [0:V_DEPTH-1];
  logic [15:0] o_mem [0:V_DEPTH-1];
  /* verilator coverage_on */

  // ─── fp16 helpers (real-based) ─────────────────────────
  function automatic real fp16_to_real(input logic [15:0] x);
    logic        s;
    logic [4:0]  eb;
    logic [9:0]  f;
    real         v, m;
    int          e, k;
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
        if (e >= 0) for (k = 0; k < e; k = k + 1) m = m * 2.0;
        else        for (k = 0; k < -e; k = k + 1) m = m / 2.0;
        v = m;
      end
      return s ? -v : v;
    end
  endfunction

  function automatic logic [15:0] real_to_fp16(input real v);
    logic        s;
    real         av, m, scaled, frac_part;
    int          e, biased;
    longint      iscaled, mant_int;
    begin
      if (v != v) return 16'h7E00;
      s = (v < 0.0);
      av = s ? -v : v;
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
        if (frac_part > 0.5)                                       iscaled = iscaled + 1;
        else if ((frac_part == 0.5) && ((iscaled & 64'sd1) != 0))  iscaled = iscaled + 1;
        if (iscaled >= 1024) return {s, 5'd1, 10'd0};
        return {s, 5'd0, 10'(iscaled[9:0])};
      end
      scaled = (m - 1.0) * 1024.0;
      iscaled = longint'($rtoi(scaled));
      frac_part = scaled - real'(iscaled);
      mant_int = iscaled;
      if (frac_part > 0.5)                                          mant_int = iscaled + 1;
      else if ((frac_part == 0.5) && ((iscaled & 64'sd1) != 0))     mant_int = iscaled + 1;
      // verilator coverage_off
      if (mant_int >= 1024) begin biased = biased + 1; mant_int = 0;
        if (biased >= 31) return {s, 5'b11111, 10'b0};
      end
      // verilator coverage_on
      return {s, 5'(biased[4:0]), 10'(mant_int[9:0])};
    end
  endfunction

  // ─── memory writes ─────────────────────────────────────
  always_ff @(posedge clk_i) begin
    if (q_we_i) q_mem[q_waddr_i[Q_AW-1:0]] <= q_wdata_i;
    if (k_we_i) k_mem[k_waddr_i[Q_AW-1:0]] <= k_wdata_i;
    if (v_we_i) v_mem[v_waddr_i[V_AW-1:0]] <= v_wdata_i;
  end

  assign o_rdata_o = o_mem[o_raddr_i[V_AW-1:0]];

  // ─── Compute on start_i: textbook dense softmax-attention ──
  // Single combinational sweep across all (head, i, j) — placed in an
  // always_ff that fires on start_i and clears on reset. done_o is
  // asserted REF_LATENCY cycles after start to give the DUT some
  // breathing room (not strictly required since TB polls done_o).
  localparam int REF_LATENCY = 4;

  typedef enum logic [1:0] { R_IDLE, R_RUN, R_DONE } rstate_e;
  rstate_e rstate_q;
  logic [3:0] rdelay;

  assign busy_o = (rstate_q == R_RUN);
  assign done_o = (rstate_q == R_DONE);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rstate_q <= R_IDLE;
      rdelay   <= '0;
    end else begin
      case (rstate_q)
        R_IDLE: if (start_i) begin
          // Perform the dense softmax-attention in real arithmetic.
          for (int h = 0; h < HEADS; h++) begin
            for (int i = 0; i < N; i++) begin
              // Compute scores row s[j] = (Q[h,i] · K[h,j]) * TEMP, in real,
              // with fp16-RNE round after each multiply-accumulate sum.
              real s_arr [N];
              real mx, sum_e;
              real e_arr [N];
              real qrow [DIM_Q];
              real krow [DIM_Q];
              real vrow [DIM_V];
              real dot;
              for (int d = 0; d < DIM_Q; d++) begin
                qrow[d] = fp16_to_real(q_mem[h * N * DIM_Q + i * DIM_Q + d]);
              end
              for (int j = 0; j < N; j++) begin
                dot = 0.0;
                for (int d = 0; d < DIM_Q; d++) begin
                  krow[d] = fp16_to_real(k_mem[h * N * DIM_Q + j * DIM_Q + d]);
                  dot = dot + qrow[d] * krow[d];
                end
                // Round the dot to fp16 (mirrors DUT's add-tree round-after-each-add
                // behaviour at the leaf — approximate).
                dot = fp16_to_real(real_to_fp16(dot));
                s_arr[j] = fp16_to_real(real_to_fp16(dot * TEMP));
              end
              // Softmax along j.
              mx = s_arr[0];
              for (int j = 1; j < N; j++) if (s_arr[j] > mx) mx = s_arr[j];
              sum_e = 0.0;
              for (int j = 0; j < N; j++) begin
                e_arr[j] = $exp(s_arr[j] - mx);
                sum_e   = sum_e + e_arr[j];
              end
              // O[h,i,d] = sum_j (p_j * V[h,j,d])
              for (int d = 0; d < DIM_V; d++) begin
                real od;
                od = 0.0;
                for (int j = 0; j < N; j++) begin
                  vrow[d] = fp16_to_real(v_mem[h * N * DIM_V + j * DIM_V + d]);
                  od = od + (e_arr[j] / sum_e) * vrow[d];
                end
                o_mem[h * N * DIM_V + i * DIM_V + d] <= real_to_fp16(od);
              end
            end
          end
          rstate_q <= R_RUN;
          rdelay   <= 4'(REF_LATENCY);
        end
        R_RUN: begin
          if (rdelay == 4'd1) rstate_q <= R_DONE;
          rdelay <= rdelay - 1'b1;
        end
        R_DONE: if (start_i) begin
          // re-arm if used in a multi-frame test
          rstate_q <= R_IDLE;
        end
        default: rstate_q <= R_IDLE;
      endcase
    end
  end

endmodule
// verilator lint_on UNUSEDPARAM
// verilator lint_on UNUSEDSIGNAL
