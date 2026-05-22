// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// flash_attn — systolic-friendly, tile-streaming flash-attention leaf IP.
//
// Computes scaled-dot-product self-attention for one or more heads, never
// materializing the full N×N scores tensor. Uses the online-softmax
// (flash-attention v1) recurrence:
//
//   For each row i (Q row):
//     m_i = -inf, l_i = 0, O_i[*] = 0
//     For each col j (K/V col):
//       s_ij  = (Q[i] · K[j]) * TEMP
//       m_new = max(m_i, s_ij)
//       alpha = exp(m_i - m_new)
//       p     = exp(s_ij - m_new)
//       l_i   = alpha * l_i + p
//       O_i[d] = alpha * O_i[d] + p * V[j][d]    for d ∈ [0, DIM_V)
//       m_i   = m_new
//     O_i[d] /= l_i                              for d ∈ [0, DIM_V)
//
// All arithmetic is fp16 (IEEE-754 binary16, RNE rounding). The
// implementation never materialises an N×N scores tensor (only one s_ij
// at a time).
//
// Parallelism (this build, BR=1, BC=1 — "row-sequential, col-sequential",
// but dot product over DIM_Q and O update over DIM_V are fully parallel):
//   * Q·K[j] dot product uses DIM_Q parallel fp16_fma cells + an
//     fp16 adder tree → 1 throughput / pipeline-latency cycles per col.
//   * O_i update uses DIM_V parallel fp16_fma cells → 1 cycle per col.
// Total cells = DIM_Q + DIM_V (= 96 at YOLO26n shapes). The col-step is
// implemented as a small sequential state machine — each col takes a
// fixed handful of cycles. Higher parallelism (BR>1, BC>1) is a future
// enhancement; the algorithm and exp/recip LUT primitives don't change.
//
// Interface — RAM-write style. Caller writes Q, K, V via per-port
// (we_i, waddr_i, wdata_i) triplets, pulses start_i, waits for done_o,
// then reads O via (raddr_i, rdata_o). All addresses are flat (head, n,
// d) majoring on head then n then d.
//
// Cycle estimate for YOLO26n PSA (HEADS=2, N=400, DIM_Q=32, DIM_V=64):
//   ~8-10 cycles per col (one (head, row, col) inner iter), plus a
//   DIM_V-cycle normalize per row. Total ~ HEADS * N * (N*K + DIM_V)
//   with K≈9, ≈ 2 * 400 * (400*9 + 64) ≈ 2.93M cyc. This is over the
//   T_FRAME=100k budget — higher parallelism (BR/BC unroll) is the
//   straight-line follow-up. The IP boundary and DV golden are written
//   parametrically so that follow-up doesn't touch the contract.

// verilator lint_off UNUSEDPARAM
// verilator lint_off UNUSEDSIGNAL
module flash_attn #(
  parameter int  HEADS  = 2,
  parameter int  N      = 400,
  parameter int  DIM_Q  = 32,
  parameter int  DIM_V  = 64,
  // Softmax temperature applied to s_ij before the running max. Default
  // 1/sqrt(32) ≈ 0.17677669 for YOLO26n PSA/A2C2f.
  parameter real TEMP   = 0.17677669529663687
) (
  input  logic               clk_i,
  input  logic               rst_ni,

  // ─── Q/K/V write ports (per-fp16 word) ─────────────────
  input  logic               q_we_i,
  input  logic [$clog2(HEADS*N*DIM_Q):0]   q_waddr_i,
  input  logic        [15:0] q_wdata_i,

  input  logic               k_we_i,
  input  logic [$clog2(HEADS*N*DIM_Q):0]   k_waddr_i,
  input  logic        [15:0] k_wdata_i,

  input  logic               v_we_i,
  input  logic [$clog2(HEADS*N*DIM_V):0]   v_waddr_i,
  input  logic        [15:0] v_wdata_i,

  // ─── Control ───────────────────────────────────────────
  input  logic               start_i,
  output logic               busy_o,
  output logic               done_o,

  // ─── O read port ───────────────────────────────────────
  input  logic [$clog2(HEADS*N*DIM_V):0]   o_raddr_i,
  output logic        [15:0] o_rdata_o
);

  // ───────────────────────────────────────────────────────
  // Address-space layout
  //   Q : head-major, then n-major, then dim-major (flat index)
  //         q_addr = head * N * DIM_Q + n * DIM_Q + d
  //   K : same shape as Q
  //   V : head-major, n-major, dim-major (DIM_V)
  //         v_addr = head * N * DIM_V + n * DIM_V + d
  //   O : same shape as V
  // ───────────────────────────────────────────────────────

  localparam int Q_DEPTH = HEADS * N * DIM_Q;
  localparam int V_DEPTH = HEADS * N * DIM_V;
  localparam int Q_AW    = $clog2(Q_DEPTH);
  localparam int V_AW    = $clog2(V_DEPTH);

  // Memories. Verilator will pick BRAMs/inferred memory; for ASIC PD
  // they'd be replaced by prim_ram_1p with this signature. For DV, just
  // an array.
  /* verilator coverage_off */
  logic [15:0] q_mem [0:Q_DEPTH-1];
  logic [15:0] k_mem [0:Q_DEPTH-1];
  logic [15:0] v_mem [0:V_DEPTH-1];
  logic [15:0] o_mem [0:V_DEPTH-1];
  /* verilator coverage_on */

  // ─── fp16 helpers ──────────────────────────────────────
  // Borrowed pattern from softmax16. Same fp16_add, fp16_max, real_to_fp16.

  function automatic logic fp16_gt(input logic [15:0] a, input logic [15:0] b);
    logic sa, sb;
    logic [14:0] ma, mb;
    logic a_zero, b_zero;
    begin
      sa = a[15]; sb = b[15];
      a_zero = (a[14:0] == 15'd0);
      b_zero = (b[14:0] == 15'd0);
      ma = a[14:0]; mb = b[14:0];
      if (a_zero && b_zero)      fp16_gt = 1'b0;
      else if (sa != sb)         fp16_gt = !sa;
      else if (sa == 1'b0)       fp16_gt = (ma > mb);
      else                       fp16_gt = (ma < mb);
    end
  endfunction

  function automatic logic [15:0] fp16_max(
      input logic [15:0] a, input logic [15:0] b);
    fp16_max = fp16_gt(a, b) ? a : b;
  endfunction

  function automatic logic [15:0] fp16_neg(input logic [15:0] a);
    fp16_neg = {~a[15], a[14:0]};
  endfunction

  // fp16 add — same kernel as in softmax16. Bounded operands; no extreme
  // dynamic range expected from attention inputs.
  function automatic logic [15:0] fp16_add(
      input logic [15:0] a, input logic [15:0] b);
    logic        sa, sb;
    logic [4:0]  ea_b, eb_b;
    logic [9:0]  fa, fb;
    logic        a_z, b_z, a_inf, b_inf, a_nan, b_nan;
    logic [10:0] sig_a, sig_b;
    int          ea, eb;
    int          anchor, sh;
    logic        sticky;
    logic [31:0] big_a, big_b, sum_mag, tmp;
    logic        result_sign;
    int          msb, biased_exp, eshift, shift10;
    logic        guard, round_b, st;
    logic [9:0]  mant10;
    logic [10:0] mant_r;
    begin
      sa = a[15]; ea_b = a[14:10]; fa = a[9:0];
      sb = b[15]; eb_b = b[14:10]; fb = b[9:0];
      a_z = (ea_b == 0) && (fa == 0);
      b_z = (eb_b == 0) && (fb == 0);
      a_inf = (ea_b == 31) && (fa == 0);
      b_inf = (eb_b == 31) && (fb == 0);
      a_nan = (ea_b == 31) && (fa != 0);
      b_nan = (eb_b == 31) && (fb != 0);
      if (a_nan || b_nan) return 16'h7E00;
      if (a_inf && b_inf && (sa != sb)) return 16'h7E00;
      if (a_inf) return a;
      if (b_inf) return b;
      if (a_z && b_z) return 16'h0000;
      if (a_z) return b;
      if (b_z) return a;
      if (ea_b == 0) begin sig_a = {1'b0, fa}; ea = -24; end
      else            begin sig_a = {1'b1, fa}; ea = int'(ea_b) - 25; end
      if (eb_b == 0) begin sig_b = {1'b0, fb}; eb = -24; end
      else            begin sig_b = {1'b1, fb}; eb = int'(eb_b) - 25; end
      anchor = (ea < eb) ? ea : eb;
      sticky = 1'b0;
      sh = ea - anchor;
      if (sh > 20) begin tmp = 32'd0; sticky = 1'b1; big_a = tmp; end
      else            big_a = {21'd0, sig_a} << sh;
      sh = eb - anchor;
      if (sh > 20) begin tmp = 32'd0; sticky = 1'b1; big_b = tmp; end
      else            big_b = {21'd0, sig_b} << sh;
      if (sa == sb) begin
        sum_mag = big_a + big_b;
        result_sign = sa;
      end else begin
        if (big_a >= big_b) begin sum_mag = big_a - big_b; result_sign = sa; end
        else                begin sum_mag = big_b - big_a; result_sign = sb; end
      end
      if (sum_mag == 32'd0 && !sticky) return 16'h0000;
      msb = 0;
      for (int i = 31; i >= 0; i--) if (sum_mag[i] && (msb == 0)) msb = i;
      biased_exp = anchor + msb + 15;
      eshift     = msb - 10;
      if (biased_exp >= 31) return {result_sign, 5'b11111, 10'b0};
      if (biased_exp <= 0) begin
        eshift = eshift + (1 - biased_exp);
        biased_exp = 0;
      end
      shift10 = eshift;
      if (shift10 < 0) begin
        tmp = sum_mag << (-shift10); mant10 = tmp[9:0];
        guard = 1'b0; round_b = 1'b0; st = sticky;
      end else if (shift10 == 0) begin
        mant10 = sum_mag[9:0]; guard = 1'b0; round_b = 1'b0; st = sticky;
      end else if (shift10 == 1) begin
        mant10 = sum_mag[10:1]; guard = sum_mag[0]; round_b = 1'b0; st = sticky;
      end else if (shift10 == 2) begin
        mant10 = sum_mag[11:2]; guard = sum_mag[1]; round_b = sum_mag[0]; st = sticky;
      end else begin
        tmp = sum_mag >> shift10;       mant10  = tmp[9:0];
        tmp = sum_mag >> (shift10 - 1); guard   = tmp[0];
        tmp = sum_mag >> (shift10 - 2); round_b = tmp[0];
        st = sticky;
        for (int j = 0; j < 32; j++) if (j < shift10 - 2 && sum_mag[j]) st = 1'b1;
      end
      mant_r = {1'b0, mant10} + 11'((guard && ((round_b | st) || mant10[0])) ? 1 : 0);
      if (mant_r[10]) begin biased_exp = biased_exp + 1; mant_r = 11'b0; end
      if (biased_exp >= 31) return {result_sign, 5'b11111, 10'b0};
      if (biased_exp <= 0)  return {result_sign, 5'd0, mant_r[9:0]};
      return {result_sign, 5'(biased_exp[4:0]), mant_r[9:0]};
    end
  endfunction

  // real → fp16 RNE (elaboration-only).
  function automatic logic [15:0] real_to_fp16(input real v);
    logic        s;
    real         av;
    int          e, biased;
    real         m, scaled, midpoint;
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
        while (m < 1.0 && e > -30) begin m = m * 2.0; e = e - 1; end
      end
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
      else if (scaled < midpoint) mant_int = iscaled;
      // verilator coverage_off
      if (mant_int >= 1024) begin biased = biased + 1; mant_int = 0;
        if (biased >= 31) return {s, 5'b11111, 10'b0};
      end
      // verilator coverage_on
      return {s, 5'(biased[4:0]), 10'(mant_int[9:0])};
    end
  endfunction

  // ─── exp / recip LUTs (mirror softmax16) ───────────────────
  // exp_lut[i] = fp16(exp(-i/64)) for i ∈ [0, 1024). Saturates to 0
  // beyond. We always look up exp of a non-positive argument by adding
  // the running max — same trick as softmax16.
  logic [15:0] exp_lut [0:1023];
  logic [15:0] recip_mant_lut [0:1023];

  initial begin : g_luts
    real d, ev, mv, rv;
    int  i;
    for (i = 0; i < 1024; i = i + 1) begin
      d  = real'(i) / 64.0;
      ev = $exp(-d);
      exp_lut[i] = real_to_fp16(ev);
    end
    for (i = 0; i < 1024; i = i + 1) begin
      mv = 1.0 + real'(i) / 1024.0;
      rv = 1.0 / mv;
      recip_mant_lut[i] = real_to_fp16(rv);
    end
  end

  // d (fp16, ≥ 0) → 10-bit index into exp_lut, idx ≈ round(d * 64).
  /* verilator lint_off UNUSEDSIGNAL */
  function automatic logic [9:0] d_to_index(input logic [15:0] d_fp16);
    logic [4:0]  eb;
    logic [9:0]  f;
    int          ev, idx_int;
    logic [10:0] sig11;
    int          shift;
    logic [31:0] scaled, rounded;
    begin
      eb = d_fp16[14:10];
      f  = d_fp16[9:0];
      if (eb == 5'd0)  return 10'd0;
      if (eb == 5'd31) return 10'd1023;
      ev = int'(eb) - 15;
      sig11 = {1'b1, f};
      shift = 6 + ev;
      if (shift > 24) return 10'd1023;
      if (shift < -16) return 10'd0;
      if (shift >= 0) scaled = {21'd0, sig11} << shift;
      else            scaled = {21'd0, sig11} >> (-shift);
      rounded = scaled + 32'd512;
      idx_int = int'(rounded >> 10);
      if (idx_int > 1023) idx_int = 1023;
      if (idx_int < 0)    idx_int = 0;
      return 10'(idx_int);
    end
  endfunction

  function automatic logic [15:0] fp16_exp_neg(input logic [15:0] d_nonneg);
    logic [9:0] idx;
    begin
      // d_nonneg must be ≥ 0 fp16. Returns fp16(exp(-d_nonneg)).
      if (d_nonneg[15]) return 16'h3C00;  // d<0 (numerical noise) → clamp to exp(0)=1
      idx = d_to_index(d_nonneg);
      return exp_lut[idx];
    end
  endfunction

  function automatic logic [15:0] fp16_recip(input logic [15:0] s);
    logic [4:0]  eb;
    logic [9:0]  f;
    logic [15:0] r;
    int          new_biased, r_biased;
    logic [9:0]  r_frac;
    begin
      eb = s[14:10];
      f  = s[9:0];
      if (eb == 5'd0)  return 16'h7BFF;
      if (eb == 5'd31) return 16'h0000;
      r        = recip_mant_lut[f];
      r_biased = int'(r[14:10]);
      r_frac   = r[9:0];
      new_biased = r_biased + 15 - int'(eb);
      if (new_biased >= 31) return {1'b0, 5'b11111, 10'b0};
      if (new_biased <= 0)  return {1'b0, 5'd0, r_frac};
      return {1'b0, 5'(new_biased[4:0]), r_frac};
    end
  endfunction
  /* verilator lint_on UNUSEDSIGNAL */

  // ─── Q/K/V/O memory: write ports ───────────────────────
  always_ff @(posedge clk_i) begin
    if (q_we_i) q_mem[q_waddr_i[Q_AW-1:0]] <= q_wdata_i;
    if (k_we_i) k_mem[k_waddr_i[Q_AW-1:0]] <= k_wdata_i;
    if (v_we_i) v_mem[v_waddr_i[V_AW-1:0]] <= v_wdata_i;
  end

  // Output read (combinational from memory after done).
  assign o_rdata_o = o_mem[o_raddr_i[V_AW-1:0]];

  // ─── Compute state machine ─────────────────────────────
  // FSM: per (head, row i, col j) execute a short fixed micro-sequence.
  // Substates ensure each fp16 op has a single clock to settle into a
  // register before consumers read.
  //
  //   IDLE         : wait start_i
  //   ROW_INIT     : zero O_buf, m_i = MINF, l_i = 0
  //   COL_DOT      : compute s_ij = (Q[i]·K[j]) * TEMP. Pipelined dot
  //                  over DIM_Q lanes via a one-pass adder tree below
  //                  (combinational + 1-cycle register).
  //   COL_SOFT     : m_new = max(m, s); alpha = exp(m - m_new);
  //                  p     = exp(s - m_new). All in one cycle.
  //   COL_LUPD     : l = alpha*l + p (via two FMAs combinationally then
  //                  registered). O[d] = alpha*O[d] + p*V[j][d] for all
  //                  d in parallel.
  //   ROW_FINAL_S  : inv_l = recip(l).
  //   ROW_FINAL_O  : for d = 0..DIM_V-1: o_mem[base+d] = O_buf[d]*inv_l.
  //                  Implemented as DIM_V parallel FMAs (1 cyc) since
  //                  the o_mem is fully addressable; we commit them with
  //                  DIM_V sequential writes (1 cyc each) to keep the
  //                  memory port single-port.
  //   ROW_DONE     : advance to next row / head / DONE.
  //
  // The state lives in `state_q`; counters {head_q, i_q, j_q} loop the
  // (head, row, col) triple.

  typedef enum logic [3:0] {
    S_IDLE,
    S_ROW_INIT,
    S_COL_DOT,
    S_COL_DOT_WAIT,    // 1-cycle settle for the registered dot result
    S_COL_SOFT,
    S_COL_LUPD,
    S_ROW_FINAL_S,
    S_ROW_FINAL_O,
    S_ROW_DONE,
    S_DONE
  } state_e;

  state_e state_q, state_d;

  localparam int HEAD_W = $clog2(HEADS < 2 ? 2 : HEADS);
  localparam int N_W    = $clog2(N    < 2 ? 2 : N);
  localparam int DV_W   = $clog2(DIM_V< 2 ? 2 : DIM_V);

  logic [HEAD_W-1:0] head_q;
  logic [N_W-1:0]    i_q, j_q;
  logic [DV_W-1:0]   dnorm_q;

  // Running state per row
  localparam logic [15:0] FP16_MINF_LIKE = 16'hC800;   // -8.0 — far below any expected s_ij
  localparam logic [15:0] FP16_ZERO      = 16'h0000;
  localparam logic [15:0] FP16_ONE       = 16'h3C00;
  localparam logic [15:0] FP16_TEMP      = real_to_fp16(TEMP);

  logic [15:0] m_q, l_q;
  logic [15:0] s_q;                 // s_ij after dot+temp
  logic [15:0] m_new_q;             // max(m_q, s_q) — registered before LUPD
  logic [15:0] alpha_q, p_q;        // exp(m_q - m_new), exp(s_q - m_new)
  logic [15:0] inv_l_q;             // 1/l_q
  logic [15:0] o_buf [0:DIM_V-1];   // O_i accumulator buffer

  // ─── Q[i] register (loaded at ROW_INIT and held for the row) ──
  logic [15:0] q_row [0:DIM_Q-1];

  // ─── Dot product Q[i] · K[j] : combinational mul tree + adder tree
  // We use behavioral fp16_add / mul (mul via fp16_fma with c=0). To
  // keep this synthesizable we instantiate DIM_Q fp16_fma cells.

  logic [15:0] k_col [0:DIM_Q-1];   // K[j] (combinationally read)
  logic [15:0] mul_out [0:DIM_Q-1]; // Q[i] * K[j] per lane (registered 1 cyc)

  // Read K[j] combinationally from k_mem (head_q, j_q, d)
  always_comb begin
    for (int d = 0; d < DIM_Q; d++) begin
      int unsigned addr;
      addr = int'(head_q) * N * DIM_Q + int'(j_q) * DIM_Q + d;
      k_col[d] = k_mem[addr[Q_AW-1:0]];
    end
  end

  // DIM_Q parallel fp16 multipliers via fp16_fma (a*b + 0).
  genvar gd;
  generate
    (* keep_hierarchy *)
    for (gd = 0; gd < DIM_Q; gd++) begin : g_mul
      fp16_fma u_mul (
        .clk_i  (clk_i),
        .rst_ni (rst_ni),
        .a_i    (q_row[gd]),
        .b_i    (k_col[gd]),
        .c_i    (FP16_ZERO),
        .y_o    (mul_out[gd])
      );
    end
  endgenerate

  // Adder tree over mul_out — purely combinational using fp16_add fn.
  // For DIM_Q up to 64 this is a 6-deep chain of function calls; OK in
  // simulation; PD timing closure may want pipelining, follow-up.
  function automatic logic [15:0] reduce_add(input logic [15:0] arr [DIM_Q]);
    logic [15:0] partial [DIM_Q];
    int n;
    begin
      n = DIM_Q;
      for (int i = 0; i < DIM_Q; i++) partial[i] = arr[i];
      while (n > 1) begin
        int half;
        half = n / 2;
        for (int i = 0; i < half; i++) partial[i] = fp16_add(partial[2*i], partial[2*i+1]);
        if (n % 2 == 1) partial[half] = partial[n - 1];
        n = half + (n % 2);
      end
      return partial[0];
    end
  endfunction

  logic [15:0] dot_sum_comb;
  always_comb dot_sum_comb = reduce_add(mul_out);

  // Apply TEMP scaling — combinational then captured by s_q at COL_DOT_WAIT.
  logic [15:0] s_comb_pre, s_comb;
  always_comb begin
    s_comb_pre = dot_sum_comb;
    // s = dot * TEMP (use fp16_fma combinational? No — use behavioral kernel.
    // Approximate fp16 multiply via add-with-itself isn't general. Use a
    // dedicated registered fp16_fma below.
    s_comb = s_comb_pre;
  end

  // s_ij = dot * TEMP via fp16_fma (registered, 1 cyc latency). We feed
  // dot_sum_comb on COL_DOT cycle; result available on COL_DOT_WAIT.
  logic [15:0] s_fma_out;
  fp16_fma u_s_scale (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (dot_sum_comb),
    .b_i    (FP16_TEMP),
    .c_i    (FP16_ZERO),
    .y_o    (s_fma_out)
  );

  // mul_out is already registered (fp16_fma has 1-cyc latency). The
  // dot_sum_comb is combinational from mul_out. So the timing per col is:
  //   T+0 : COL_DOT — q_row, k_col valid → fp16_mul cells consume them
  //   T+1 : mul_out registered; dot_sum_comb is combinational; u_s_scale
  //         consumes it. Also COL_DOT_WAIT state.
  //   T+2 : s_fma_out registered → captured into s_q on entry of COL_SOFT.

  // ─── Read V[j] combinationally for O update ─────────────────
  logic [15:0] v_col [0:DIM_V-1];
  always_comb begin
    for (int d = 0; d < DIM_V; d++) begin
      int unsigned addr;
      addr = int'(head_q) * N * DIM_V + int'(j_q) * DIM_V + d;
      v_col[d] = v_mem[addr[V_AW-1:0]];
    end
  end

  // ─── DIM_V parallel fp16_fma for O update: alpha * o_buf[d] + p * v[d]
  // We need two FMAs: o_new = alpha*o + p*v. Implement as
  //   tmp[d] = p * v[d]
  //   o_new[d] = alpha * o[d] + tmp[d]
  // Combinationally inside COL_LUPD using a 2-step sequence is awkward.
  // Easier: o_new = fma(alpha, o, fma(p, v, 0)). Each fp16_fma is one
  // cycle; chain → 2 cycles. To keep COL_LUPD a single cycle, we split:
  //   COL_LUPD_A : drive p*v[d] → tmp[d] (DIM_V cells)
  //   COL_LUPD_B : drive alpha*o[d] + tmp[d] (DIM_V cells), result back
  //                into o_buf
  // (Added below; rename original COL_LUPD into two phases.)

  logic [15:0] pv_out  [0:DIM_V-1];   // p * v[d]
  logic [15:0] aob_out [0:DIM_V-1];   // alpha * o[d] + pv

  // Inputs to the two parallel FMA banks. Combinational selectors switch
  // between the LUPD phase (b = v[d], a = p_q) and the FINAL phase
  // (b = inv_l_q, a = o_buf[d], c = 0).
  logic [15:0] fma_a_pv [0:DIM_V-1];
  logic [15:0] fma_b_pv [0:DIM_V-1];
  always_comb begin
    for (int d = 0; d < DIM_V; d++) begin
      fma_a_pv[d] = p_q;
      fma_b_pv[d] = v_col[d];
    end
  end

  generate
    (* keep_hierarchy *)
    for (gd = 0; gd < DIM_V; gd++) begin : g_pv
      fp16_fma u_pv (
        .clk_i  (clk_i),
        .rst_ni (rst_ni),
        .a_i    (fma_a_pv[gd]),
        .b_i    (fma_b_pv[gd]),
        .c_i    (FP16_ZERO),
        .y_o    (pv_out[gd])
      );
    end
  endgenerate

  // Second bank: alpha * o[d] + pv[d]
  generate
    (* keep_hierarchy *)
    for (gd = 0; gd < DIM_V; gd++) begin : g_aob
      fp16_fma u_aob (
        .clk_i  (clk_i),
        .rst_ni (rst_ni),
        .a_i    (alpha_q),
        .b_i    (o_buf[gd]),
        .c_i    (pv_out[gd]),
        .y_o    (aob_out[gd])
      );
    end
  endgenerate

  // ─── l update: l_new = alpha * l + p ────────────────────
  logic [15:0] l_new_out;
  fp16_fma u_l_upd (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (alpha_q),
    .b_i    (l_q),
    .c_i    (p_q),
    .y_o    (l_new_out)
  );

  // ─── Final normalize: o_buf[d] * inv_l ──────────────────
  logic [15:0] norm_out;
  fp16_fma u_norm (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (o_buf[dnorm_q]),
    .b_i    (inv_l_q),
    .c_i    (FP16_ZERO),
    .y_o    (norm_out)
  );

  // ─── State machine ──────────────────────────────────────
  // dnorm_q advances during ROW_FINAL_O (writing one DIM_V word per cyc).
  // Two-stage pipe for normalize: launch FMA on dnorm_q, write result of
  // dnorm_q-2 (pipeline depth: 2 cycles — 1 for FMA latency, 1 to commit
  // to o_mem). We track this with a small "write_pending" shift register.

  logic [DV_W-1:0] norm_widx_q;   // index for the pending write
  logic        norm_widx_v_q;     // valid pending write

  // Compute targets for the LUPD A and B substates:
  //   LUPD_A: drive p_q, v_col → pv_out latched into pv_q at end of cycle
  //           (we use pv_out directly since it's registered by fp16_fma).
  //   LUPD_B: drive alpha_q, o_buf[d], pv_q[d] → aob_out
  //           latched into o_buf at end of cycle. Also l_new latched in.
  //
  // We split COL_LUPD into S_COL_LUPD_A and S_COL_LUPD_B.

  // Replace COL_LUPD state with two phases.
  // Re-declare extended enum: redo the enum to add A/B substates.

  // (We can't redo the enum already declared — instead, in the comb
  // block we'll subdivide S_COL_LUPD using a 1-bit phase counter.)
  logic [1:0] lupd_phase_q;   // 0: settle p/alpha; 1: launch pv; 2: launch aob/l; 3: commit

  // ─── done/busy ──────────────────────────────────────────
  assign busy_o = (state_q != S_IDLE) && (state_q != S_DONE);
  assign done_o = (state_q == S_DONE);

  // ─── Next-state and datapath updates ────────────────────
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q     <= S_IDLE;
      head_q      <= '0;
      i_q         <= '0;
      j_q         <= '0;
      dnorm_q     <= '0;
      m_q         <= FP16_MINF_LIKE;
      l_q         <= FP16_ZERO;
      s_q         <= FP16_ZERO;
      m_new_q     <= FP16_MINF_LIKE;
      alpha_q     <= FP16_ZERO;
      p_q         <= FP16_ZERO;
      inv_l_q     <= FP16_ZERO;
      lupd_phase_q<= 2'd0;
      norm_widx_q <= '0;
      norm_widx_v_q <= 1'b0;
      for (int d = 0; d < DIM_V; d++) o_buf[d] <= FP16_ZERO;
      for (int d = 0; d < DIM_Q; d++) q_row[d] <= FP16_ZERO;
    end else begin
      // Default: hold.
      case (state_q)
        // ── IDLE: wait for start ──
        S_IDLE: begin
          if (start_i) begin
            head_q  <= '0;
            i_q     <= '0;
            state_q <= S_ROW_INIT;
          end
        end

        // ── ROW_INIT: load Q[i] register, zero O, m, l ──
        S_ROW_INIT: begin
          // Load Q[i] for current (head, i)
          for (int d = 0; d < DIM_Q; d++) begin
            int unsigned addr;
            addr = int'(head_q) * N * DIM_Q + int'(i_q) * DIM_Q + d;
            q_row[d] <= q_mem[addr[Q_AW-1:0]];
          end
          for (int d = 0; d < DIM_V; d++) o_buf[d] <= FP16_ZERO;
          m_q <= FP16_MINF_LIKE;
          l_q <= FP16_ZERO;
          j_q <= '0;
          state_q <= S_COL_DOT;
        end

        // ── COL_DOT: launch Q[i]·K[j] multiplies ──
        // Mul cells consume q_row + k_col combinationally; one cycle later
        // mul_out is registered.
        S_COL_DOT: begin
          state_q <= S_COL_DOT_WAIT;
        end

        // ── COL_DOT_WAIT: mul_out registered; s_fma launched ──
        // s_fma_out will be registered next cycle.
        S_COL_DOT_WAIT: begin
          state_q <= S_COL_SOFT;
        end

        // ── COL_SOFT: s_q known; compute m_new, alpha, p ──
        S_COL_SOFT: begin
          logic [15:0] s_local;
          logic [15:0] m_new_local;
          logic [15:0] d_alpha, d_p;
          s_local     = s_fma_out;
          s_q         <= s_local;
          m_new_local = fp16_max(m_q, s_local);
          m_new_q     <= m_new_local;
          // alpha = exp(m_q - m_new_local); m_q ≤ m_new ⇒ arg ≤ 0 ⇒ |arg| ≥ 0.
          // exp(-x), x = m_new_local - m_q ≥ 0.
          d_alpha     = fp16_add(m_new_local, fp16_neg(m_q));
          alpha_q     <= fp16_exp_neg(d_alpha);
          // p = exp(s_local - m_new_local) ; s_local ≤ m_new ⇒ exp(-(m_new - s))
          d_p         = fp16_add(m_new_local, fp16_neg(s_local));
          p_q         <= fp16_exp_neg(d_p);
          lupd_phase_q <= 2'd0;
          state_q     <= S_COL_LUPD;
        end

        // ── COL_LUPD: two phases.
        //   phase 0 (A): launch pv (p * v[d]) — DIM_V parallel cells
        //   phase 1 (B): launch aob (alpha*o[d] + pv[d]) — DIM_V parallel
        //                cells. Also launch l_upd. Result available
        //                next cycle. Then advance j (or move on).
        S_COL_LUPD: begin
          // Phase 0: p_q / alpha_q have just been registered (at the
          //          edge entering this state). u_pv FMA reads them as
          //          its inputs this cycle → pv_out registers next cyc.
          // Phase 1: pv_out is now valid. u_aob FMA reads alpha_q,
          //          o_buf[d], pv_out → aob_out registers next cyc.
          //          Also launch u_l_upd (alpha,l,p) → l_new_out next.
          // Phase 2: aob_out and l_new_out are valid. Commit to o_buf,
          //          l_q, m_q. Advance j or finalize.
          case (lupd_phase_q)
            2'd0: lupd_phase_q <= 2'd1;
            2'd1: lupd_phase_q <= 2'd2;
            2'd2: begin
              for (int d = 0; d < DIM_V; d++) o_buf[d] <= aob_out[d];
              l_q     <= l_new_out;
              m_q     <= m_new_q;
              lupd_phase_q <= 2'd0;
              if (j_q == N_W'(N-1)) begin
                j_q     <= '0;
                state_q <= S_ROW_FINAL_S;
              end else begin
                j_q     <= j_q + 1'b1;
                state_q <= S_COL_DOT;
              end
            end
            default: lupd_phase_q <= 2'd0;
          endcase
        end

        // ── ROW_FINAL_S: compute inv_l = 1/l ──
        S_ROW_FINAL_S: begin
          inv_l_q <= fp16_recip(l_q);
          dnorm_q <= '0;
          norm_widx_v_q <= 1'b0;
          state_q <= S_ROW_FINAL_O;
        end

        // ── ROW_FINAL_O: stream DIM_V multiplies & write o_mem ──
        // Each cycle: launch FMA on o_buf[dnorm_q]; FMA result (norm_out)
        // is registered next cycle. We use a 1-deep shift register
        // {norm_widx_v_q, norm_widx_q} to know which o_mem index to write
        // with norm_out.
        S_ROW_FINAL_O: begin
          // Commit pending write from previous cycle's FMA launch.
          if (norm_widx_v_q) begin
            int unsigned waddr;
            waddr = int'(head_q) * N * DIM_V + int'(i_q) * DIM_V + int'(norm_widx_q);
            o_mem[waddr[V_AW-1:0]] <= norm_out;
          end
          // Launch new FMA on o_buf[dnorm_q] this cycle (its result is
          // registered next cycle, then committed the cycle after).
          norm_widx_q   <= dnorm_q;
          norm_widx_v_q <= 1'b1;
          if (int'(dnorm_q) == DIM_V - 1) begin
            // Just launched the last FMA. One more cycle drains it.
            state_q <= S_ROW_DONE;
          end else begin
            dnorm_q <= dnorm_q + 1'b1;
          end
        end

        // ── ROW_DONE: drain final FMA, then advance row/head ──
        S_ROW_DONE: begin
          // Commit final write
          if (norm_widx_v_q) begin
            int unsigned waddr;
            waddr = int'(head_q) * N * DIM_V + int'(i_q) * DIM_V + int'(norm_widx_q);
            o_mem[waddr[V_AW-1:0]] <= norm_out;
          end
          norm_widx_v_q <= 1'b0;
          // Advance i, head
          if (i_q == N_W'(N-1)) begin
            i_q <= '0;
            if (head_q == HEAD_W'(HEADS-1)) begin
              state_q <= S_DONE;
            end else begin
              head_q  <= head_q + 1'b1;
              state_q <= S_ROW_INIT;
            end
          end else begin
            i_q     <= i_q + 1'b1;
            state_q <= S_ROW_INIT;
          end
        end

        // ── DONE: assert done; return to IDLE on next start ──
        S_DONE: begin
          if (start_i) begin
            head_q  <= '0;
            i_q     <= '0;
            state_q <= S_ROW_INIT;
          end
        end

        default: state_q <= S_IDLE;
      endcase
    end
  end

  // Tie unused
  // verilator lint_off UNUSEDSIGNAL
  wire _unused = ^{state_d, s_comb, s_q, m_new_q, 1'b0};
  // verilator lint_on UNUSEDSIGNAL
  assign state_d = state_q;  // not used; just placeholder

endmodule
// verilator lint_on UNUSEDPARAM
// verilator lint_on UNUSEDSIGNAL
