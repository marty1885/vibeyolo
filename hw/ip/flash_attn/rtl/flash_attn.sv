// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// flash_attn — forward-only multi-head flash attention leaf IP.
//
//   O[h][r][d] = sum_j softmax(Q[h][r] · K[h][j] * TEMP)[j] * V[h][j][d]
//
// Online tiled softmax (no N×N materialisation). Row-tile BR, col-tile
// BC. The BR*BC fp16_fma cells are reused for both QK^T and PV. The
// inner reduction dims (DIM_Q and BC) are TIME-MULTIPLEXED so the cell
// count is BR*BC, not BR*BC*DIM_Q.
//
// fp16_fma in this repo has 1-cycle pipeline latency (single output
// register). Sequential accumulation `c_i ← y_o` runs back-to-back
// without stall.
//
// Default parameters: BR=4, BC=4 (test-friendly). Production override
// at instantiation, e.g. BR=16 BC=32 for HEADS=2 N=400 DIM_Q=32 DIM_V=64.
//
// Cell counts at BR=16, BC=32:
//   score/PV fp16_fma array : BR*BC = 512
//   LUPD per-row fp16_fma   : BR    = 16
//   norm reuses score array : 0 extra
//   TOTAL fp16_fma cells    : 528   ≤ 1500 budget
//
// State machine (one schedule per (head, row-tile)):
//
//   foreach head h:
//     foreach row-tile br:
//       init m_r=-∞ l_r=0 O_acc=0; first_col=1
//       foreach col-tile bc:
//         DOT_K  : DIM_Q cycles — cell(r,j) ← cell(r,j) + Q[r,k]*K[j,k]
//         SCALE  : 1 cycle      — cell(r,j) ← TEMP * cell(r,j)
//         WAIT0  : 1 cycle      — let SCALE's result settle into cell_y
//         RMAX_S : 1 cycle      — sample row_max_r from cell_y
//         MNEW   : 1 cycle      — register m_new_r, alpha_r
//         EXP_D  : 1 cycle      — drive cells with exp(s-m_new)
//         EXP_S  : 1 cycle      — sample P_reg from cell_y; row_sum_r
//         LUPD_D : 1 cycle      — drive lupd fmas
//         LUPD_S : 1 cycle      — sample l_r ← lupd_y
//         foreach dv_chunk:
//           PV_K   : BC cycles  — cell(r,c) ← cell(r,c) + P[r,j]*V[j,d]
//           PV_LAT : 1 cycle    — sample PV_part_reg from cell_y
//           MERGE_D: 1 cycle    — drive cells: alpha*O_old + PV_part
//           MERGE_S: 1 cycle    — sample O_acc ← cell_y
//         m_r ← m_new_r; first_col ← 0
//       foreach norm_chunk:
//         NORM_D : 1 cycle      — drive cells: O_acc*inv_l
//         NORM_S : 1 cycle      — sample O_out ← cell_y
//   raise done_o
//
// Per-col-tile cycle cost (BR=16, BC=32, DIM_Q=32, DIM_V=64):
//   DOT_K   DIM_Q     = 32
//   SCALE+WAIT0       = 2
//   RMAX_S            = 1
//   MNEW              = 1
//   EXP_D + EXP_S     = 2
//   LUPD_D + LUPD_S   = 2
//   PV per chunk      = BC + 1 + 1 + 1 = BC + 3
//   PV total          = N_DV_CHUNK * (BC + 3) = 2 * 35 = 70
//   total per col-tile: 32+2+1+1+2+2+70 = 110 cycles
// Total col-tiles: HEADS * ceil(N/BR) * ceil(N/BC) = 2 * 25 * 13 = 650.
//   col-tile cycles  = 650 * 110 = 71_500
//   norm cycles      = HEADS * ceil(N/BR) * N_DV_CHUNK * 2
//                    = 2 * 25 * 2 * 2 = 200
//   TOTAL          ≈ 71_700 cycles — under 100k budget. ✔

module flash_attn #(
  parameter int HEADS = 1,
  parameter int N     = 8,
  parameter int DIM_Q = 4,
  parameter int DIM_V = 4,
  parameter int BR    = 4,
  parameter int BC    = 4,
  // Softmax temperature, fp16. Default 1/sqrt(32) ≈ 0.17677669 = 0x31A8.
  parameter logic [15:0] TEMP_FP16 = 16'h31A8,
  // Hint to TB / SoC about expected cycle budget.
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

  // ───────────────────────── derived ───────────────────────────
  localparam int N_BR        = (N + BR - 1) / BR;
  localparam int N_BC        = (N + BC - 1) / BC;
  localparam int N_DV_CHUNK  = (DIM_V + BC - 1) / BC;
  localparam logic [15:0] FP16_ONE     = 16'h3C00;
  localparam logic [15:0] FP16_ZERO    = 16'h0000;
  localparam logic [15:0] FP16_NEG_BIG = 16'hFBFF;  // very negative finite

  // ───────────────────────── tensor unpack ─────────────────────
  logic [15:0] Q [HEADS][N][DIM_Q];
  logic [15:0] K [HEADS][N][DIM_Q];
  logic [15:0] V [HEADS][N][DIM_V];
  logic [15:0] O_out [HEADS][N][DIM_V];

  always_comb begin
    for (int h = 0; h < HEADS; h++) begin
      for (int r = 0; r < N; r++) begin
        for (int k = 0; k < DIM_Q; k++) begin
          Q[h][r][k] = q_flat_i[16*(h*N*DIM_Q + r*DIM_Q + k) +: 16];
          K[h][r][k] = k_flat_i[16*(h*N*DIM_Q + r*DIM_Q + k) +: 16];
        end
        for (int d = 0; d < DIM_V; d++)
          V[h][r][d] = v_flat_i[16*(h*N*DIM_V + r*DIM_V + d) +: 16];
      end
    end
  end

  always_comb begin
    o_flat_o = '0;
    for (int h = 0; h < HEADS; h++)
      for (int r = 0; r < N; r++)
        for (int d = 0; d < DIM_V; d++)
          o_flat_o[16*(h*N*DIM_V + r*DIM_V + d) +: 16] = O_out[h][r][d];
  end

  // ───────────────────────── fp16 helpers ──────────────────────
  function automatic logic fp16_gt(input logic [15:0] a, input logic [15:0] b);
    logic sa, sb, az, bz;
    logic [14:0] ma, mb;
    begin
      sa = a[15]; sb = b[15];
      az = (a[14:0] == 15'd0); bz = (b[14:0] == 15'd0);
      ma = a[14:0]; mb = b[14:0];
      if (az && bz)        fp16_gt = 1'b0;
      else if (sa != sb)   fp16_gt = !sa;
      else if (sa == 1'b0) fp16_gt = (ma > mb);
      else                 fp16_gt = (ma < mb);
    end
  endfunction
  function automatic logic [15:0] fp16_max(input logic [15:0] a, input logic [15:0] b);
    fp16_max = fp16_gt(a, b) ? a : b;
  endfunction
  function automatic logic [15:0] fp16_neg(input logic [15:0] a);
    fp16_neg = {~a[15], a[14:0]};
  endfunction
  function automatic logic [15:0] fp16_add(input logic [15:0] a, input logic [15:0] b);
    logic        sa, sb;
    logic [4:0]  ea_b, eb_b;
    logic [9:0]  fa, fb;
    logic        a_z, b_z, a_inf, b_inf, a_nan, b_nan;
    logic [10:0] sig_a, sig_b;
    int          ea, eb, anchor, sh, msb, shift10, biased_exp, eshift;
    logic [31:0] big_a, big_b, sum_mag, tmp;
    logic        sticky, result_sign, guard, round_b, st;
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
      else           begin sig_a = {1'b1, fa}; ea = int'(ea_b) - 25; end
      if (eb_b == 0) begin sig_b = {1'b0, fb}; eb = -24; end
      else           begin sig_b = {1'b1, fb}; eb = int'(eb_b) - 25; end
      anchor = (ea < eb) ? ea : eb;
      sticky = 1'b0;
      sh = ea - anchor;
      if (sh > 20) begin tmp = 32'd0; sticky = 1'b1; big_a = tmp; end
      else             big_a = {21'd0, sig_a} << sh;
      sh = eb - anchor;
      if (sh > 20) begin tmp = 32'd0; sticky = 1'b1; big_b = tmp; end
      else             big_b = {21'd0, sig_b} << sh;
      if (sa == sb) begin sum_mag = big_a + big_b; result_sign = sa; end
      else begin
        if (big_a >= big_b) begin sum_mag = big_a - big_b; result_sign = sa; end
        else                begin sum_mag = big_b - big_a; result_sign = sb; end
      end
      if (sum_mag == 32'd0 && !sticky) return 16'h0000;
      msb = 0;
      for (int i = 31; i >= 0; i--) if (sum_mag[i] && (msb == 0)) msb = i;
      biased_exp = anchor + msb + 15;
      eshift     = msb - 10;
      if (biased_exp >= 31) return {result_sign, 5'b11111, 10'b0};
      if (biased_exp <= 0) begin eshift = eshift + (1 - biased_exp); biased_exp = 0; end
      shift10 = eshift;
      if (shift10 < 0) begin
        tmp = sum_mag << (-shift10);
        mant10 = tmp[9:0]; guard = 1'b0; round_b = 1'b0; st = sticky;
      end else if (shift10 == 0) begin
        mant10 = sum_mag[9:0]; guard = 1'b0; round_b = 1'b0; st = sticky;
      end else if (shift10 == 1) begin
        mant10 = sum_mag[10:1]; guard = sum_mag[0]; round_b = 1'b0; st = sticky;
      end else if (shift10 == 2) begin
        mant10 = sum_mag[11:2]; guard = sum_mag[1]; round_b = sum_mag[0]; st = sticky;
      end else begin
        tmp = sum_mag >> shift10;        mant10  = tmp[9:0];
        tmp = sum_mag >> (shift10 - 1);  guard   = tmp[0];
        tmp = sum_mag >> (shift10 - 2);  round_b = tmp[0];
        st  = sticky;
        for (int j = 0; j < 32; j++)
          if (j < shift10 - 2 && sum_mag[j]) st = 1'b1;
      end
      mant_r = {1'b0, mant10} + 11'((guard && ((round_b | st) || mant10[0])) ? 1 : 0);
      if (mant_r[10]) begin biased_exp = biased_exp + 1; mant_r = 11'b0; end
      if (biased_exp >= 31) return {result_sign, 5'b11111, 10'b0};
      if (biased_exp <= 0)  return {result_sign, 5'd0, mant_r[9:0]};
      return {result_sign, 5'(biased_exp), mant_r[9:0]};
    end
  endfunction
  function automatic logic [15:0] fp16_sub(input logic [15:0] a, input logic [15:0] b);
    fp16_sub = fp16_add(a, fp16_neg(b));
  endfunction

  // ───────────────────────── LUTs ──────────────────────────────
  logic [15:0] exp_lut       [0:1023];
  logic [15:0] recip_mant_lut [0:1023];

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

  initial begin : g_luts
    real d, ev, mv, rv;
    for (int i = 0; i < 1024; i++) begin
      d = real'(i) / 64.0; ev = $exp(-d);
      exp_lut[i] = real_to_fp16(ev);
    end
    for (int i = 0; i < 1024; i++) begin
      mv = 1.0 + real'(i) / 1024.0; rv = 1.0 / mv;
      recip_mant_lut[i] = real_to_fp16(rv);
    end
  end

  function automatic logic [9:0] d_to_index(input logic [15:0] d_fp16);
    logic [4:0]  eb;
    logic [9:0]  f;
    int          ev, shift, idx_int;
    logic [10:0] sig11;
    logic [31:0] scaled, rounded;
    begin
      eb = d_fp16[14:10]; f = d_fp16[9:0];
      if (eb == 5'd0)  return 10'd0;
      if (eb == 5'd31) return 10'd1023;
      ev = int'(eb) - 15;
      sig11 = {1'b1, f};
      shift = 6 + ev;
      if (shift > 24)  return 10'd1023;
      if (shift < -16) return 10'd0;
      if (shift >= 0)  scaled = {21'd0, sig11} << shift;
      else             scaled = {21'd0, sig11} >> (-shift);
      rounded = scaled + 32'd512;
      idx_int = int'(rounded >> 10);
      if (idx_int > 1023) idx_int = 1023;
      if (idx_int < 0)    idx_int = 0;
      return 10'(idx_int);
    end
  endfunction

  function automatic logic [15:0] fp16_exp_negative(input logic [15:0] d_pos);
    logic [9:0]  idx;
    logic [15:0] dd;
    begin
      dd = d_pos;
      if (dd[15]) dd = 16'h0000;
      idx = d_to_index(dd);
      return exp_lut[idx];
    end
  endfunction

  function automatic logic [15:0] fp16_recip(input logic [15:0] s);
    logic [4:0] eb;
    logic [9:0] f, r_frac;
    logic [15:0] r;
    int         new_biased, r_biased;
    begin
      eb = s[14:10]; f = s[9:0];
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

  // ───────────────────────── FMA array ─────────────────────────
  logic [15:0] cell_a [BR][BC];
  logic [15:0] cell_b [BR][BC];
  logic [15:0] cell_c [BR][BC];
  logic [15:0] cell_y [BR][BC];

  generate
    for (genvar gr = 0; gr < BR; gr++) begin : g_row
      for (genvar gc = 0; gc < BC; gc++) begin : g_col
        fp16_fma u_fma (
          .clk_i  (clk_i),
          .rst_ni (rst_ni),
          .a_i    (cell_a[gr][gc]),
          .b_i    (cell_b[gr][gc]),
          .c_i    (cell_c[gr][gc]),
          .y_o    (cell_y[gr][gc])
        );
      end
    end
  endgenerate

  // Per-row LUPD fmas
  logic [15:0] lupd_a [BR];
  logic [15:0] lupd_b [BR];
  logic [15:0] lupd_c [BR];
  logic [15:0] lupd_y [BR];

  generate
    for (genvar gr = 0; gr < BR; gr++) begin : g_lupd
      fp16_fma u_lupd (
        .clk_i  (clk_i),
        .rst_ni (rst_ni),
        .a_i    (lupd_a[gr]),
        .b_i    (lupd_b[gr]),
        .c_i    (lupd_c[gr]),
        .y_o    (lupd_y[gr])
      );
    end
  endgenerate

  // ───────────────────────── state ─────────────────────────────
  typedef enum logic [4:0] {
    S_IDLE,
    S_DOT_K,
    S_SCALE,
    S_WAIT0,
    S_RMAX_S,
    S_MNEW,
    S_EXP_D,
    S_EXP_S,
    S_LUPD_D,
    S_LUPD_S,
    S_PV_K,
    S_PV_LAT,
    S_MERGE_D,
    S_MERGE_S,
    S_NORM_D,
    S_NORM_S,
    S_DONE
  } state_e;

  state_e state_q;

  int unsigned head_q, br_q, bc_q, kq_q;
  int unsigned pv_j_q, dv_chunk_q;
  int unsigned norm_chunk_q;
  logic first_col;

  // Per-row scalars
  logic [15:0] m_r       [BR];
  logic [15:0] l_r       [BR];
  logic [15:0] alpha_r   [BR];
  logic [15:0] row_max_r [BR];
  logic [15:0] m_new_r   [BR];
  logic [15:0] row_sum_r [BR];
  // Per-row × DIM_V running unnormalised output
  logic [15:0] O_acc     [BR][DIM_V];
  // Latches between FMA stages
  logic [15:0] P_reg       [BR][BC];
  logic [15:0] PV_part_reg [BR][BC];

  // ───────────────────────── reductions (combinational) ────────
  logic [15:0] row_max_comb [BR];
  logic [15:0] row_sum_comb [BR];
  logic [15:0] inv_l_r      [BR];

  always_comb begin
    for (int r = 0; r < BR; r++) begin
      logic [15:0] acc_m, acc_s;
      acc_m = cell_y[r][0];
      acc_s = cell_y[r][0];
      for (int j = 1; j < BC; j++) begin
        acc_m = fp16_max(acc_m, cell_y[r][j]);
        acc_s = fp16_add(acc_s, cell_y[r][j]);
      end
      row_max_comb[r] = acc_m;
      row_sum_comb[r] = acc_s;
    end
    for (int r = 0; r < BR; r++) inv_l_r[r] = fp16_recip(l_r[r]);
  end

  // ───────────────────────── helpers ───────────────────────────
  function automatic int row_idx(input int r); row_idx = int'(br_q) * BR + r; endfunction
  function automatic int col_idx(input int c); col_idx = int'(bc_q) * BC + c; endfunction

  // ───────────────────────── FMA drivers (comb) ────────────────
  // Default drive is the IDENTITY operation: a*1+0 = a. This keeps
  // cell_y unchanged from cycle to cycle in "hold" states. Producing
  // states override with their own a/b/c.
  always_comb begin
    for (int r = 0; r < BR; r++) begin
      for (int c = 0; c < BC; c++) begin
        cell_a[r][c] = cell_y[r][c];
        cell_b[r][c] = FP16_ONE;
        cell_c[r][c] = FP16_ZERO;
      end
    end
    case (state_q)
      S_DOT_K: begin
        for (int r = 0; r < BR; r++) begin
          for (int j = 0; j < BC; j++) begin
            int r_g_l, j_g_l;
            r_g_l = row_idx(r);
            j_g_l = col_idx(j);
            cell_a[r][j] = (r_g_l < N && j_g_l < N && int'(kq_q) < DIM_Q)
                            ? Q[head_q][r_g_l][kq_q] : FP16_ZERO;
            cell_b[r][j] = (r_g_l < N && j_g_l < N && int'(kq_q) < DIM_Q)
                            ? K[head_q][j_g_l][kq_q] : FP16_ZERO;
            cell_c[r][j] = (kq_q == 0) ? FP16_ZERO : cell_y[r][j];
          end
        end
      end
      S_SCALE: begin
        for (int r = 0; r < BR; r++)
          for (int j = 0; j < BC; j++) begin
            cell_a[r][j] = cell_y[r][j];
            cell_b[r][j] = TEMP_FP16;
            cell_c[r][j] = FP16_ZERO;
          end
      end
      S_EXP_D: begin
        for (int r = 0; r < BR; r++) begin
          for (int j = 0; j < BC; j++) begin
            logic [15:0] d_l;
            d_l = fp16_sub(m_new_r[r], cell_y[r][j]);
            cell_a[r][j] = fp16_exp_negative(d_l);
            cell_b[r][j] = FP16_ONE;
            cell_c[r][j] = FP16_ZERO;
          end
        end
      end
      S_PV_K: begin
        for (int r = 0; r < BR; r++) begin
          for (int c = 0; c < BC; c++) begin
            int j_g_l, d_idx_l;
            j_g_l   = col_idx(pv_j_q);
            d_idx_l = int'(dv_chunk_q) * BC + c;
            cell_a[r][c] = (j_g_l < N && d_idx_l < DIM_V)
                            ? P_reg[r][pv_j_q] : FP16_ZERO;
            cell_b[r][c] = (j_g_l < N && d_idx_l < DIM_V)
                            ? V[head_q][j_g_l][d_idx_l] : FP16_ZERO;
            cell_c[r][c] = (pv_j_q == 0) ? FP16_ZERO : cell_y[r][c];
          end
        end
      end
      S_MERGE_D: begin
        for (int r = 0; r < BR; r++) begin
          for (int c = 0; c < BC; c++) begin
            int d_idx_l;
            d_idx_l = int'(dv_chunk_q) * BC + c;
            cell_a[r][c] = first_col ? PV_part_reg[r][c] : alpha_r[r];
            cell_b[r][c] = first_col
                            ? FP16_ONE
                            : ((d_idx_l < DIM_V) ? O_acc[r][d_idx_l] : FP16_ZERO);
            cell_c[r][c] = first_col ? FP16_ZERO : PV_part_reg[r][c];
          end
        end
      end
      S_NORM_D: begin
        for (int r = 0; r < BR; r++) begin
          for (int c = 0; c < BC; c++) begin
            int d_idx_l;
            d_idx_l = int'(norm_chunk_q) * BC + c;
            cell_a[r][c] = (d_idx_l < DIM_V) ? O_acc[r][d_idx_l] : FP16_ZERO;
            cell_b[r][c] = inv_l_r[r];
            cell_c[r][c] = FP16_ZERO;
          end
        end
      end
      default: ;
    endcase
  end

  // LUPD driver
  always_comb begin
    for (int r = 0; r < BR; r++) begin
      if (first_col) begin
        lupd_a[r] = row_sum_r[r];
        lupd_b[r] = FP16_ONE;
        lupd_c[r] = FP16_ZERO;
      end else begin
        lupd_a[r] = alpha_r[r];
        lupd_b[r] = l_r[r];
        lupd_c[r] = row_sum_r[r];
      end
    end
  end

  // ───────────────────────── sequential ────────────────────────
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q       <= S_IDLE;
      head_q        <= 0;
      br_q          <= 0;
      bc_q          <= 0;
      kq_q          <= 0;
      pv_j_q        <= 0;
      dv_chunk_q    <= 0;
      norm_chunk_q  <= 0;
      first_col     <= 1'b1;
      done_o        <= 1'b0;
      for (int r = 0; r < BR; r++) begin
        m_r[r]       <= FP16_NEG_BIG;
        l_r[r]       <= FP16_ZERO;
        alpha_r[r]   <= FP16_ZERO;
        row_max_r[r] <= FP16_ZERO;
        m_new_r[r]   <= FP16_ZERO;
        row_sum_r[r] <= FP16_ZERO;
        for (int d = 0; d < DIM_V; d++) O_acc[r][d] <= FP16_ZERO;
        for (int c = 0; c < BC; c++) begin
          P_reg[r][c]       <= FP16_ZERO;
          PV_part_reg[r][c] <= FP16_ZERO;
        end
      end
      for (int h = 0; h < HEADS; h++)
        for (int rr = 0; rr < N; rr++)
          for (int d = 0; d < DIM_V; d++) O_out[h][rr][d] <= FP16_ZERO;
    end else begin
      done_o <= 1'b0;

      case (state_q)

        S_IDLE: begin
          if (start_i) begin
            head_q     <= 0;
            br_q       <= 0;
            bc_q       <= 0;
            kq_q       <= 0;
            first_col  <= 1'b1;
            for (int r = 0; r < BR; r++) begin
              m_r[r] <= FP16_NEG_BIG;
              l_r[r] <= FP16_ZERO;
              for (int d = 0; d < DIM_V; d++) O_acc[r][d] <= FP16_ZERO;
            end
            state_q <= S_DOT_K;
          end
        end

        S_DOT_K: begin
          if (kq_q == DIM_Q - 1) begin
            state_q <= S_SCALE;
          end
          kq_q <= kq_q + 1;
        end

        S_SCALE: begin
          state_q <= S_WAIT0;
        end

        // S_WAIT0: gives the SCALE fma a cycle to land in cell_y.
        S_WAIT0: begin
          state_q <= S_RMAX_S;
        end

        S_RMAX_S: begin
          for (int r = 0; r < BR; r++) row_max_r[r] <= row_max_comb[r];
          state_q <= S_MNEW;
        end

        S_MNEW: begin
          for (int r = 0; r < BR; r++) begin
            logic [15:0] mn;
            mn = first_col ? row_max_r[r] : fp16_max(m_r[r], row_max_r[r]);
            m_new_r[r] <= mn;
            // alpha = exp(m_old - m_new). Since m_new ≥ m_old, the
            // argument is ≤ 0. fp16_exp_negative(x) = exp(-x) for x ≥ 0,
            // so we pass (m_new - m_old) ≥ 0 to get exp(-(m_new-m_old))
            // = exp(m_old - m_new). Correct.
            alpha_r[r] <= first_col
              ? FP16_ZERO
              : fp16_exp_negative(fp16_sub(mn, m_r[r]));
          end
          state_q <= S_EXP_D;
        end

        S_EXP_D: begin
          // cell_y still holds S (scaled scores) — used by driver this cycle.
          // We need to sample S now before EXP overwrites it next edge.
          // Driver above used cell_y for the sub; that uses CURRENT cell_y
          // which is S. cell_a/b/c set to (P_lut, 1, 0). Next edge cell_y=P.
          state_q <= S_EXP_S;
        end

        S_EXP_S: begin
          // cell_y now holds P. Sample into P_reg and row_sum.
          for (int r = 0; r < BR; r++)
            for (int c = 0; c < BC; c++) P_reg[r][c] <= cell_y[r][c];
          for (int r = 0; r < BR; r++) row_sum_r[r] <= row_sum_comb[r];
          state_q <= S_LUPD_D;
        end

        S_LUPD_D: begin
          // Drives lupd inputs (combinational); next edge lupd_y updates.
          state_q <= S_LUPD_S;
        end

        S_LUPD_S: begin
          for (int r = 0; r < BR; r++) l_r[r] <= lupd_y[r];
          pv_j_q     <= 0;
          dv_chunk_q <= 0;
          state_q    <= S_PV_K;
        end

        S_PV_K: begin
          if (pv_j_q == BC - 1) begin
            state_q <= S_PV_LAT;
          end
          pv_j_q <= pv_j_q + 1;
        end

        S_PV_LAT: begin
          // Sample PV partials from cell_y (just produced by last PV_K).
          for (int r = 0; r < BR; r++)
            for (int c = 0; c < BC; c++) PV_part_reg[r][c] <= cell_y[r][c];
          state_q <= S_MERGE_D;
        end

        S_MERGE_D: begin
          state_q <= S_MERGE_S;
        end

        S_MERGE_S: begin
          for (int r = 0; r < BR; r++) begin
            for (int c = 0; c < BC; c++) begin
              int d_idx;
              d_idx = int'(dv_chunk_q) * BC + c;
              if (d_idx < DIM_V) O_acc[r][d_idx] <= cell_y[r][c];
            end
          end
          if (dv_chunk_q == N_DV_CHUNK - 1) begin
            // Done all DV chunks for this col-tile.
            for (int r = 0; r < BR; r++) m_r[r] <= m_new_r[r];
            if (bc_q == N_BC - 1) begin
              // Last col-tile of this row-tile → normalise.
              bc_q         <= 0;
              first_col    <= 1'b1;
              kq_q         <= 0;
              norm_chunk_q <= 0;
              state_q      <= S_NORM_D;
            end else begin
              bc_q      <= bc_q + 1;
              first_col <= 1'b0;
              kq_q      <= 0;
              pv_j_q    <= 0;
              dv_chunk_q <= 0;
              state_q   <= S_DOT_K;
            end
          end else begin
            dv_chunk_q <= dv_chunk_q + 1;
            pv_j_q     <= 0;
            state_q    <= S_PV_K;
          end
        end

        S_NORM_D: begin
          state_q <= S_NORM_S;
        end

        S_NORM_S: begin
          for (int r = 0; r < BR; r++) begin
            for (int c = 0; c < BC; c++) begin
              int d_idx, r_g;
              d_idx = int'(norm_chunk_q) * BC + c;
              r_g   = int'(br_q) * BR + r;
              if (d_idx < DIM_V && r_g < N) begin
                O_out[head_q][r_g][d_idx] <= cell_y[r][c];
              end
            end
          end
          if (norm_chunk_q == N_DV_CHUNK - 1) begin
            // Done normalising this row-tile. Reset row state & advance.
            for (int r = 0; r < BR; r++) begin
              m_r[r] <= FP16_NEG_BIG;
              l_r[r] <= FP16_ZERO;
              for (int d = 0; d < DIM_V; d++) O_acc[r][d] <= FP16_ZERO;
            end
            if (br_q == N_BR - 1) begin
              br_q      <= 0;
              first_col <= 1'b1;
              kq_q      <= 0;
              if (head_q == HEADS - 1) begin
                state_q <= S_DONE;
              end else begin
                head_q  <= head_q + 1;
                state_q <= S_DOT_K;
              end
            end else begin
              br_q      <= br_q + 1;
              first_col <= 1'b1;
              kq_q      <= 0;
              state_q   <= S_DOT_K;
            end
          end else begin
            norm_chunk_q <= norm_chunk_q + 1;
            state_q      <= S_NORM_D;
          end
        end

        S_DONE: begin
          done_o  <= 1'b1;
          state_q <= S_IDLE;
        end

        default: state_q <= S_IDLE;
      endcase
    end
  end

  // ───────────────────────── unused suppression ────────────────
  // (none)

endmodule
