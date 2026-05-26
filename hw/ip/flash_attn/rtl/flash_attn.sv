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
// The MAC cells are fp16_macw: fp16 a*b plus a WIDE-float accumulator
// (ACC_EXP/ACC_MANT, default 8/21 ≈ fp32 precision), 5-cycle pipeline
// latency (FMA_LAT=5, mirrors fp16_lat_pkg::FP16_FMA_LAT). A result is not
// in cell_y until FMA_LAT cycles after its operands are driven. cell_y is
// the wide accumulator; cell_y_f16 is its combinational fp16 view, used by
// every fp16 consumer (softmax max/sum, exp arg, output sampling).
//
// The two REDUCTION LOOPS (DOT_K over DIM_Q, PV_K over BC) run at full
// throughput using INTERLEAVED PARTIAL SUMS: one MAC is issued every
// cycle, and the accumulator init is WIDE_ZERO for the first FMA_LAT
// indices (else cell_y feedback). The pipeline therefore maintains
// FMA_LAT independent partial sums per cell (residue classes k mod
// FMA_LAT). After the loop, a small drain state (S_DOT_CAP / S_PV_CAP,
// FMA_LAT cycles) combines the FMA_LAT final partials — which appear in
// cell_y on FMA_LAT consecutive cycles — with the combinational wide_add,
// and rounds the completed reduction to fp16 ONCE into Pdot / PV_part_reg.
//
// Accumulating wide (not fp16) is the point: the whole DIM_Q / BC sum is
// kept at ~fp32 precision and rounded to fp16 only at the very end, so the
// interleaved-partial reordering no longer perturbs the result. This is
// what recovers the marginal detection that fp16 accumulation dropped on
// bus.jpg (the chained real-RTL run matches ORT 5/5 at ACC_MANT≥16).
//
// The constant per-col-tile "drive then sample" gaps (SCALE, EXP_D,
// LUPD_D, MERGE_D, NORM_D) still span FMA_LAT cycles via ph_q; they are
// not ×reduction-length so they are not the throughput problem.
//
// Default parameters: BR=4, BC=4 (test-friendly). Production override
// at instantiation, e.g. BR=16 BC=32 for HEADS=2 N=400 DIM_Q=32 DIM_V=64.
//
// Cell counts at BR=16, BC=32:
//   score/PV fp16_macw array : BR*BC = 512  (wide-accumulate MAC)
//   LUPD per-row fp16_fma    : BR    = 16   (still fp16; short reduction)
//   norm reuses score array  : 0 extra
//   TOTAL MAC cells          : 528   ≤ 1500 budget
// The 512 score/PV cells are now fp16_macw rather than fp16_fma: a*b stays
// fp16 but the accumulator is a (1+ACC_EXP+ACC_MANT)-bit float, so each cell
// (and the per-cell wide_add / wide_to_fp16 in the *_CAP drains) is larger
// than the old fp16 cell. Area scales with ACC_MANT — dial it down if tight;
// ACC_MANT≥16 recovers the bus.jpg detection (default 21 for margin).
//
// State machine (one schedule per (head, row-tile)). Cycle costs below
// are with FMA_LAT=5 (L below):
//
//   foreach head h:
//     foreach row-tile br:
//       init m_r=-∞ l_r=0 O_acc=0; first_col=1
//       foreach col-tile bc:
//         DOT_K  : DIM_Q*L cyc — cell(r,j) ← cell(r,j) + Q[r,k]*K[j,k]
//         SCALE  : L cycles    — cell(r,j) ← TEMP * cell(r,j)
//         WAIT0  : 1 cycle     — settle hop into cell_y
//         RMAX_S : 1 cycle     — sample row_max_r from cell_y
//         MNEW   : 1 cycle     — register m_new_r, alpha_r
//         EXP_D  : L cycles    — drive cells with exp(s-m_new)
//         EXP_S  : 1 cycle     — sample P_reg from cell_y; row_sum_r
//         (LUPD overlaps PV: the per-row lupd fmas run on separate cells;
//          l_r is sampled inside the first PV_CAP, costing 0 extra cycles)
//         foreach dv_chunk:
//           PV_K   : BC*L cyc  — cell(r,c) ← cell(r,c) + P[r,j]*V[j,d]
//           PV_CAP : L cycles  — drain partials; sample l_r on dv_chunk 0
//           MERGE_D: L cycles  — drive cells: alpha*O_old + PV_part
//           MERGE_S: 1 cycle   — sample O_acc ← cell_y
//         m_r ← m_new_r; first_col ← 0
//       foreach norm_chunk:
//         NORM_D : L cycles    — drive cells: O_acc*inv_l
//         NORM_S : 1 cycle     — sample O_out ← cell_y
//   raise done_o
//
// Per-col-tile cycle cost (BR=16, BC=32, DIM_Q=32, DIM_V=64, L=FMA_LAT=5),
// with the interleaved-partial reduction loops (~1 MAC/cycle). The reduction
// loops run max(reduction,L) cycles; here DIM_Q=BC=32 ≥ L so they equal the
// reduction length and only the fixed L-spanning states grew (3→5):
//   DOT_K  DIM_Q + DOT_CAP(L) = 32 + 5 = 37
//   SCALE*L + WAIT0           = 5 + 1 = 6
//   RMAX_S                    = 1
//   MNEW                      = 1
//   EXP_D*L + EXP_S           = 5 + 1 = 6
//   (LUPD now overlaps PV — 0 cycles, was 6)
//   PV per chunk = BC + PV_CAP(L) + MERGE_D(L) + MERGE_S(1)
//                = 32 + 5 + 5 + 1 = 43
//   PV total                  = N_DV_CHUNK * 43 = 2 * 43 = 86
//   total per col-tile: 37+6+1+1+6+86 = 137 cycles
// Total col-tiles: HEADS * ceil(N/BR) * ceil(N/BC) = 2 * 25 * 13 = 650.
//   col-tile cycles  ≈ 650 * 137 = 89_050
//   norm cycles      = HEADS * ceil(N/BR) * N_DV_CHUNK * (L+1) = 600
//   TOTAL          ≈ 89.6k cycles (LUPD overlapped into PV, −3.9k vs the
//                    serial schedule). Throughput ≥10k FPS @1 GHz.

module flash_attn #(
  parameter int HEADS = 1,
  parameter int N     = 8,
  parameter int DIM_Q = 4,
  parameter int DIM_V = 4,
  parameter int BR    = 4,
  parameter int BC    = 4,
  // Softmax temperature, fp16. Default 1/sqrt(32) ≈ 0.17677669 = 0x31A8.
  parameter logic [15:0] TEMP_FP16 = 16'h31A8,
  // ── Wide accumulator format (the fix for fp16-accumulation error) ──
  // The two inner reductions (QK^T over DIM_Q, P·V over BC) accumulate in a
  // parameterized binary float (ACC_EXP exp bits, ACC_MANT stored mantissa
  // bits) and round to fp16 only once, at the end of each reduction. Wider
  // ACC_MANT ⇒ accumulation order matters less and the result tracks the
  // real-arithmetic golden far more closely. Default = 8/21 (a 30-bit,
  // ~fp32-precision accumulator). Sweep on real data to trade area/accuracy.
  parameter int unsigned ACC_EXP  = 8,
  parameter int unsigned ACC_MANT = 21,
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
  // fp16_fma / fp16_macw are now a 5-cycle pipeline (was 3, originally 1).
  // Mirrors fp16_lat_pkg::FP16_FMA_LAT (NOT imported here, kept as a local
  // copy so this leaf has no package dependency). Every "drive then sample"
  // gap and every running-accumulation step must span FMA_LAT cycles so
  // a fma result has landed in cell_y before it is read/re-accumulated.
  localparam int unsigned FMA_LAT = 5;
  localparam int N_BR        = (N + BR - 1) / BR;
  localparam int N_BC        = (N + BC - 1) / BC;
  localparam int N_DV_CHUNK  = (DIM_V + BC - 1) / BC;

  // The interleaved-partial reductions keep FMA_LAT independent residue
  // classes (k mod FMA_LAT). The drain (S_DOT_CAP / S_PV_CAP) reads the
  // FMA_LAT final partials on FMA_LAT consecutive cycles, which requires
  // each residue class to have been issued at least once *before* the
  // drain window opens — i.e. the loop must run for at least FMA_LAT
  // cycles. When the reduction length is shorter than FMA_LAT we pad the
  // loop with zero-MACs (a=b=0 via the existing < DIM_Q / < BC guards, so
  // the unused residue slots accumulate to WIDE_ZERO and drop out of the
  // sum). For production dims (DIM_Q,BC ≥ FMA_LAT) these equal DIM_Q/BC and
  // nothing changes.
  localparam int DOT_STEPS = (DIM_Q > int'(FMA_LAT)) ? DIM_Q : int'(FMA_LAT);
  localparam int PV_STEPS  = (BC    > int'(FMA_LAT)) ? BC    : int'(FMA_LAT);
  localparam logic [15:0] FP16_ONE     = 16'h3C00;
  localparam logic [15:0] FP16_ZERO    = 16'h0000;
  localparam logic [15:0] FP16_NEG_BIG = 16'hFBFF;  // very negative finite

  // Wide accumulator format (mirrors fp16_macw params).
  localparam int unsigned ACC_W    = 1 + ACC_EXP + ACC_MANT;
  localparam int          ACC_BIAS = (1 << (ACC_EXP-1)) - 1;
  localparam int          ACC_EXPMAX = (1 << ACC_EXP) - 1;
  localparam logic [ACC_W-1:0] WIDE_ZERO = '0;

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

  // ─────────────────── wide-float accumulator helpers ──────────────────
  // The cell array feeds back a wide-float accumulator (ACC_EXP/ACC_MANT)
  // for the two inner reductions. These convert at the fp16 boundaries and
  // combine the FMA_LAT interleaved partials in wide precision.
  localparam int CW    = int'(ACC_MANT) + 1;            // wide significand width
  localparam int WADD  = CW + 48;                       // wide-add working width

  // exact fp16 → wide (ACC_MANT ≥ 10, ACC_EXP ≥ 5 so fp16 always fits).
  function automatic logic [ACC_W-1:0] fp16_to_wide(input logic [15:0] x);
    logic        s; logic [4:0] be; logic [9:0] f;
    int          e, biased, p; logic [9:0] fn; logic [ACC_MANT-1:0] frac;
    begin
      s = x[15]; be = x[14:10]; f = x[9:0];
      if (be == 5'd0 && f == 10'd0) return {s, {(ACC_W-1){1'b0}}};        // zero
      if (be == 5'd31)
        return (f == 0) ? {s, {ACC_EXP{1'b1}}, {ACC_MANT{1'b0}}}          // inf
                        : {1'b0, {ACC_EXP{1'b1}}, 1'b1, {(ACC_MANT-1){1'b0}}}; // nan
      if (be == 5'd0) begin
        // fp16 subnormal: value = f * 2^-24. Normalize into a wide normal.
        p = 0;
        for (int i = 9; i >= 0; i--) if (f[i] && p == 0) p = i;
        // leading-1 at bit p → value = 1.<rest> * 2^(p-24)
        fn     = f << (9 - p);                                            // align MSB to bit 9
        frac   = {fn[8:0], {(ACC_MANT-9){1'b0}}};                         // drop implicit, left-justify
        biased = (p - 24) + ACC_BIAS;
        return {s, ACC_EXP'(biased), frac};
      end
      // normal: value = 1.f * 2^(be-15)
      biased = int'(be) - 15 + ACC_BIAS;
      frac   = {f, {(ACC_MANT-10){1'b0}}};                                // left-justify 10-bit frac
      return {s, ACC_EXP'(biased), frac};
    end
  endfunction

  // wide → fp16 with RNE rounding.
  function automatic logic [15:0] wide_to_fp16(input logic [ACC_W-1:0] x);
    logic                s; logic [ACC_EXP-1:0] be; logic [ACC_MANT-1:0] f;
    logic [ACC_MANT:0]   sig;                 // {1, frac}
    int                  E, biased, sh;
    logic [9:0]          m10; logic guard, round_b, sticky;
    logic [10:0]         m_r; int bexp;
    logic [ACC_MANT:0]   shifted; logic dropped;
    begin
      s = x[ACC_W-1]; be = x[ACC_W-2 -: ACC_EXP]; f = x[ACC_MANT-1:0];
      if (be == '0 && f == '0) return {s, 15'd0};                         // zero
      if (be == ACC_EXPMAX[ACC_EXP-1:0])
        return (f == '0) ? {s, 5'b11111, 10'b0} : 16'h7E00;               // inf/nan
      E = int'(be) - ACC_BIAS;                  // unbiased
      biased = E + 15;                          // fp16 biased exp
      sig = {1'b1, f};                          // ACC_MANT+1 bits, MSB at ACC_MANT
      if (biased >= 31) return {s, 5'b11111, 10'b0};                      // overflow → Inf
      if (biased <= 0) begin
        // subnormal/underflow: right-shift significand by (1-biased), RNE.
        sh = 1 - biased;
        if (sh > ACC_MANT) return {s, 15'd0};                            // total underflow
        shifted = sig >> sh;
        dropped = 1'b0;
        for (int i = 0; i < sh; i++) if (sig[i]) dropped = 1'b1;
        // now treat 'shifted' as a value with implicit point above bit ACC_MANT
        m10     = shifted[ACC_MANT-1 -: 10];
        guard   = (ACC_MANT-11 >= 0) ? shifted[ACC_MANT-11] : 1'b0;
        round_b = 1'b0; sticky = dropped;
        for (int i = 0; i < ACC_MANT-11; i++) if (shifted[i]) sticky = 1'b1;
        m_r = {1'b0, m10} + 11'((guard && ((round_b|sticky) || m10[0])) ? 1 : 0);
        if (m_r[10]) return {s, 5'd1, 10'd0};                            // promoted to min normal
        return {s, 5'd0, m_r[9:0]};
      end
      // normal: keep top 10 fraction bits, RNE from the rest. (Guards keep
      // the bit-selects legal for the degenerate ACC_MANT==10 case, where
      // the wide mantissa already matches fp16 and no rounding bits exist.)
      m10     = f[ACC_MANT-1 -: 10];
      guard   = (ACC_MANT-11 >= 0) ? f[ACC_MANT-11] : 1'b0;
      round_b = (ACC_MANT-12 >= 0) ? f[ACC_MANT-12] : 1'b0;
      sticky  = 1'b0;
      for (int i = 0; i < ACC_MANT-12; i++) if (f[i]) sticky = 1'b1;
      m_r  = {1'b0, m10} + 11'((guard && ((round_b|sticky) || m10[0])) ? 1 : 0);
      bexp = biased;
      if (m_r[10]) begin bexp = bexp + 1; m_r = 11'b0; end
      if (bexp >= 31) return {s, 5'b11111, 10'b0};
      return {s, 5'(bexp), m_r[9:0]};
    end
  endfunction

  // wide + wide → wide, RNE (max-anchor, combinational). Used to combine the
  // FMA_LAT interleaved partials. Mirrors the fp16_macw add datapath.
  function automatic logic [ACC_W-1:0] wide_add(input logic [ACC_W-1:0] a,
                                                input logic [ACC_W-1:0] b);
    logic sa, sb; logic [ACC_EXP-1:0] bea, beb; logic [ACC_MANT-1:0] fa, fb;
    logic az, bz, ainf, binf, anan, bnan;
    logic [CW-1:0] siga, sigb; int ea, eb, ta, tb, anchor, sA, sB;
    logic [WADD-1:0] biga, bigb, dropped; logic stky;
    logic [WADD:0] mag; logic rsign, same; int msb; logic any;
    int e_msb, biased, sh; logic [WADD:0] norm, mpost; logic subdrop;
    logic [ACC_MANT-1:0] mant; logic g, r, st; logic [ACC_MANT:0] mr; int bx;
    begin
      sa = a[ACC_W-1]; bea = a[ACC_W-2 -: ACC_EXP]; fa = a[ACC_MANT-1:0];
      sb = b[ACC_W-1]; beb = b[ACC_W-2 -: ACC_EXP]; fb = b[ACC_MANT-1:0];
      az = (bea=='0)&&(fa=='0); bz = (beb=='0)&&(fb=='0);
      ainf=(bea==ACC_EXPMAX[ACC_EXP-1:0])&&(fa=='0); binf=(beb==ACC_EXPMAX[ACC_EXP-1:0])&&(fb=='0);
      anan=(bea==ACC_EXPMAX[ACC_EXP-1:0])&&(fa!='0); bnan=(beb==ACC_EXPMAX[ACC_EXP-1:0])&&(fb!='0);
      if (anan||bnan||(ainf&&binf&&(sa!=sb))) return {1'b0,{ACC_EXP{1'b1}},1'b1,{(ACC_MANT-1){1'b0}}};
      if (ainf) return a;
      if (binf) return b;
      if (az) return b;
      if (bz) return a;
      siga = {1'b1, fa}; ea = int'(bea) - ACC_BIAS - int'(ACC_MANT);
      sigb = {1'b1, fb}; eb = int'(beb) - ACC_BIAS - int'(ACC_MANT);
      if (bea=='0) begin siga = {1'b0, fa}; ea = 1 - ACC_BIAS - int'(ACC_MANT); end
      if (beb=='0) begin sigb = {1'b0, fb}; eb = 1 - ACC_BIAS - int'(ACC_MANT); end
      ta = ea + (CW-1); tb = eb + (CW-1);
      anchor = (ta > tb) ? ta : tb;
      // place a
      sA = ea - anchor + (WADD-2); stky = 1'b0;
      if (sA >= 0) biga = (sA < WADD) ? (WADD'(siga) << sA) : '0;
      else begin
        if (-sA < WADD) begin biga = WADD'(siga) >> (-sA); dropped = WADD'(siga) << (WADD+sA); if (dropped!='0) stky=1'b1; end
        else begin biga = '0; if (siga!='0) stky=1'b1; end
      end
      // place b
      sB = eb - anchor + (WADD-2);
      if (sB >= 0) bigb = (sB < WADD) ? (WADD'(sigb) << sB) : '0;
      else begin
        if (-sB < WADD) begin bigb = WADD'(sigb) >> (-sB); dropped = WADD'(sigb) << (WADD+sB); if (dropped!='0) stky=1'b1; end
        else begin bigb = '0; if (sigb!='0) stky=1'b1; end
      end
      same = (sa == sb);
      if (same) begin mag = {1'b0,biga} + {1'b0,bigb}; rsign = sa; end
      else if (biga >= bigb) begin mag = {1'b0,(biga-bigb)}; rsign = sa; end
      else begin mag = {1'b0,(bigb-biga)}; rsign = sb; end
      any = 1'b0; msb = 0;
      for (int i = WADD; i >= 0; i--) if (mag[i] && !any) begin any=1'b1; msb=i; end
      if (!any && !stky) return {1'b0,{(ACC_W-1){1'b0}}};
      e_msb = msb - (WADD-2) + anchor;
      sh = WADD - msb;
      norm = (!any) ? '0 : ((sh<=0) ? mag : (mag << sh));
      biased = e_msb + ACC_BIAS;
      if (biased <= 0) begin
        int ss; ss = 1 - biased;
        mpost = (ss >= WADD+1) ? '0 : (norm >> ss);
        subdrop = 1'b0;
        for (int i = 0; i <= WADD; i++) if ((i < ss) && norm[i]) subdrop = 1'b1;
        bx = 0;
      end else begin
        mpost = norm; subdrop = 1'b0; bx = biased;
      end
      mant = mpost[WADD-1 -: ACC_MANT];
      g    = mpost[WADD-1-ACC_MANT];
      r    = (WADD-2-ACC_MANT >= 0) ? mpost[WADD-2-ACC_MANT] : 1'b0;
      st   = stky | subdrop;
      for (int i = 0; i < WADD-2-ACC_MANT; i++) if (mpost[i]) st = 1'b1;
      mr = {1'b0, mant} + (ACC_MANT+1)'((g && ((r|st) || mant[0])) ? 1 : 0);
      if (mr[ACC_MANT]) begin
        if (bx == 0) bx = 1; else bx = bx + 1;
        mant = '0;
      end else mant = mr[ACC_MANT-1:0];
      if (bx >= ACC_EXPMAX) return {rsign, {ACC_EXP{1'b1}}, {ACC_MANT{1'b0}}};
      return {rsign, ACC_EXP'(bx), mant};
    end
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
  // a/b stay fp16 (multiply operands); c/y are the WIDE accumulator so the
  // two inner reductions never round to fp16 until their *_CAP drain.
  // cell_y_f16 is the combinational fp16 view used by every fp16 consumer
  // (softmax max/sum, exp argument, sampling into P_reg/O_acc/O_out).
  logic [15:0]      cell_a   [BR][BC];
  logic [15:0]      cell_b   [BR][BC];
  logic [ACC_W-1:0] cell_c   [BR][BC];
  logic [ACC_W-1:0] cell_y   [BR][BC];
  logic [15:0]      cell_y_f16 [BR][BC];

  generate
    for (genvar gr = 0; gr < BR; gr++) begin : g_row
      for (genvar gc = 0; gc < BC; gc++) begin : g_col
        fp16_macw #(.ACC_EXP(ACC_EXP), .ACC_MANT(ACC_MANT)) u_fma (
          .clk_i  (clk_i),
          .rst_ni (rst_ni),
          .a_i    (cell_a[gr][gc]),
          .b_i    (cell_b[gr][gc]),
          .c_i    (cell_c[gr][gc]),
          .y_o    (cell_y[gr][gc])
        );
        assign cell_y_f16[gr][gc] = wide_to_fp16(cell_y[gr][gc]);
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
    S_DOT_CAP,
    S_SCALE,
    S_WAIT0,
    S_RMAX_S,
    S_MNEW,
    S_EXP_D,
    S_EXP_S,
    S_PV_K,
    S_PV_CAP,
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

  // Sub-phase counter: holds a producing/accumulation state for FMA_LAT
  // cycles so the 3-cycle fma pipeline result is in cell_y/lupd_y before
  // we sample it or feed it back. Reset to 0 on every state transition.
  int unsigned ph_q;

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
  // Interleaved-partial capture accumulators. The pipelined mac maintains
  // FMA_LAT independent partial sums (residue classes k mod FMA_LAT); after
  // the accumulation loop the FMA_LAT final partials appear on FMA_LAT
  // consecutive cycles and are combined here in WIDE precision (wide_add).
  // The reduction result is rounded to fp16 exactly once at the *_CAP drain.
  logic [ACC_W-1:0] Pdot_wide [BR][BC]; // QK^T dot product, wide (over DIM_Q)
  logic [ACC_W-1:0] Ppv_wide  [BR][BC]; // P*V partial, wide (over BC)
  logic [15:0]      Pdot      [BR][BC]; // QK^T dot, rounded to fp16 for SCALE

  // ───────────────────────── reductions (combinational) ────────
  logic [15:0] row_max_comb [BR];
  logic [15:0] row_sum_comb [BR];
  logic [15:0] inv_l_r      [BR];

  always_comb begin
    for (int r = 0; r < BR; r++) begin
      logic [15:0] acc_m, acc_s;
      acc_m = cell_y_f16[r][0];
      acc_s = cell_y_f16[r][0];
      for (int j = 1; j < BC; j++) begin
        acc_m = fp16_max(acc_m, cell_y_f16[r][j]);
        acc_s = fp16_add(acc_s, cell_y_f16[r][j]);
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
  // Default drive is the IDENTITY operation: 0*0 + c = c, with c = the wide
  // cell_y feedback. (a/b are fp16 so we can't pass the wide accumulator as a
  // multiply operand; driving a=b=0 and c=cell_y holds the wide value
  // exactly.) This keeps cell_y unchanged in "hold" states. Producing states
  // override with their own a/b (fp16) and c (wide).
  always_comb begin
    for (int r = 0; r < BR; r++) begin
      for (int c = 0; c < BC; c++) begin
        cell_a[r][c] = FP16_ZERO;
        cell_b[r][c] = FP16_ZERO;
        cell_c[r][c] = cell_y[r][c];
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
            // Interleaved partials: zero-init the FMA_LAT residue classes
            // (kq_q < FMA_LAT), then feed back each residue's running WIDE
            // partial (cell_y is the result from FMA_LAT indices ago).
            cell_c[r][j] = (int'(kq_q) < FMA_LAT) ? WIDE_ZERO : cell_y[r][j];
          end
        end
      end
      S_SCALE: begin
        for (int r = 0; r < BR; r++)
          for (int j = 0; j < BC; j++) begin
            // Dot product now lives in Pdot (combined from the interleaved
            // partials in S_DOT_CAP), not in cell_y.
            cell_a[r][j] = Pdot[r][j];
            cell_b[r][j] = TEMP_FP16;
            cell_c[r][j] = WIDE_ZERO;
          end
      end
      S_EXP_D: begin
        for (int r = 0; r < BR; r++) begin
          for (int j = 0; j < BC; j++) begin
            logic [15:0] d_l;
            // cell_y here holds the scaled score S (wide); read its fp16 view.
            d_l = fp16_sub(m_new_r[r], cell_y_f16[r][j]);
            cell_a[r][j] = fp16_exp_negative(d_l);
            cell_b[r][j] = FP16_ONE;
            cell_c[r][j] = WIDE_ZERO;
          end
        end
      end
      S_PV_K: begin
        for (int r = 0; r < BR; r++) begin
          for (int c = 0; c < BC; c++) begin
            int j_g_l, d_idx_l;
            j_g_l   = col_idx(pv_j_q);
            d_idx_l = int'(dv_chunk_q) * BC + c;
            // int'(pv_j_q) < BC also masks the FMA_LAT-pad cycles (when
            // BC < FMA_LAT) so P_reg[r][pv_j_q] is never read out of range.
            cell_a[r][c] = (int'(pv_j_q) < BC && j_g_l < N && d_idx_l < DIM_V)
                            ? P_reg[r][pv_j_q] : FP16_ZERO;
            cell_b[r][c] = (int'(pv_j_q) < BC && j_g_l < N && d_idx_l < DIM_V)
                            ? V[head_q][j_g_l][d_idx_l] : FP16_ZERO;
            // Interleaved partials over the BC reduction (same scheme as DOT_K).
            cell_c[r][c] = (int'(pv_j_q) < FMA_LAT) ? WIDE_ZERO : cell_y[r][c];
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
            cell_c[r][c] = first_col ? WIDE_ZERO : fp16_to_wide(PV_part_reg[r][c]);
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
            cell_c[r][c] = WIDE_ZERO;
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
      ph_q          <= 0;
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
          Pdot[r][c]        <= FP16_ZERO;
          Pdot_wide[r][c]   <= WIDE_ZERO;
          Ppv_wide[r][c]    <= WIDE_ZERO;
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
            ph_q    <= 0;
            state_q <= S_DOT_K;
          end
        end

        // S_DOT_K — running QK^T accumulation with INTERLEAVED partials.
        // Issue one MAC per cycle (advance kq_q every cycle). The pipelined
        // fma keeps FMA_LAT independent partial sums (residue k mod FMA_LAT).
        // After the last index we go to S_DOT_CAP to drain & combine them.
        S_DOT_K: begin
          if (kq_q == DOT_STEPS - 1) begin
            kq_q    <= 0;
            ph_q    <= 0;
            state_q <= S_DOT_CAP;
          end else begin
            kq_q <= kq_q + 1;
          end
        end

        // S_DOT_CAP — drain the FMA_LAT final interleaved partials. They
        // appear in cell_y on FMA_LAT consecutive cycles (the last MACs
        // issued in S_DOT_K are still in flight). The default identity
        // drive issues here land after this window and don't disturb them.
        // Combine the wide partials with wide_add, then round the completed
        // dot product to fp16 ONCE into Pdot (the single rounding of the
        // whole DIM_Q reduction — this is the accuracy win).
        S_DOT_CAP: begin
          for (int r = 0; r < BR; r++)
            for (int j = 0; j < BC; j++) begin
              logic [ACC_W-1:0] comb;
              comb = (ph_q == 0) ? cell_y[r][j]
                                 : wide_add(Pdot_wide[r][j], cell_y[r][j]);
              Pdot_wide[r][j] <= comb;
              if (ph_q == FMA_LAT - 1) Pdot[r][j] <= wide_to_fp16(comb);
            end
          if (ph_q == FMA_LAT - 1) begin
            ph_q    <= 0;
            state_q <= S_SCALE;
          end else begin
            ph_q <= ph_q + 1;
          end
        end

        // S_SCALE drives cell_y*TEMP for FMA_LAT cycles so the scaled
        // scores have settled before S_RMAX_S samples row_max.
        S_SCALE: begin
          if (ph_q == FMA_LAT - 1) begin
            ph_q    <= 0;
            state_q <= S_WAIT0;
          end else begin
            ph_q <= ph_q + 1;
          end
        end

        // S_WAIT0: scaled scores are already in cell_y here (SCALE held
        // FMA_LAT). One identity cycle keeps cell_y stable into S_RMAX_S.
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

        // S_EXP_D drives cell(r,j) <- exp(scaledScore - m_new) for
        // FMA_LAT cycles. cell_y holds S (scaled scores) for the whole
        // window (the identity ops in WAIT0/RMAX/MNEW keep it stable), so
        // the exp argument is identical on every re-issue. After the
        // window cell_y holds P.
        S_EXP_D: begin
          if (ph_q == FMA_LAT - 1) begin
            ph_q    <= 0;
            state_q <= S_EXP_S;
          end else begin
            ph_q <= ph_q + 1;
          end
        end

        S_EXP_S: begin
          // cell_y now holds P. Sample its fp16 view into P_reg and row_sum.
          for (int r = 0; r < BR; r++)
            for (int c = 0; c < BC; c++) P_reg[r][c] <= cell_y_f16[r][c];
          for (int r = 0; r < BR; r++) row_sum_r[r] <= row_sum_comb[r];
          // The l-update (LUPD) now overlaps PV: the per-row lupd fmas are
          // driven combinationally and run on the SEPARATE per-row fma cells
          // (disjoint from the BR*BC score/PV array), so once row_sum_r /
          // alpha_r are stable here, lupd_y settles within FMA_LAT cycles —
          // far inside the PV reduction (≥ FMA_LAT cyc). l_r is sampled in
          // S_PV_CAP. This drops the dedicated S_LUPD_D/S_LUPD_S states
          // (−(FMA_LAT+1) cyc/col-tile) at zero area. l_r is only consumed
          // at the final NORM and by the next col-tile, both far later.
          pv_j_q     <= 0;
          dv_chunk_q <= 0;
          state_q    <= S_PV_K;
        end

        // S_PV_K — running P*V accumulation with INTERLEAVED partials
        // (same scheme as S_DOT_K). One MAC per cycle over j=0..BC-1.
        S_PV_K: begin
          if (pv_j_q == PV_STEPS - 1) begin
            pv_j_q  <= 0;
            ph_q    <= 0;
            state_q <= S_PV_CAP;
          end else begin
            pv_j_q <= pv_j_q + 1;
          end
        end

        // S_PV_CAP — drain & combine the FMA_LAT final PV partials in WIDE
        // precision into Ppv_wide, then round the completed P·V partial to
        // fp16 ONCE into PV_part_reg so S_MERGE_D is unchanged downstream.
        S_PV_CAP: begin
          for (int r = 0; r < BR; r++)
            for (int c = 0; c < BC; c++) begin
              logic [ACC_W-1:0] comb;
              comb = (ph_q == 0) ? cell_y[r][c]
                                 : wide_add(Ppv_wide[r][c], cell_y[r][c]);
              Ppv_wide[r][c] <= comb;
              // On the final CAP cycle, comb is the complete PV partial.
              if (ph_q == FMA_LAT - 1) PV_part_reg[r][c] <= wide_to_fp16(comb);
            end
          // Overlapped LUPD sample: by the first PV drain (dv_chunk 0), the
          // lupd fmas have seen stable row_sum_r/alpha_r for ≥ PV_STEPS+FMA_LAT
          // cycles, so lupd_y is fully settled. Sample l_r exactly once here
          // (after which the always-on lupd driver re-accumulates a stale
          // value, but it is never re-sampled this col-tile).
          if (dv_chunk_q == 0 && ph_q == FMA_LAT - 1)
            for (int r = 0; r < BR; r++) l_r[r] <= lupd_y[r];
          if (ph_q == FMA_LAT - 1) begin
            ph_q    <= 0;
            state_q <= S_MERGE_D;
          end else begin
            ph_q <= ph_q + 1;
          end
        end

        // S_MERGE_D drives cell(r,c) <- alpha*O_old + PV_part (or PV_part
        // on first_col) for FMA_LAT cycles. Operands are stable registers.
        S_MERGE_D: begin
          if (ph_q == FMA_LAT - 1) begin
            ph_q    <= 0;
            state_q <= S_MERGE_S;
          end else begin
            ph_q <= ph_q + 1;
          end
        end

        S_MERGE_S: begin
          for (int r = 0; r < BR; r++) begin
            for (int c = 0; c < BC; c++) begin
              int d_idx;
              d_idx = int'(dv_chunk_q) * BC + c;
              if (d_idx < DIM_V) O_acc[r][d_idx] <= cell_y_f16[r][c];
            end
          end
          ph_q <= 0;
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

        // S_NORM_D drives O_acc*inv_l for FMA_LAT cycles; inv_l_r and
        // O_acc are stable, so the re-issues are identical and the
        // normalised output has settled before S_NORM_S samples it.
        S_NORM_D: begin
          if (ph_q == FMA_LAT - 1) begin
            ph_q    <= 0;
            state_q <= S_NORM_S;
          end else begin
            ph_q <= ph_q + 1;
          end
        end

        S_NORM_S: begin
          for (int r = 0; r < BR; r++) begin
            for (int c = 0; c < BC; c++) begin
              int d_idx, r_g;
              d_idx = int'(norm_chunk_q) * BC + c;
              r_g   = int'(br_q) * BR + r;
              if (d_idx < DIM_V && r_g < N) begin
                O_out[head_q][r_g][d_idx] <= cell_y_f16[r][c];
              end
            end
          end
          ph_q <= 0;
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
