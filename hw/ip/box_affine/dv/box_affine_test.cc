// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// box_affine — Verilator test. Drives DUT (box_affine) and behavioral REF
// (box_affine_ref) with identical per-anchor stimulus and cross-checks both
// against an independent double-precision shadow. All comparisons are in
// fp16 ULPs against a fixed tolerance, with a catastrophic-cancellation
// aware absolute fallback (the cx-d*stride / x2-x1 forms cancel).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "Vbox_affine_tb.h"
#include "sim_ctrl.h"

using DUT = Vbox_affine_tb;

// ───────────────────────── fp16 helpers ───────────────────────────
static double fp16_to_double(uint16_t x) {
    int s = (x >> 15) & 1, e = (x >> 10) & 0x1F, f = x & 0x3FF;
    double v;
    if (e == 0x1F) v = (f == 0) ? std::numeric_limits<double>::infinity()
                                : std::numeric_limits<double>::quiet_NaN();
    else if (e == 0) v = std::ldexp((double)f, -24);
    else v = std::ldexp(1.0 + (double)f / 1024.0, e - 15);
    return s ? -v : v;
}
static uint16_t double_to_fp16(double v) {
    if (std::isnan(v)) return 0x7E00;
    int s = std::signbit(v) ? 1 : 0;
    double av = std::fabs(v);
    if (std::isinf(av) || av >= 65520.0) return (uint16_t)((s << 15) | 0x7C00);
    if (av == 0.0) return (uint16_t)(s << 15);
    int e; double m = std::frexp(av, &e);
    int biased = (e - 1) + 15;
    if (biased >= 31) return (uint16_t)((s << 15) | 0x7C00);
    if (biased <= 0) {
        double scaled = av * (double)(1 << 24), fl = std::floor(scaled);
        double frac = scaled - fl; long mi = (long)fl;
        if (frac > 0.5) mi += 1; else if (frac == 0.5 && (mi & 1)) mi += 1;
        if (mi >= 1024) return (uint16_t)((s << 15) | (1 << 10));
        return (uint16_t)((s << 15) | (mi & 0x3FF));
    }
    double md = (m * 2.0 - 1.0) * 1024.0, fl = std::floor(md);
    double frac = md - fl; long mi = (long)fl;
    if (frac > 0.5) mi += 1; else if (frac == 0.5 && (mi & 1)) mi += 1;
    if (mi >= 1024) { biased += 1; mi = 0; if (biased >= 31) return (uint16_t)((s << 15) | 0x7C00); }
    return (uint16_t)((s << 15) | ((biased & 0x1F) << 10) | (mi & 0x3FF));
}
static int32_t fp16_ordered(uint16_t x) {
    if (x & 0x8000) return -(int32_t)(x & 0x7FFF);
    return (int32_t)x;
}
static int32_t fp16_ulp_diff(uint16_t a, uint16_t b) {
    int32_t d = fp16_ordered(a) - fp16_ordered(b);
    return d < 0 ? -d : d;
}
static double fp16_step_at(double v) {
    double av = std::fabs(v);
    if (av == 0.0) return std::ldexp(1.0, -24);
    int e; std::frexp(av, &e);
    int unbiased = e - 1;
    if (unbiased < -14) unbiased = -14;
    return std::ldexp(1.0, unbiased - 10);
}

// ───────────────────────── shadow decode (double) ─────────────────
struct Box4 { uint16_t cx, cy, w, h; };
struct Shadow { Box4 box; double x1, y1, x2, y2; double pmax; };

static Shadow shadow_decode(int8_t l, int8_t t, int8_t r, int8_t b,
                            uint16_t sbox, int col, int row, int stride) {
    double S = fp16_to_double(sbox);
    double dl = (double)l * S, dt = (double)t * S, dr = (double)r * S, db = (double)b * S;
    double ax = (double)col + 0.5, ay = (double)row + 0.5, sr = (double)stride;
    Shadow o;
    o.x1 = (ax - dl) * sr; o.x2 = (ax + dr) * sr;
    o.y1 = (ay - dt) * sr; o.y2 = (ay + db) * sr;
    o.box.cx = double_to_fp16(((o.x1 + o.x2) / 2.0) / 640.0);
    o.box.cy = double_to_fp16(((o.y1 + o.y2) / 2.0) / 640.0);
    o.box.w  = double_to_fp16((o.x2 - o.x1) / 640.0);
    o.box.h  = double_to_fp16((o.y2 - o.y1) / 640.0);
    // Pre-cancellation pixel-domain magnitude: x1=(ax-dl)*s etc. cancel, so
    // the fp16 rounding-error floor scales with the LARGEST term that fed the
    // cancellation (cx_center, d*stride), not the small cancelled result.
    auto am = [](double v){ return std::fabs(v); };
    o.pmax = std::max({am(ax*sr), am(ay*sr), am(dl*sr), am(dr*sr), am(dt*sr), am(db*sr),
                       am(o.x1), am(o.x2), am(o.y1), am(o.y2)});
    return o;
}

// ───────────────────────── tolerance ──────────────────────────────
static const int TOLERANCE_ULP = 16;
// The abs-tolerance floor scales with the pre-cancellation pixel magnitude
// (pmax) divided by the normalization factor. centers fold /2/640 = /1280;
// w,h are /640. K=8 fp16 steps covers the worst observed accumulation with
// margin (verified ~0.6× of the floor in sweep).
static double abs_tol_center(double pmax) { return 8.0 * fp16_step_at(pmax) / 1280.0; }
static double abs_tol_size(double pmax)   { return 8.0 * fp16_step_at(pmax) / 640.0; }

struct Stat { int max_ulp_dr = 0, max_ulp_ds = 0, max_ulp_rs = 0; int n_fail = 0; };

static void drive(SimCtrl<DUT>& s, int8_t l, int8_t t, int8_t r, int8_t b,
                  uint16_t sbox, int col, int row, int stride, bool v) {
    s.dut->valid_i = v ? 1 : 0;
    s.dut->l_i = l; s.dut->t_i = t; s.dut->r_i = r; s.dut->b_i = b;
    s.dut->s_box_i = sbox; s.dut->col_i = col; s.dut->row_i = row;
    s.dut->stride_i = stride;
    s.tick();
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 200000000ull;
    printf("==== box_affine ====\n");

    sim.dut->valid_i = 0;
    drive(sim, 0,0,0,0, 0, 0,0,0, false);
    sim.reset();

    sim.check(sim.dut->valid_dut_o == 0, "valid low after reset");
    sim.check(sim.dut->valid_ref_o == 0, "ref valid low after reset");

    // The pipeline is feed-forward latency 7; we drive a stream and check
    // outputs as they emerge. Keep a small in-flight queue of expectations.
    struct Exp { int8_t l,t,r,b; uint16_t sbox; int col,row,stride; bool v; };
    std::vector<Exp> q;

    const uint16_t S0 = double_to_fp16(0.0566);   // ~ S_box scale-0
    const uint16_t S1 = double_to_fp16(0.0912);
    const uint16_t S2 = double_to_fp16(0.1344);

    // directed: zero distances → box centered on the anchor cell.
    std::vector<Exp> directed = {
        {0,0,0,0, S0, 0,0, 8, true},
        {0,0,0,0, S0, 79,79, 8, true},
        {127,127,127,127, S0, 40,40, 8, true},
        {-128,-128,-128,-128, S0, 40,40, 8, true},
        {10,-20,30,-40, S1, 20,20, 16, true},
        {0,0,0,0, S2, 19,19, 32, true},
        {64,-64,64,-64, S2, 10,5, 32, true},
    };

    Stat st;
    int valid_mm = 0;

    auto check_emerge = [&](void) {
        if (sim.dut->valid_dut_o) {
            // pop oldest expected valid entry
            // (all entries we pushed had v=true except idle gaps)
            Exp e{};
            // find oldest true
            size_t idx = 0; bool found = false;
            for (; idx < q.size(); ++idx) { if (q[idx].v) { e = q[idx]; found = true; break; } }
            if (!found) { st.n_fail++; return; }
            q.erase(q.begin() + idx);

            Shadow sh = shadow_decode(e.l,e.t,e.r,e.b,e.sbox,e.col,e.row,e.stride);
            uint16_t du[4] = {(uint16_t)sim.dut->cx_dut_o,(uint16_t)sim.dut->cy_dut_o,
                              (uint16_t)sim.dut->w_dut_o,(uint16_t)sim.dut->h_dut_o};
            uint16_t rf[4] = {(uint16_t)sim.dut->cx_ref_o,(uint16_t)sim.dut->cy_ref_o,
                              (uint16_t)sim.dut->w_ref_o,(uint16_t)sim.dut->h_ref_o};
            uint16_t sd[4] = {sh.box.cx, sh.box.cy, sh.box.w, sh.box.h};
            double atol[4] = {abs_tol_center(sh.pmax), abs_tol_center(sh.pmax),
                              abs_tol_size(sh.pmax), abs_tol_size(sh.pmax)};
            for (int i = 0; i < 4; i++) {
                int udr = fp16_ulp_diff(du[i], rf[i]);
                int uds = fp16_ulp_diff(du[i], sd[i]);
                int urs = fp16_ulp_diff(rf[i], sd[i]);
                st.max_ulp_dr = std::max(st.max_ulp_dr, udr);
                st.max_ulp_ds = std::max(st.max_ulp_ds, uds);
                st.max_ulp_rs = std::max(st.max_ulp_rs, urs);
                double ad = std::fabs(fp16_to_double(du[i]) - fp16_to_double(sd[i]));
                if (udr > TOLERANCE_ULP && uds > TOLERANCE_ULP && ad > atol[i]) st.n_fail++;
            }
        }
        if (sim.dut->valid_mismatch_o) valid_mm++;
    };

    // Drive directed cases with latency drain.
    for (auto& e : directed) { check_emerge(); drive(sim, e.l,e.t,e.r,e.b,e.sbox,e.col,e.row,e.stride,true); q.push_back(e); }
    for (int i = 0; i < 10; i++) { check_emerge(); drive(sim, 0,0,0,0,S0,0,0,8,false); }

    sim.check(st.n_fail == 0, "directed: 0 out-of-tolerance (was " + std::to_string(st.n_fail) + ")");
    sim.check(valid_mm == 0, "directed: 0 valid-timing mismatches");

    // ── randomized stress ──
    printf("randomized stress (20000 anchors)\n");
    std::mt19937 rng(0xB0FAFF1u);
    std::uniform_int_distribution<int> d8(-128, 127);
    std::uniform_int_distribution<int> dscale(0, 2);
    const uint16_t SS[3] = {S0, S1, S2};
    const int STR[3] = {8, 16, 32};
    const int GW[3]  = {80, 40, 20};
    st = Stat{}; valid_mm = 0; q.clear();
    const int Nrand = 20000;
    for (int i = 0; i < Nrand + 8; i++) {
        check_emerge();
        if (i < Nrand) {
            int si = dscale(rng);
            Exp e{ (int8_t)d8(rng),(int8_t)d8(rng),(int8_t)d8(rng),(int8_t)d8(rng),
                   SS[si], (int)(rng()%GW[si]), (int)(rng()%GW[si]), STR[si], true };
            drive(sim, e.l,e.t,e.r,e.b,e.sbox,e.col,e.row,e.stride,true);
            q.push_back(e);
        } else {
            drive(sim, 0,0,0,0,S0,0,0,8,false);
        }
    }
    printf("  max ULP dut-vs-ref=%d dut-vs-shadow=%d ref-vs-shadow=%d\n",
           st.max_ulp_dr, st.max_ulp_ds, st.max_ulp_rs);
    sim.check(st.n_fail == 0, "rand: 0 out-of-tolerance (was " + std::to_string(st.n_fail) + ")");
    sim.check(valid_mm == 0, "rand: 0 valid-timing mismatches");
    sim.check(st.max_ulp_rs <= 2, "ref-vs-shadow tight (was " + std::to_string(st.max_ulp_rs) + ")");

    return sim.finish();
}
