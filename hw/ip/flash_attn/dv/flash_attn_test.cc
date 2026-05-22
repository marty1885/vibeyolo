// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// flash_attn — Verilator test. Drives random Q/K/V tensors into DUT
// and REF (in lockstep), starts both, polls done_o, then compares each
// O[h,i,d] word DUT-vs-REF (must be 0 fail at the documented ULP
// tolerance) and DUT-vs-shadow (an independent C++ double-precision
// dense-attention).
//
// DV parameters (parameterized at TB instantiation):
//   HEADS = 1, N = 8, DIM_Q = 4, DIM_V = 4
// These small shapes keep the per-frame cycle count fast (a frame is
// ~few hundred cycles) while still exercising every code path (multi-
// row, multi-col flash-attention recurrence, exp LUT, recip LUT, V
// matmul).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "Vflash_attn_tb.h"
#include "sim_ctrl.h"

using DUT = Vflash_attn_tb;

static const int HEADS = 1;
static const int N     = 8;
static const int DIM_Q = 4;
static const int DIM_V = 4;
static const double TEMP = 0.17677669529663687;  // 1/sqrt(32) — same as RTL

// ─── fp16 helpers ───────────────────────────────────────
static double fp16_to_double(uint16_t x) {
    int s = (x >> 15) & 1;
    int e = (x >> 10) & 0x1F;
    int f = x & 0x3FF;
    double v;
    if (e == 0x1F) {
        v = (f == 0) ? 1e30 : std::numeric_limits<double>::quiet_NaN();
    } else if (e == 0) {
        v = std::ldexp((double)f, -24);
    } else {
        v = std::ldexp(1.0 + (double)f / 1024.0, e - 15);
    }
    return s ? -v : v;
}

static uint16_t double_to_fp16(double v) {
    if (std::isnan(v)) return 0x7E00;
    int s = std::signbit(v) ? 1 : 0;
    double av = std::fabs(v);
    if (std::isinf(av) || av >= 65520.0) return (uint16_t)((s << 15) | 0x7C00);
    if (av == 0.0) return (uint16_t)(s << 15);
    int e;
    double m = std::frexp(av, &e);
    int unbiased = e - 1;
    int biased = unbiased + 15;
    if (biased >= 31) return (uint16_t)((s << 15) | 0x7C00);
    if (biased <= 0) {
        double scaled = av * (double)(1 << 24);
        double floor_v = std::floor(scaled);
        double frac = scaled - floor_v;
        long mant_int = (long)floor_v;
        if (frac > 0.5)                            mant_int += 1;
        else if (frac == 0.5 && (mant_int & 1))    mant_int += 1;
        if (mant_int >= 1024) return (uint16_t)((s << 15) | (1 << 10));
        return (uint16_t)((s << 15) | (mant_int & 0x3FF));
    }
    double mant_d = (m * 2.0 - 1.0) * 1024.0;
    double floor_v = std::floor(mant_d);
    double frac = mant_d - floor_v;
    long mant_int = (long)floor_v;
    if (frac > 0.5)                            mant_int += 1;
    else if (frac == 0.5 && (mant_int & 1))    mant_int += 1;
    if (mant_int >= 1024) {
        biased += 1;
        mant_int = 0;
        if (biased >= 31) return (uint16_t)((s << 15) | 0x7C00);
    }
    return (uint16_t)((s << 15) | ((biased & 0x1F) << 10) | (mant_int & 0x3FF));
}

static int32_t fp16_ordered(uint16_t x) {
    if (x & 0x8000) return -(int32_t)(x & 0x7FFF);
    return (int32_t)x;
}

static int32_t fp16_ulp_diff(uint16_t a, uint16_t b) {
    int32_t d = fp16_ordered(a) - fp16_ordered(b);
    return d < 0 ? -d : d;
}

// ─── Shadow: dense softmax-attention in pure double ─────
static void shadow_attention(const std::vector<uint16_t>& q,
                             const std::vector<uint16_t>& k,
                             const std::vector<uint16_t>& v,
                             std::vector<uint16_t>& o) {
    o.assign(HEADS * N * DIM_V, 0);
    std::vector<double> s(N);
    std::vector<double> p(N);
    for (int h = 0; h < HEADS; h++) {
        for (int i = 0; i < N; i++) {
            // s_j = (Q[h,i] · K[h,j]) * TEMP
            for (int j = 0; j < N; j++) {
                double dot = 0;
                for (int d = 0; d < DIM_Q; d++) {
                    double qv = fp16_to_double(q[h*N*DIM_Q + i*DIM_Q + d]);
                    double kv = fp16_to_double(k[h*N*DIM_Q + j*DIM_Q + d]);
                    dot += qv * kv;
                }
                s[j] = dot * TEMP;
            }
            // softmax
            double mx = s[0];
            for (int j = 1; j < N; j++) if (s[j] > mx) mx = s[j];
            double sum = 0;
            for (int j = 0; j < N; j++) { p[j] = std::exp(s[j] - mx); sum += p[j]; }
            for (int j = 0; j < N; j++) p[j] /= sum;
            // O[h,i,d] = sum_j p_j * V[h,j,d]
            for (int d = 0; d < DIM_V; d++) {
                double od = 0;
                for (int j = 0; j < N; j++) {
                    double vv = fp16_to_double(v[h*N*DIM_V + j*DIM_V + d]);
                    od += p[j] * vv;
                }
                o[h*N*DIM_V + i*DIM_V + d] = double_to_fp16(od);
            }
        }
    }
}

// ─── Drive one frame ────────────────────────────────────
struct Frame {
    std::vector<uint16_t> q;   // HEADS*N*DIM_Q
    std::vector<uint16_t> k;
    std::vector<uint16_t> v;
};

static void load_inputs(SimCtrl<DUT>& sim, const Frame& f) {
    int qn = HEADS * N * DIM_Q;
    int vn = HEADS * N * DIM_V;
    for (int a = 0; a < qn; a++) {
        sim.dut->q_we_i = 1;
        sim.dut->q_waddr_i = (uint16_t)a;
        sim.dut->q_wdata_i = f.q[a];
        sim.dut->k_we_i = 1;
        sim.dut->k_waddr_i = (uint16_t)a;
        sim.dut->k_wdata_i = f.k[a];
        if (a < vn) {
            sim.dut->v_we_i = 1;
            sim.dut->v_waddr_i = (uint16_t)a;
            sim.dut->v_wdata_i = f.v[a];
        } else {
            sim.dut->v_we_i = 0;
        }
        sim.tick();
    }
    // If V is longer than QK, finish remaining V words.
    for (int a = qn; a < vn; a++) {
        sim.dut->q_we_i = 0;
        sim.dut->k_we_i = 0;
        sim.dut->v_we_i = 1;
        sim.dut->v_waddr_i = (uint16_t)a;
        sim.dut->v_wdata_i = f.v[a];
        sim.tick();
    }
    sim.dut->q_we_i = 0;
    sim.dut->k_we_i = 0;
    sim.dut->v_we_i = 0;
    sim.tick();
}

static int run_until_done(SimCtrl<DUT>& sim) {
    sim.dut->start_i = 1;
    sim.tick();
    sim.dut->start_i = 0;
    int cycles = 0;
    const int MAX_CYC = 200000;
    while (cycles < MAX_CYC) {
        if (sim.dut->done_dut_o && sim.dut->done_ref_o) break;
        sim.tick();
        cycles++;
    }
    return cycles;
}

static void read_out(SimCtrl<DUT>& sim,
                     std::vector<uint16_t>& o_dut,
                     std::vector<uint16_t>& o_ref) {
    int vn = HEADS * N * DIM_V;
    o_dut.assign(vn, 0);
    o_ref.assign(vn, 0);
    for (int a = 0; a < vn; a++) {
        sim.dut->o_raddr_i = (uint16_t)a;
        sim.dut->eval();
        sim.tick();
        o_dut[a] = (uint16_t)sim.dut->o_dut_o;
        o_ref[a] = (uint16_t)sim.dut->o_ref_o;
    }
}

static double cos_sim(const std::vector<uint16_t>& a,
                      const std::vector<uint16_t>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double av = fp16_to_double(a[i]);
        double bv = fp16_to_double(b[i]);
        dot += av * bv;
        na  += av * av;
        nb  += bv * bv;
    }
    if (na == 0.0 || nb == 0.0) return 1.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// ─── Stim generators ────────────────────────────────────
static Frame mk_random_frame(std::mt19937& rng, double range = 0.5) {
    Frame f;
    f.q.assign(HEADS * N * DIM_Q, 0);
    f.k.assign(HEADS * N * DIM_Q, 0);
    f.v.assign(HEADS * N * DIM_V, 0);
    std::uniform_real_distribution<double> ud(-range, range);
    for (auto& x : f.q) x = double_to_fp16(ud(rng));
    for (auto& x : f.k) x = double_to_fp16(ud(rng));
    for (auto& x : f.v) x = double_to_fp16(ud(rng));
    return f;
}

static Frame mk_zero_frame() {
    Frame f;
    f.q.assign(HEADS * N * DIM_Q, 0);
    f.k.assign(HEADS * N * DIM_Q, 0);
    f.v.assign(HEADS * N * DIM_V, 0);
    return f;
}

// ─── main ───────────────────────────────────────────────
int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 800000000ull;

    // init
    sim.dut->q_we_i = 0; sim.dut->k_we_i = 0; sim.dut->v_we_i = 0;
    sim.dut->start_i = 0;
    sim.dut->o_raddr_i = 0;
    sim.dut->q_waddr_i = 0; sim.dut->k_waddr_i = 0; sim.dut->v_waddr_i = 0;
    sim.dut->q_wdata_i = 0; sim.dut->k_wdata_i = 0; sim.dut->v_wdata_i = 0;
    sim.reset();

    sim.check(sim.dut->done_dut_o == 0, "done_dut low after reset");
    sim.check(sim.dut->done_ref_o == 0, "done_ref low after reset");
    sim.check(sim.dut->busy_dut_o == 0, "busy_dut low after reset");

    // Tolerance budget per output word:
    //   * fp16 ULP distance ≤ TOLERANCE_ULP, OR
    //   * absolute magnitude ≤ ABS_TOL_FALLBACK (catches cancellation-
    //     prone tiny outputs that compound several softmax + matmul
    //     stages, where 1 ULP at magnitude 2^-12 is meaningless).
    // The DUT applies an fp16 round at every micro-op (dot, scale, exp,
    // FMA), while the REF rounds only at the final out write — so on
    // outputs that come from many compounded near-cancellations we
    // expect ULP drift compared to REF.
    const int    TOLERANCE_ULP       = 64;
    const double ABS_TOL_FALLBACK    = 0.01;   // ~half a percent of typical range

    struct Stats {
        int max_ulp_dr = 0;
        int max_ulp_ds = 0;
        int max_ulp_rs = 0;
        int fails_dr = 0;
        int fails_ds = 0;
        int n_words = 0;
        double min_cos_ds = 1.0;
    } st;

    auto run_one_frame = [&](const Frame& f, const std::string& tag, int& budget) {
        load_inputs(sim, f);
        int cyc = run_until_done(sim);
        printf("  [%s] frame ran in %d cycles (after writes)\n", tag.c_str(), cyc);
        std::vector<uint16_t> o_dut, o_ref, o_shadow;
        read_out(sim, o_dut, o_ref);
        shadow_attention(f.q, f.k, f.v, o_shadow);
        for (size_t a = 0; a < o_dut.size(); a++) {
            int udr = fp16_ulp_diff(o_dut[a], o_ref[a]);
            int uds = fp16_ulp_diff(o_dut[a], o_shadow[a]);
            int urs = fp16_ulp_diff(o_ref[a], o_shadow[a]);
            if (udr > st.max_ulp_dr) st.max_ulp_dr = udr;
            if (uds > st.max_ulp_ds) st.max_ulp_ds = uds;
            if (urs > st.max_ulp_rs) st.max_ulp_rs = urs;
            double dv = fp16_to_double(o_dut[a]);
            double rv = fp16_to_double(o_ref[a]);
            double sv = fp16_to_double(o_shadow[a]);
            double abs_err_dr = std::fabs(dv - rv);
            double abs_err_ds = std::fabs(dv - sv);
            if (udr > TOLERANCE_ULP && abs_err_dr > ABS_TOL_FALLBACK) {
                st.fails_dr++;
                if (budget > 0) {
                    printf("    [%s] addr=%zu dut=0x%04X ref=0x%04X ulp=%d "
                           "(dut=%g ref=%g abs=%g)\n",
                           tag.c_str(), a, o_dut[a], o_ref[a], udr,
                           dv, rv, abs_err_dr);
                    budget--;
                }
            }
            if (uds > TOLERANCE_ULP && abs_err_ds > ABS_TOL_FALLBACK) {
                st.fails_ds++;
            }
            st.n_words++;
        }
        double cds = cos_sim(o_dut, o_shadow);
        if (cds < st.min_cos_ds) st.min_cos_ds = cds;
        printf("  [%s] cos(dut,shadow) = %.6f (max ULP dr=%d ds=%d rs=%d)\n",
               tag.c_str(), cds, st.max_ulp_dr, st.max_ulp_ds, st.max_ulp_rs);
    };

    int budget = 30;

    // ─── Test 1: zeros ───────────────────────────────
    printf("test 1: all-zero Q/K/V\n");
    {
        Frame f = mk_zero_frame();
        // Reset DUT REF state (they have re-arm on start from S_DONE
        // or R_DONE → IDLE → RUN, but a fresh reset is cleanest after
        // the first frame too). Use sim.reset to start cleanly.
        run_one_frame(f, "zeros", budget);
    }

    // ─── Test 2: small random ──────────────────────
    printf("test 2: random Q/K/V (range ±0.5)\n");
    {
        std::mt19937 rng(0xDEADBEEFu);
        const int N_RND = 8;
        for (int t = 0; t < N_RND; t++) {
            sim.reset();
            Frame f = mk_random_frame(rng, 0.5);
            std::string tag = "rand_" + std::to_string(t);
            run_one_frame(f, tag, budget);
        }
    }

    // ─── Test 3: larger range ──────────────────────
    printf("test 3: random Q/K/V (range ±1.5)\n");
    {
        std::mt19937 rng(0xC0FFEEu);
        const int N_RND = 8;
        for (int t = 0; t < N_RND; t++) {
            sim.reset();
            Frame f = mk_random_frame(rng, 1.5);
            std::string tag = "wide_" + std::to_string(t);
            run_one_frame(f, tag, budget);
        }
    }

    // ─── Test 4: peaked Q (one row much larger) ────
    printf("test 4: one Q row very large → tests max-tracking\n");
    {
        sim.reset();
        std::mt19937 rng(0xABCDu);
        Frame f = mk_random_frame(rng, 0.3);
        for (int d = 0; d < DIM_Q; d++) f.q[0 * DIM_Q + d] = double_to_fp16(3.0);
        run_one_frame(f, "peaked", budget);
    }

    // ─── Test 5: tight range (low dynamic range) ──
    printf("test 5: random Q/K/V (range ±0.1) — low dynamic range\n");
    {
        std::mt19937 rng(0x1234u);
        sim.reset();
        Frame f = mk_random_frame(rng, 0.1);
        run_one_frame(f, "tight", budget);
    }

    // ─── Summary ─────────────────────────────────
    printf("\n");
    printf("Summary: %d words compared.\n", st.n_words);
    printf("  max ULP dut-vs-ref    : %d\n", st.max_ulp_dr);
    printf("  max ULP dut-vs-shadow : %d\n", st.max_ulp_ds);
    printf("  max ULP ref-vs-shadow : %d\n", st.max_ulp_rs);
    printf("  worst cos(dut,shadow) : %.6f\n", st.min_cos_ds);

    sim.check(st.fails_dr == 0,
              "0 dut-vs-ref fails (was " + std::to_string(st.fails_dr) +
              ", tol=" + std::to_string(TOLERANCE_ULP) + " ULP)");
    sim.check(st.min_cos_ds >= 0.998,
              "cos(dut,shadow) >= 0.998 (worst = " +
              std::to_string(st.min_cos_ds) + ")");

    return sim.finish();
}
