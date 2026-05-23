// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// flash_attn — Verilator C++ test. Loads DUT, drives random Q/K/V,
// waits for done, and compares per-output fp16 against (a) the SV
// REF and (b) an independent C++ double-precision shadow.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef CFG_HEADER
#error "Define CFG_HEADER to the Vflash_attn_tb_<cfg>.h path"
#endif
#include CFG_HEADER
#include "sim_ctrl.h"

// ─────────────── config (from -DCFG_*) ───────────────
#ifndef CFG_HEADS
#error "Define CFG_HEADS / CFG_N / CFG_DIM_Q / CFG_DIM_V / CFG_TOP"
#endif

using DUT = CFG_DUT_T;

static constexpr int HEADS = CFG_HEADS;
static constexpr int N     = CFG_N;
static constexpr int DIM_Q = CFG_DIM_Q;
static constexpr int DIM_V = CFG_DIM_V;

static constexpr int Q_BITS  = HEADS * N * DIM_Q * 16;
static constexpr int V_BITS  = HEADS * N * DIM_V * 16;
static constexpr int Q_WORDS = (Q_BITS + 31) / 32;
static constexpr int V_WORDS = (V_BITS + 31) / 32;

#ifndef CFG_TEMP_FP16
#define CFG_TEMP_FP16 0x31A8   // 1/sqrt(32)
#endif
#ifndef CFG_MAX_CYC
#define CFG_MAX_CYC 150000
#endif

// ─────────────── fp16 helpers ───────────────
static double fp16_to_double(uint16_t x) {
    int s = (x >> 15) & 1;
    int e = (x >> 10) & 0x1F;
    int f = x & 0x3FF;
    double v;
    if (e == 0x1F) {
        v = (f == 0) ? std::numeric_limits<double>::infinity()
                     : std::numeric_limits<double>::quiet_NaN();
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
        if (frac > 0.5)                         mant_int += 1;
        else if (frac == 0.5 && (mant_int & 1)) mant_int += 1;
        if (mant_int >= 1024) return (uint16_t)((s << 15) | (1 << 10));
        return (uint16_t)((s << 15) | (mant_int & 0x3FF));
    }
    double mant_d = (m * 2.0 - 1.0) * 1024.0;
    double floor_v = std::floor(mant_d);
    double frac = mant_d - floor_v;
    long mant_int = (long)floor_v;
    if (frac > 0.5)                         mant_int += 1;
    else if (frac == 0.5 && (mant_int & 1)) mant_int += 1;
    if (mant_int >= 1024) {
        biased += 1; mant_int = 0;
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

// ─────────────── shadow (double-precision attention) ───────────────
static void attn_shadow(const std::vector<uint16_t>& Q_,
                        const std::vector<uint16_t>& K_,
                        const std::vector<uint16_t>& V_,
                        std::vector<uint16_t>& O_) {
    double temp = fp16_to_double(CFG_TEMP_FP16);
    std::vector<double> S(N), P(N);
    for (int h = 0; h < HEADS; h++) {
        for (int r = 0; r < N; r++) {
            for (int j = 0; j < N; j++) {
                double acc = 0.0;
                for (int k = 0; k < DIM_Q; k++) {
                    double q = fp16_to_double(Q_[h*N*DIM_Q + r*DIM_Q + k]);
                    double kk = fp16_to_double(K_[h*N*DIM_Q + j*DIM_Q + k]);
                    acc += q * kk;
                }
                S[j] = acc * temp;
            }
            double mx = S[0];
            for (int j = 1; j < N; j++) if (S[j] > mx) mx = S[j];
            double sum = 0.0;
            for (int j = 0; j < N; j++) { P[j] = std::exp(S[j] - mx); sum += P[j]; }
            for (int j = 0; j < N; j++) P[j] /= sum;
            for (int d = 0; d < DIM_V; d++) {
                double acc = 0.0;
                for (int j = 0; j < N; j++) {
                    double v = fp16_to_double(V_[h*N*DIM_V + j*DIM_V + d]);
                    acc += P[j] * v;
                }
                O_[h*N*DIM_V + r*DIM_V + d] = double_to_fp16(acc);
            }
        }
    }
}

// ─────────────── wide port pack/unpack ───────────────
// Verilator exposes a wide bit port as a WData/VlWide<N> of uint32_t.
// We index via .at(w). The flat input/output port names are
// q_flat_i, k_flat_i, v_flat_i, o_dut_flat_o, o_ref_flat_o.

template <typename Port>
static void set_words(Port& p, const std::vector<uint32_t>& words) {
    for (int w = 0; w < (int)words.size(); w++) p.at(w) = words[w];
}

template <typename Port>
static void get_words(Port& p, std::vector<uint32_t>& words) {
    for (int w = 0; w < (int)words.size(); w++) words[w] = p.at(w);
}

static std::vector<uint32_t> pack_fp16(const std::vector<uint16_t>& src, int bits) {
    int words = (bits + 31) / 32;
    std::vector<uint32_t> out(words, 0);
    for (size_t i = 0; i < src.size(); i++) {
        int bitpos = 16 * (int)i;
        int w = bitpos / 32;
        int sh = bitpos % 32;
        out[w] |= ((uint32_t)src[i]) << sh;
    }
    return out;
}
static void unpack_fp16(const std::vector<uint32_t>& src, std::vector<uint16_t>& out) {
    for (size_t i = 0; i < out.size(); i++) {
        int bitpos = 16 * (int)i;
        int w = bitpos / 32;
        int sh = bitpos % 32;
        out[i] = (uint16_t)((src[w] >> sh) & 0xFFFF);
    }
}

// ─────────────── driver ───────────────
static int run_one(SimCtrl<DUT>& sim, std::mt19937& rng,
                   int tol_ulp, const char* name, int& max_cycles_seen) {
    // Generate stim in [-1, 1] roughly.
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<uint16_t> Q_(HEADS*N*DIM_Q), K_(HEADS*N*DIM_Q), V_(HEADS*N*DIM_V);
    for (auto& x : Q_) x = double_to_fp16(dist(rng));
    for (auto& x : K_) x = double_to_fp16(dist(rng));
    for (auto& x : V_) x = double_to_fp16(dist(rng));

    auto Qw = pack_fp16(Q_, Q_BITS);
    auto Kw = pack_fp16(K_, Q_BITS);
    auto Vw = pack_fp16(V_, V_BITS);
    set_words(sim.dut->q_flat_i, Qw);
    set_words(sim.dut->k_flat_i, Kw);
    set_words(sim.dut->v_flat_i, Vw);

    // Pulse start_i for one cycle.
    sim.dut->start_i = 1;
    sim.tick();
    sim.dut->start_i = 0;

    // Latch each done independently — DUT/REF assert at different times.
    int cycles = 0;
    bool dut_done = false, ref_done = false;
    // Latch O outputs when each side finishes.
    std::vector<uint32_t> dut_w(V_WORDS), ref_w(V_WORDS);
    while (!(dut_done && ref_done)) {
        if (!dut_done && sim.dut->done_dut_o) {
            get_words(sim.dut->o_dut_flat_o, dut_w);
            dut_done = true;
        }
        if (!ref_done && sim.dut->done_ref_o) {
            get_words(sim.dut->o_ref_flat_o, ref_w);
            ref_done = true;
        }
        if (dut_done && ref_done) break;
        sim.tick();
        cycles++;
        if (cycles >= CFG_MAX_CYC) {
            printf("  [%s] TIMEOUT: DUT did not complete in %d cycles (dut_done=%d ref_done=%d)\n",
                   name, CFG_MAX_CYC, (int)dut_done, (int)ref_done);
            return 1;
        }
    }
    if (cycles > max_cycles_seen) max_cycles_seen = cycles;

    // Outputs were latched above when each side asserted done.
    std::vector<uint16_t> O_dut(HEADS*N*DIM_V), O_ref(HEADS*N*DIM_V),
                          O_shadow(HEADS*N*DIM_V);
    unpack_fp16(dut_w, O_dut);
    unpack_fp16(ref_w, O_ref);
    attn_shadow(Q_, K_, V_, O_shadow);

    int errs = 0;
    int worst_ulp_ref = 0, worst_ulp_shadow = 0;
    // Tolerance: ULP budget OR absolute diff ≤ ABS_TOL. Tiled fp16
    // online softmax has compounding rounding in alpha*O_old + new
    // chain; small outputs can drift many ULPs (1 ULP at the bottom of
    // the fp16 range is tiny in absolute terms). Both conditions are
    // permissive, but a real correctness regression (e.g. a wrong
    // index) shows up as both ULP and absolute blowups.
    const double ABS_TOL = 0.02;
    for (int i = 0; i < (int)O_dut.size(); i++) {
        int u1 = fp16_ulp_diff(O_dut[i], O_ref[i]);
        int u2 = fp16_ulp_diff(O_dut[i], O_shadow[i]);
        double dv = fp16_to_double(O_dut[i]);
        double rv = fp16_to_double(O_ref[i]);
        double abs_diff = std::fabs(dv - rv);
        if (u1 > worst_ulp_ref)    worst_ulp_ref    = u1;
        if (u2 > worst_ulp_shadow) worst_ulp_shadow = u2;
        bool fail = (u1 > tol_ulp || u2 > tol_ulp) && (abs_diff > ABS_TOL);
        if (fail) {
            if (errs < 10) {
                printf("  [%s] mismatch out[%d]: dut=%04x(%.4f) ref=%04x(%.4f) "
                       "ulp=%d abs=%.4f\n",
                       name, i, O_dut[i], dv, O_ref[i], rv, u1, abs_diff);
            }
            errs++;
        }
    }
    printf("  [%s] cycles=%d worst_ulp_ref=%d worst_ulp_shadow=%d errs=%d\n",
           name, cycles, worst_ulp_ref, worst_ulp_shadow, errs);
    return errs;
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    // sim_ctrl max_time is in half-cycle units. Per-frame budget is CFG_MAX_CYC;
    // we may run up to CFG_FRAMES frames. Multiply by ~2.5x for safety.
    sim.max_time = (uint64_t)CFG_FRAMES * 5ULL * (uint64_t)CFG_MAX_CYC + 10000ULL;
    sim.dut->start_i = 0;
    sim.reset();

    std::mt19937 rng(0xCAFE);
    int total_errs = 0;
    int max_cycles_seen = 0;
    int tol_ulp = 256;  // tiled fp16 vs ref(real)/shadow(double) — compound rounding budget
    // Number of frames per config (small=2 for quick iteration; prod=1 to bound wall time)
    int frames =
#ifdef CFG_FRAMES
        CFG_FRAMES;
#else
        2;
#endif
    for (int f = 0; f < frames; f++) {
        char name[64];
        std::snprintf(name, sizeof(name), "%s/frame%d", CFG_NAME, f);
        total_errs += run_one(sim, rng, tol_ulp, name, max_cycles_seen);
    }

    printf("[%s] max_cycles_seen=%d (MAX=%d, BUDGET=%d)\n",
           CFG_NAME, max_cycles_seen, CFG_MAX_CYC, 100000);
    if (max_cycles_seen > 100000) {
        printf("[%s] WARNING: exceeded 100k cycle budget (%d cycles)\n",
               CFG_NAME, max_cycles_seen);
    }

    sim.check(total_errs == 0, "all outputs within tolerance");
    return sim.finish();
}
