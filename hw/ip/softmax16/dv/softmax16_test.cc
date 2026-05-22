// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// softmax16 — Verilator test. Drives a 16-lane fp16 vector into the
// DUT/REF lockstep TB, and compares per-lane outputs (a) against the
// SV REF and (b) against an independent C++ double-precision shadow.
// All comparisons are in fp16 ULPs against a tolerance.
//
// Independent shadow: uses pure double-precision math (max → exp →
// sum → divide), then rounds each output to fp16 with RNE. This shares
// no code with the SV REF or DUT, so they cross-check each other.

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

#include "Vsoftmax16_tb.h"
#include "sim_ctrl.h"

using DUT = Vsoftmax16_tb;

// ───────────────────────── fp16 helpers ───────────────────────────

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
    double m = std::frexp(av, &e);  // m in [0.5, 1), av = m * 2^e
    // Reform to: av = (m * 2) * 2^(e-1), with leading 1 at integer bit.
    int unbiased = e - 1;
    int biased = unbiased + 15;
    if (biased >= 31) return (uint16_t)((s << 15) | 0x7C00);
    if (biased <= 0) {
        // subnormal: av = mant_int * 2^-24, round-to-nearest-even
        double scaled = av * (double)(1 << 24);
        double floor_v = std::floor(scaled);
        double frac = scaled - floor_v;
        long mant_int = (long)floor_v;
        if (frac > 0.5)                            mant_int += 1;
        else if (frac == 0.5 && (mant_int & 1))    mant_int += 1;
        if (mant_int >= 1024) return (uint16_t)((s << 15) | (1 << 10));
        return (uint16_t)((s << 15) | (mant_int & 0x3FF));
    }
    // Normal: mant = (m*2 - 1) * 1024, RNE
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

// Sign-magnitude fp16 → ordered signed integer suitable for ULP-diff.
static int32_t fp16_ordered(uint16_t x) {
    if (x & 0x8000) return -(int32_t)(x & 0x7FFF);
    return (int32_t)x;
}

static int32_t fp16_ulp_diff(uint16_t a, uint16_t b) {
    int32_t da = fp16_ordered(a);
    int32_t db = fp16_ordered(b);
    int32_t d  = da - db;
    return d < 0 ? -d : d;
}

// ───────────────────────── shadow softmax ─────────────────────────

static void softmax_shadow(const std::array<uint16_t, 16>& x_fp16,
                           std::array<uint16_t, 16>& y_fp16) {
    double x[16];
    for (int i = 0; i < 16; i++) x[i] = fp16_to_double(x_fp16[i]);
    double mx = x[0];
    for (int i = 1; i < 16; i++) if (x[i] > mx) mx = x[i];
    double e[16];
    double sum = 0.0;
    for (int i = 0; i < 16; i++) { e[i] = std::exp(x[i] - mx); sum += e[i]; }
    for (int i = 0; i < 16; i++) y_fp16[i] = double_to_fp16(e[i] / sum);
}

// ───────────────────────── wide-port I/O ──────────────────────────
// Verilator exposes a 256-bit port as a VlWide<8> (array of 8 uint32_t).
// We access via the .data() pointer member or via direct subscript.

static void set_x_flat(SimCtrl<DUT>& s, const std::array<uint16_t, 16>& x) {
    // x_flat_i[16*i +: 16] — lane 0 in LSBs.
    uint32_t words[8] = {0};
    for (int i = 0; i < 16; i++) {
        int bitpos = 16 * i;
        int w      = bitpos / 32;
        int sh     = bitpos % 32;
        words[w]  |= ((uint32_t)x[i]) << sh;
    }
    for (int w = 0; w < 8; w++) s.dut->x_flat_i.at(w) = words[w];
}

static void get_y_flat(SimCtrl<DUT>& s, bool from_dut,
                       std::array<uint16_t, 16>& y) {
    uint32_t words[8];
    for (int w = 0; w < 8; w++)
        words[w] = from_dut ? s.dut->y_dut_flat_o.at(w)
                            : s.dut->y_ref_flat_o.at(w);
    for (int i = 0; i < 16; i++) {
        int bitpos = 16 * i;
        int w      = bitpos / 32;
        int sh     = bitpos % 32;
        y[i] = (uint16_t)((words[w] >> sh) & 0xFFFF);
    }
}

// ───────────────────────── driver ────────────────────────────────

static const int LATENCY = 12;
static const int TOLERANCE_ULP = 32;
// Tolerance budget (per fp16 output lane, sign-magnitude ULP distance):
//   exp LUT (1024 entries, step 1/64)         : ~8 ULP
//   reciprocal LUT (1024-entry mantissa)      : ~1 ULP
//   m - x fp16 subtract                       : ~1 ULP
//   sum tree (4 fp16-add stages)              : ~4 ULP per term
//   final fp16 FMA multiply                   : ~1 ULP
//   compounded over per-lane × per-sample     : up to ~25 ULP observed.
// 32 ULP leaves comfortable headroom while still being tight enough to
// catch correctness regressions.

struct DriveResult {
    std::array<uint16_t, 16> y_dut;
    std::array<uint16_t, 16> y_ref;
    std::array<uint16_t, 16> y_shadow;
    bool valid_dut;
    bool valid_ref;
};

// Drive one vector through the pipeline and return outputs after LATENCY+1 cycles.
static DriveResult drive_one(SimCtrl<DUT>& sim,
                             const std::array<uint16_t, 16>& x) {
    set_x_flat(sim, x);
    sim.dut->valid_i = 1;
    sim.tick();
    sim.dut->valid_i = 0;
    // Zero inputs after to avoid X-prop noise on later cycles' computations.
    std::array<uint16_t, 16> zero{}; set_x_flat(sim, zero);
    for (int k = 0; k < LATENCY - 1; k++) sim.tick();
    // Now valid_dut_o should be high.
    DriveResult r;
    r.valid_dut = (sim.dut->valid_dut_o != 0);
    r.valid_ref = (sim.dut->valid_ref_o != 0);
    get_y_flat(sim, true,  r.y_dut);
    get_y_flat(sim, false, r.y_ref);
    softmax_shadow(x, r.y_shadow);
    return r;
}

// ───────────────────────── pretty printers ───────────────────────

static std::string hex16(uint16_t v) {
    char b[8]; std::snprintf(b, sizeof(b), "0x%04X", v); return std::string(b);
}

static std::string vec_to_str(const std::array<uint16_t, 16>& v) {
    std::string s = "[";
    for (int i = 0; i < 16; i++) {
        s += hex16(v[i]);
        if (i != 15) s += ",";
    }
    s += "]";
    return s;
}

// ───────────────────────── checks ────────────────────────────────

struct Stats {
    int  max_ulp_dut_ref = 0;
    int  max_ulp_dut_shadow = 0;
    int  max_ulp_ref_shadow = 0;
    int  fails_dut_ref = 0;
    int  fails_dut_shadow = 0;
    double max_sum_err = 0.0;
};

static void compare_vec(const DriveResult& r, Stats& st,
                        const std::string& tag, int& print_budget) {
    for (int i = 0; i < 16; i++) {
        int u_dr = fp16_ulp_diff(r.y_dut[i], r.y_ref[i]);
        int u_ds = fp16_ulp_diff(r.y_dut[i], r.y_shadow[i]);
        int u_rs = fp16_ulp_diff(r.y_ref[i], r.y_shadow[i]);
        if (u_dr > st.max_ulp_dut_ref)        st.max_ulp_dut_ref = u_dr;
        if (u_ds > st.max_ulp_dut_shadow)     st.max_ulp_dut_shadow = u_ds;
        if (u_rs > st.max_ulp_ref_shadow)     st.max_ulp_ref_shadow = u_rs;
        if (u_dr > TOLERANCE_ULP) {
            st.fails_dut_ref++;
            if (print_budget > 0) {
                printf("  [%s] lane %d dut=%s ref=%s ulp_dr=%d\n",
                       tag.c_str(), i,
                       hex16(r.y_dut[i]).c_str(), hex16(r.y_ref[i]).c_str(), u_dr);
                print_budget--;
            }
        }
        if (u_ds > TOLERANCE_ULP) {
            st.fails_dut_shadow++;
            if (print_budget > 0) {
                printf("  [%s] lane %d dut=%s shad=%s ulp_ds=%d\n",
                       tag.c_str(), i,
                       hex16(r.y_dut[i]).c_str(), hex16(r.y_shadow[i]).c_str(), u_ds);
                print_budget--;
            }
        }
    }
    // Sum of outputs ≈ 1.0 (within tolerance)
    double dsum = 0.0;
    for (int i = 0; i < 16; i++) dsum += fp16_to_double(r.y_dut[i]);
    double err = std::fabs(dsum - 1.0);
    if (err > st.max_sum_err) st.max_sum_err = err;
}

// Build fp16 from sign/exponent/frac for stimulus.
static uint16_t mkfp16(int s, int e, int f) {
    return (uint16_t)(((s & 1) << 15) | ((e & 0x1F) << 10) | (f & 0x3FF));
}

// Random fp16 in roughly [-8, 8] range — softmax inputs are normally bounded.
static uint16_t random_softmax_logit(std::mt19937& rng) {
    // double in [-8, 8]
    std::uniform_real_distribution<double> ud(-8.0, 8.0);
    return double_to_fp16(ud(rng));
}

// ───────────────────────── main ──────────────────────────────────

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 6000000000ull;

    // init
    sim.dut->valid_i = 0;
    {
        std::array<uint16_t, 16> z{}; set_x_flat(sim, z);
    }
    sim.reset();

    // ─── Test 1: reset behaviour ───────────────────────
    printf("test 1: reset behaviour\n");
    sim.check(sim.dut->valid_dut_o == 0, "valid_dut low after reset");
    sim.check(sim.dut->valid_ref_o == 0, "valid_ref low after reset");

    // ─── Test 2: directed all-equal → uniform 1/16 ─────
    printf("test 2: all-equal logits → uniform output\n");
    {
        // 1/16 ≈ 0.0625 → fp16 0x2C00
        uint16_t target = double_to_fp16(1.0 / 16.0);
        std::array<uint16_t, 16> x;
        x.fill(mkfp16(0, 15, 0));   // 1.0 in each lane
        DriveResult r = drive_one(sim, x);
        sim.check(r.valid_dut, "dut valid after pipeline (all-equal)");
        sim.check(r.valid_ref, "ref valid after pipeline (all-equal)");
        int max_ulp = 0;
        for (int i = 0; i < 16; i++) {
            int u = fp16_ulp_diff(r.y_dut[i], target);
            if (u > max_ulp) max_ulp = u;
        }
        sim.check(max_ulp <= TOLERANCE_ULP,
                  "all-equal: max ULP to 1/16 = " + std::to_string(max_ulp));
    }

    // ─── Test 3: one-hot dominant ───────────────────────
    printf("test 3: one-hot dominant logit\n");
    {
        std::array<uint16_t, 16> x;
        // -8.0 everywhere, +8.0 in lane 7. e^16 ~ 8.9e6 → fp16 saturates,
        // but ratio is ~ 1 in lane 7, ~exp(-16)≈0 in others.
        x.fill(double_to_fp16(-8.0));
        x[7] = double_to_fp16(8.0);
        DriveResult r = drive_one(sim, x);
        sim.check(r.valid_dut, "dut valid (one-hot)");
        Stats st; int budget = 10;
        compare_vec(r, st, "one-hot", budget);
        sim.check(st.fails_dut_ref == 0,
                  "one-hot: 0 lane fails dut-vs-ref (max " +
                  std::to_string(st.max_ulp_dut_ref) + " ULP)");
        sim.check(st.fails_dut_shadow == 0,
                  "one-hot: 0 lane fails dut-vs-shadow (max " +
                  std::to_string(st.max_ulp_dut_shadow) + " ULP)");
    }

    // ─── Test 4: all very negative (still well-defined relative softmax) ──
    printf("test 4: all very negative logits\n");
    {
        std::array<uint16_t, 16> x;
        // Logits at -10.0 .. -10.0 with one at -10.5 etc. After subtracting
        // max, d ∈ [0, 0.5] for most lanes → exp LUT well-conditioned.
        std::mt19937 rng(0xABCDEF01u);
        std::uniform_real_distribution<double> jd(-10.5, -10.0);
        for (int i = 0; i < 16; i++) x[i] = double_to_fp16(jd(rng));
        DriveResult r = drive_one(sim, x);
        Stats st; int budget = 10;
        compare_vec(r, st, "all-neg", budget);
        sim.check(st.fails_dut_ref == 0,
                  "all-neg: 0 lane fails dut-vs-ref (max " +
                  std::to_string(st.max_ulp_dut_ref) + " ULP)");
        sim.check(st.fails_dut_shadow == 0,
                  "all-neg: 0 lane fails dut-vs-shadow (max " +
                  std::to_string(st.max_ulp_dut_shadow) + " ULP)");
        sim.check(st.max_sum_err < 0.05,
                  "all-neg: |sum-1| = " + std::to_string(st.max_sum_err));
    }

    // ─── Test 5: zeros everywhere → uniform 1/16 ─────────
    printf("test 5: all zeros\n");
    {
        std::array<uint16_t, 16> x{};
        DriveResult r = drive_one(sim, x);
        Stats st; int budget = 10;
        compare_vec(r, st, "zeros", budget);
        sim.check(st.fails_dut_ref == 0,
                  "zeros: 0 lane fails dut-vs-ref (max " +
                  std::to_string(st.max_ulp_dut_ref) + " ULP)");
        sim.check(st.max_sum_err < 0.05,
                  "zeros: |sum-1| = " + std::to_string(st.max_sum_err));
    }

    // ─── Test 6: gradient logits ─────────────────────────
    printf("test 6: gradient logits 0,1,2,...,15\n");
    {
        std::array<uint16_t, 16> x;
        for (int i = 0; i < 16; i++) x[i] = double_to_fp16((double)i * 0.5);
        DriveResult r = drive_one(sim, x);
        Stats st; int budget = 10;
        compare_vec(r, st, "grad", budget);
        sim.check(st.fails_dut_ref == 0,
                  "grad: 0 lane fails dut-vs-ref (max " +
                  std::to_string(st.max_ulp_dut_ref) + " ULP)");
        sim.check(st.max_sum_err < 0.05,
                  "grad: |sum-1| = " + std::to_string(st.max_sum_err));
    }

    // ─── Test 7: randomized stress (5000 vectors) ───────
    printf("test 7: 5000 random fp16 vectors\n");
    {
        std::mt19937 rng(0xFA1A4F00u);
        Stats st; int budget = 20;
        const int N = 5000;
        for (int t = 0; t < N; t++) {
            std::array<uint16_t, 16> x;
            for (int i = 0; i < 16; i++) x[i] = random_softmax_logit(rng);
            DriveResult r = drive_one(sim, x);
            compare_vec(r, st, "rand", budget);
        }
        printf("  max ULP dut-vs-ref    : %d\n", st.max_ulp_dut_ref);
        printf("  max ULP dut-vs-shadow : %d\n", st.max_ulp_dut_shadow);
        printf("  max ULP ref-vs-shadow : %d\n", st.max_ulp_ref_shadow);
        printf("  max |sum-1|           : %g\n", st.max_sum_err);
        sim.check(st.fails_dut_ref == 0,
                  "rand: 0 lane fails dut-vs-ref (was " +
                  std::to_string(st.fails_dut_ref) + ")");
        sim.check(st.fails_dut_shadow == 0,
                  "rand: 0 lane fails dut-vs-shadow (was " +
                  std::to_string(st.fails_dut_shadow) + ")");
        sim.check(st.max_sum_err < 0.05,
                  "rand: max |sum-1| = " + std::to_string(st.max_sum_err));
    }

    // ─── Test 8: tighter range (±2) — DFL-typical inputs ─
    printf("test 8: 1000 tight-range (±2) vectors\n");
    {
        std::mt19937 rng(0xC0FFEEu);
        std::uniform_real_distribution<double> ud(-2.0, 2.0);
        Stats st; int budget = 10;
        const int N = 1000;
        for (int t = 0; t < N; t++) {
            std::array<uint16_t, 16> x;
            for (int i = 0; i < 16; i++) x[i] = double_to_fp16(ud(rng));
            DriveResult r = drive_one(sim, x);
            compare_vec(r, st, "tight", budget);
        }
        printf("  max ULP dut-vs-ref    : %d\n", st.max_ulp_dut_ref);
        printf("  max ULP dut-vs-shadow : %d\n", st.max_ulp_dut_shadow);
        printf("  max |sum-1|           : %g\n", st.max_sum_err);
        sim.check(st.fails_dut_ref == 0,
                  "tight: 0 lane fails dut-vs-ref");
        sim.check(st.fails_dut_shadow == 0,
                  "tight: 0 lane fails dut-vs-shadow");
        sim.check(st.max_sum_err < 0.02,
                  "tight: max |sum-1| = " + std::to_string(st.max_sum_err));
    }

    return sim.finish();
}
