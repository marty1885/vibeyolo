// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// box_decode — Verilator test. Drives the 4×16 fp16 probability vectors
// plus (cx, cy, stride) into the DUT/REF lockstep TB and compares each
// of x1, y1, x2, y2 against (a) the SV REF and (b) an independent C++
// double-precision shadow. All comparisons are in fp16 ULPs against a
// fixed tolerance.

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

#include "Vbox_decode_tb.h"
#include "sim_ctrl.h"

using DUT = Vbox_decode_tb;

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
    int32_t da = fp16_ordered(a);
    int32_t db = fp16_ordered(b);
    int32_t d  = da - db;
    return d < 0 ? -d : d;
}

// ───────────────────────── shadow box decode ──────────────────────

struct Box { uint16_t x1, y1, x2, y2; };

struct ShadowResult {
    Box     box;
    double  cx_c, cy_c;
    double  dl_S, dt_S, dr_S, db_S;
};

static ShadowResult box_decode_shadow(const std::array<uint16_t, 16>& pl,
                                      const std::array<uint16_t, 16>& pt,
                                      const std::array<uint16_t, 16>& pr,
                                      const std::array<uint16_t, 16>& pb,
                                      int16_t cx, int16_t cy, int16_t stride) {
    auto dot = [](const std::array<uint16_t, 16>& p) {
        double s = 0.0;
        for (int i = 0; i < 16; i++) s += fp16_to_double(p[i]) * (double)i;
        return s;
    };
    double dl = dot(pl), dt = dot(pt), dr = dot(pr), db = dot(pb);
    double S  = (double)stride;
    double cx_c = ((double)cx + 0.5) * S;
    double cy_c = ((double)cy + 0.5) * S;
    ShadowResult r;
    r.box.x1 = double_to_fp16(cx_c - dl * S);
    r.box.y1 = double_to_fp16(cy_c - dt * S);
    r.box.x2 = double_to_fp16(cx_c + dr * S);
    r.box.y2 = double_to_fp16(cy_c + db * S);
    r.cx_c = cx_c; r.cy_c = cy_c;
    r.dl_S = dl * S; r.dt_S = dt * S; r.dr_S = dr * S; r.db_S = db * S;
    return r;
}

// ───────────────────────── wide-port I/O ──────────────────────────

static void set_p_flat(SimCtrl<DUT>& s, int which,
                       const std::array<uint16_t, 16>& p) {
    uint32_t words[8] = {0};
    for (int i = 0; i < 16; i++) {
        int bitpos = 16 * i;
        int w      = bitpos / 32;
        int sh     = bitpos % 32;
        words[w]  |= ((uint32_t)p[i]) << sh;
    }
    auto write = [&](auto& port) {
        for (int w = 0; w < 8; w++) port.at(w) = words[w];
    };
    switch (which) {
        case 0: write(s.dut->p_l_flat_i); break;
        case 1: write(s.dut->p_t_flat_i); break;
        case 2: write(s.dut->p_r_flat_i); break;
        case 3: write(s.dut->p_b_flat_i); break;
    }
}

// ───────────────────────── driver ────────────────────────────────

static const int LATENCY = 7;
static const int TOLERANCE_ULP = 16;
// Tolerance scheme — per-output-coordinate:
//   • If |dut| and |ref| are both ≥ FP16-representable order ~1, the
//     dominant error source is the 4-stage fp16 add tree (≈4 ULP) plus
//     two final fp16_fma stages (≈2 ULP), so ~16 ULP is enough.
//   • The xyxy formula `cx_center − d*stride` is catastrophically
//     cancellation-prone when the box edge passes near the cell center.
//     In that regime, both operands have magnitude up to ~cx_center
//     (which can be several thousand) and their difference can be near
//     zero. The fp16 ULP of the *larger* operand is what matters, not
//     the ULP of the (small) result. We translate that into a per-sample
//     absolute tolerance:
//        abs_tol = 8 * fp16_step(max(|cx_center|, |d*stride|))
//     and pass when either the ULP distance is within 16 OR the absolute
//     difference is within abs_tol.
//   The SV ref does the same real-arithmetic computation, then rounds
//   to fp16 at the output — so ref-vs-shadow is exact, and dut-vs-ref
//   measures only the DUT's fp16-chain accumulation error.

// fp16 ULP at the given magnitude (in real units): for normal values,
// step = 2^(exp_unbiased - 10). For tiny values use subnormal step 2^-24.
static double fp16_step_at(double v) {
    double av = std::fabs(v);
    if (av == 0.0) return std::ldexp(1.0, -24);
    int e;
    std::frexp(av, &e);                 // av ∈ [2^(e-1), 2^e)
    int unbiased = e - 1;
    if (unbiased < -14) unbiased = -14; // subnormal floor
    return std::ldexp(1.0, unbiased - 10);
}

struct DriveResult {
    Box  dut;
    Box  ref;
    Box  shadow;
    bool valid_dut;
    bool valid_ref;
    // Per-coord scale info for catastrophic-cancellation aware tolerance.
    // scale[i] is the magnitude of the largest operand entering the final
    // (cx_center ± d*stride) combine for coordinate i.
    double scale[4];
};

// Drive one input. The DUT/REF have LATENCY=7 cycles before valid_o is
// asserted. We use a single-shot scheme: present input one cycle with
// valid_i=1, then tick 6 more cycles → at end of the 7th tick the output
// is valid.
static DriveResult drive_one(SimCtrl<DUT>& sim,
                             const std::array<uint16_t, 16>& pl,
                             const std::array<uint16_t, 16>& pt,
                             const std::array<uint16_t, 16>& pr,
                             const std::array<uint16_t, 16>& pb,
                             int16_t cx, int16_t cy, int16_t stride) {
    set_p_flat(sim, 0, pl);
    set_p_flat(sim, 1, pt);
    set_p_flat(sim, 2, pr);
    set_p_flat(sim, 3, pb);
    sim.dut->cx_i     = (uint16_t)cx;
    sim.dut->cy_i     = (uint16_t)cy;
    sim.dut->stride_i = (uint16_t)stride;
    sim.dut->valid_i  = 1;
    sim.tick();
    sim.dut->valid_i  = 0;
    std::array<uint16_t, 16> z{};
    set_p_flat(sim, 0, z);
    set_p_flat(sim, 1, z);
    set_p_flat(sim, 2, z);
    set_p_flat(sim, 3, z);
    for (int k = 0; k < LATENCY - 1; k++) sim.tick();
    DriveResult r;
    r.valid_dut = (sim.dut->valid_dut_o != 0);
    r.valid_ref = (sim.dut->valid_ref_o != 0);
    r.dut = {(uint16_t)sim.dut->x1_dut_o,
             (uint16_t)sim.dut->y1_dut_o,
             (uint16_t)sim.dut->x2_dut_o,
             (uint16_t)sim.dut->y2_dut_o};
    r.ref = {(uint16_t)sim.dut->x1_ref_o,
             (uint16_t)sim.dut->y1_ref_o,
             (uint16_t)sim.dut->x2_ref_o,
             (uint16_t)sim.dut->y2_ref_o};
    ShadowResult sh = box_decode_shadow(pl, pt, pr, pb, cx, cy, stride);
    r.shadow = sh.box;
    r.scale[0] = std::max(std::fabs(sh.cx_c), std::fabs(sh.dl_S));
    r.scale[1] = std::max(std::fabs(sh.cy_c), std::fabs(sh.dt_S));
    r.scale[2] = std::max(std::fabs(sh.cx_c), std::fabs(sh.dr_S));
    r.scale[3] = std::max(std::fabs(sh.cy_c), std::fabs(sh.db_S));
    return r;
}

// ───────────────────────── pretty printers ───────────────────────

static std::string hex16(uint16_t v) {
    char b[8]; std::snprintf(b, sizeof(b), "0x%04X", v); return std::string(b);
}

// ───────────────────────── checks ────────────────────────────────

struct Stats {
    int max_ulp_dut_ref    = 0;
    int max_ulp_dut_shadow = 0;
    int max_ulp_ref_shadow = 0;
    int fails_dut_ref      = 0;
    int fails_dut_shadow   = 0;
};

static void compare_box(const DriveResult& r, Stats& st,
                        const std::string& tag, int& print_budget) {
    const char* names[4] = {"x1", "y1", "x2", "y2"};
    uint16_t dut_arr[4] = {r.dut.x1, r.dut.y1, r.dut.x2, r.dut.y2};
    uint16_t ref_arr[4] = {r.ref.x1, r.ref.y1, r.ref.x2, r.ref.y2};
    uint16_t shd_arr[4] = {r.shadow.x1, r.shadow.y1, r.shadow.x2, r.shadow.y2};
    for (int i = 0; i < 4; i++) {
        int u_dr = fp16_ulp_diff(dut_arr[i], ref_arr[i]);
        int u_ds = fp16_ulp_diff(dut_arr[i], shd_arr[i]);
        int u_rs = fp16_ulp_diff(ref_arr[i], shd_arr[i]);
        if (u_dr > st.max_ulp_dut_ref)        st.max_ulp_dut_ref = u_dr;
        if (u_ds > st.max_ulp_dut_shadow)     st.max_ulp_dut_shadow = u_ds;
        if (u_rs > st.max_ulp_ref_shadow)     st.max_ulp_ref_shadow = u_rs;
        // Absolute-error tolerance derived from operand magnitudes
        // (catastrophic cancellation: error scales with the larger operand).
        double abs_tol = 8.0 * fp16_step_at(r.scale[i]);
        double dut_v   = fp16_to_double(dut_arr[i]);
        double ref_v   = fp16_to_double(ref_arr[i]);
        double shd_v   = fp16_to_double(shd_arr[i]);
        bool ok_dr = (u_dr <= TOLERANCE_ULP) || (std::fabs(dut_v - ref_v) <= abs_tol);
        bool ok_ds = (u_ds <= TOLERANCE_ULP) || (std::fabs(dut_v - shd_v) <= abs_tol);
        if (!ok_dr) {
            st.fails_dut_ref++;
            if (print_budget > 0) {
                printf("  [%s] %s dut=%s ref=%s ulp=%d scale=%.2f abs_err=%.4g abs_tol=%.4g\n",
                       tag.c_str(), names[i],
                       hex16(dut_arr[i]).c_str(), hex16(ref_arr[i]).c_str(), u_dr,
                       r.scale[i], std::fabs(dut_v - ref_v), abs_tol);
                print_budget--;
            }
        }
        if (!ok_ds) {
            st.fails_dut_shadow++;
            if (print_budget > 0) {
                printf("  [%s] %s dut=%s shad=%s ulp=%d scale=%.2f abs_err=%.4g abs_tol=%.4g\n",
                       tag.c_str(), names[i],
                       hex16(dut_arr[i]).c_str(), hex16(shd_arr[i]).c_str(), u_ds,
                       r.scale[i], std::fabs(dut_v - shd_v), abs_tol);
                print_budget--;
            }
        }
    }
}

// ───────────────────────── stimulus generators ───────────────────

// Build a probability vector by softmaxing random logits with double
// precision. The resulting fp16 values approximately sum to 1.0 — this is
// what the upstream softmax16 would emit.
static void rand_prob_vec(std::mt19937& rng, std::array<uint16_t, 16>& p,
                          double scale = 4.0) {
    std::uniform_real_distribution<double> ud(-scale, scale);
    double logits[16];
    double mx = -1e30;
    for (int i = 0; i < 16; i++) { logits[i] = ud(rng); if (logits[i] > mx) mx = logits[i]; }
    double sum = 0.0;
    double e[16];
    for (int i = 0; i < 16; i++) { e[i] = std::exp(logits[i] - mx); sum += e[i]; }
    for (int i = 0; i < 16; i++) p[i] = double_to_fp16(e[i] / sum);
}

// One-hot at bin k → d = k.
static void onehot_vec(std::array<uint16_t, 16>& p, int k) {
    p.fill(0);
    p[k] = double_to_fp16(1.0);
}

// Uniform 1/16 everywhere → d = sum(i)/16 = 7.5.
static void uniform_vec(std::array<uint16_t, 16>& p) {
    uint16_t v = double_to_fp16(1.0 / 16.0);
    p.fill(v);
}

// ───────────────────────── main ──────────────────────────────────

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 6000000000ull;

    sim.dut->valid_i = 0;
    sim.dut->cx_i = 0; sim.dut->cy_i = 0; sim.dut->stride_i = 8;
    {
        std::array<uint16_t, 16> z{};
        set_p_flat(sim, 0, z); set_p_flat(sim, 1, z);
        set_p_flat(sim, 2, z); set_p_flat(sim, 3, z);
    }
    sim.reset();

    // ─── Test 1: reset ──────────────────────────────────
    printf("test 1: reset behaviour\n");
    sim.check(sim.dut->valid_dut_o == 0, "valid_dut low after reset");
    sim.check(sim.dut->valid_ref_o == 0, "valid_ref low after reset");

    // ─── Test 2: uniform → d=7.5 ────────────────────────
    printf("test 2: uniform probabilities → d=7.5 each side\n");
    {
        std::array<uint16_t, 16> pl, pt, pr, pb;
        uniform_vec(pl); uniform_vec(pt); uniform_vec(pr); uniform_vec(pb);
        int16_t cx = 5, cy = 7, stride = 8;
        DriveResult r = drive_one(sim, pl, pt, pr, pb, cx, cy, stride);
        sim.check(r.valid_dut, "valid_dut after pipeline (uniform)");
        sim.check(r.valid_ref, "valid_ref after pipeline (uniform)");
        Stats st; int budget = 10;
        compare_box(r, st, "uniform", budget);
        sim.check(st.fails_dut_ref == 0,
                  "uniform: 0 fails dut-vs-ref (max " + std::to_string(st.max_ulp_dut_ref) + " ULP)");
        sim.check(st.fails_dut_shadow == 0,
                  "uniform: 0 fails dut-vs-shadow (max " + std::to_string(st.max_ulp_dut_shadow) + " ULP)");
        // Expected: cx_c = 5.5*8 = 44, d=7.5, x1 = 44 - 60 = -16, x2 = 44 + 60 = 104
        double x1_exp = (5.0 + 0.5 - 7.5) * 8.0;
        double x2_exp = (5.0 + 0.5 + 7.5) * 8.0;
        uint16_t x1_e = double_to_fp16(x1_exp);
        uint16_t x2_e = double_to_fp16(x2_exp);
        sim.check(fp16_ulp_diff(r.dut.x1, x1_e) <= TOLERANCE_ULP,
                  "uniform: x1 close to (-16)  got=" + hex16(r.dut.x1) + " exp=" + hex16(x1_e));
        sim.check(fp16_ulp_diff(r.dut.x2, x2_e) <= TOLERANCE_ULP,
                  "uniform: x2 close to (104)  got=" + hex16(r.dut.x2) + " exp=" + hex16(x2_e));
    }

    // ─── Test 3: one-hot at i=0 → d=0 ───────────────────
    printf("test 3: one-hot at i=0 → d=0 (box collapses to center)\n");
    {
        std::array<uint16_t, 16> pl, pt, pr, pb;
        onehot_vec(pl, 0); onehot_vec(pt, 0); onehot_vec(pr, 0); onehot_vec(pb, 0);
        int16_t cx = 10, cy = 20, stride = 16;
        DriveResult r = drive_one(sim, pl, pt, pr, pb, cx, cy, stride);
        Stats st; int budget = 10;
        compare_box(r, st, "onehot0", budget);
        sim.check(st.fails_dut_ref == 0,
                  "onehot0: 0 fails dut-vs-ref (max " + std::to_string(st.max_ulp_dut_ref) + " ULP)");
        sim.check(st.fails_dut_shadow == 0,
                  "onehot0: 0 fails dut-vs-shadow (max " + std::to_string(st.max_ulp_dut_shadow) + " ULP)");
        // x1==x2==(cx+0.5)*stride = 10.5*16 = 168
        uint16_t exp = double_to_fp16(10.5 * 16.0);
        sim.check(fp16_ulp_diff(r.dut.x1, exp) <= TOLERANCE_ULP,
                  "onehot0: x1==168 got=" + hex16(r.dut.x1));
        sim.check(fp16_ulp_diff(r.dut.x2, exp) <= TOLERANCE_ULP,
                  "onehot0: x2==168 got=" + hex16(r.dut.x2));
    }

    // ─── Test 4: one-hot at i=15 → d=15 ─────────────────
    printf("test 4: one-hot at i=15 → d=15\n");
    {
        std::array<uint16_t, 16> pl, pt, pr, pb;
        onehot_vec(pl, 15); onehot_vec(pt, 15); onehot_vec(pr, 15); onehot_vec(pb, 15);
        int16_t cx = 4, cy = 6, stride = 32;
        DriveResult r = drive_one(sim, pl, pt, pr, pb, cx, cy, stride);
        Stats st; int budget = 10;
        compare_box(r, st, "onehot15", budget);
        sim.check(st.fails_dut_ref == 0,
                  "onehot15: 0 fails dut-vs-ref (max " + std::to_string(st.max_ulp_dut_ref) + " ULP)");
        sim.check(st.fails_dut_shadow == 0,
                  "onehot15: 0 fails dut-vs-shadow (max " + std::to_string(st.max_ulp_dut_shadow) + " ULP)");
        // x1 = (4 + 0.5 - 15) * 32 = -336, x2 = (4 + 0.5 + 15) * 32 = 624
        uint16_t x1_e = double_to_fp16(-336.0);
        uint16_t x2_e = double_to_fp16( 624.0);
        sim.check(fp16_ulp_diff(r.dut.x1, x1_e) <= TOLERANCE_ULP,
                  "onehot15: x1==-336 got=" + hex16(r.dut.x1));
        sim.check(fp16_ulp_diff(r.dut.x2, x2_e) <= TOLERANCE_ULP,
                  "onehot15: x2==624 got=" + hex16(r.dut.x2));
    }

    // ─── Test 5: directed strides {8, 16, 32} ───────────
    printf("test 5: directed strides {8, 16, 32}\n");
    {
        std::mt19937 rng(0xDECAFu);
        int16_t strides[3] = {8, 16, 32};
        Stats st; int budget = 10;
        for (int sIdx = 0; sIdx < 3; sIdx++) {
            int16_t stride = strides[sIdx];
            for (int t = 0; t < 50; t++) {
                std::array<uint16_t, 16> pl, pt, pr, pb;
                rand_prob_vec(rng, pl); rand_prob_vec(rng, pt);
                rand_prob_vec(rng, pr); rand_prob_vec(rng, pb);
                int16_t cx = (int16_t)(rng() % 80);
                int16_t cy = (int16_t)(rng() % 80);
                DriveResult r = drive_one(sim, pl, pt, pr, pb, cx, cy, stride);
                compare_box(r, st, "strides", budget);
            }
        }
        printf("  strides: max ULP dut-vs-ref=%d  dut-vs-shadow=%d  ref-vs-shadow=%d\n",
               st.max_ulp_dut_ref, st.max_ulp_dut_shadow, st.max_ulp_ref_shadow);
        sim.check(st.fails_dut_ref == 0,
                  "strides: 0 fails dut-vs-ref (was " + std::to_string(st.fails_dut_ref) + ")");
        sim.check(st.fails_dut_shadow == 0,
                  "strides: 0 fails dut-vs-shadow (was " + std::to_string(st.fails_dut_shadow) + ")");
    }

    // ─── Test 6: 2000 random vectors ────────────────────
    printf("test 6: 2000 random vectors\n");
    {
        std::mt19937 rng(0xFA1A4F00u);
        Stats st; int budget = 20;
        const int N = 2000;
        int16_t strides[3] = {8, 16, 32};
        for (int t = 0; t < N; t++) {
            std::array<uint16_t, 16> pl, pt, pr, pb;
            rand_prob_vec(rng, pl); rand_prob_vec(rng, pt);
            rand_prob_vec(rng, pr); rand_prob_vec(rng, pb);
            int16_t cx = (int16_t)(rng() % 160);
            int16_t cy = (int16_t)(rng() % 160);
            int16_t stride = strides[rng() % 3];
            DriveResult r = drive_one(sim, pl, pt, pr, pb, cx, cy, stride);
            compare_box(r, st, "rand", budget);
        }
        printf("  random: max ULP dut-vs-ref=%d  dut-vs-shadow=%d  ref-vs-shadow=%d\n",
               st.max_ulp_dut_ref, st.max_ulp_dut_shadow, st.max_ulp_ref_shadow);
        sim.check(st.fails_dut_ref == 0,
                  "random: 0 fails dut-vs-ref (was " + std::to_string(st.fails_dut_ref) + ")");
        sim.check(st.fails_dut_shadow == 0,
                  "random: 0 fails dut-vs-shadow (was " + std::to_string(st.fails_dut_shadow) + ")");
    }

    // ─── Test 7: peaked distributions (narrow softmax) ──
    printf("test 7: 500 peaked-distribution vectors (narrow softmax)\n");
    {
        std::mt19937 rng(0xC0FFEEu);
        Stats st; int budget = 10;
        const int N = 500;
        int16_t strides[3] = {8, 16, 32};
        for (int t = 0; t < N; t++) {
            std::array<uint16_t, 16> pl, pt, pr, pb;
            // scale=12 → softer/sharper depending on rng; bigger scale
            // produces more peaked distributions (one bin dominates).
            rand_prob_vec(rng, pl, 12.0);
            rand_prob_vec(rng, pt, 12.0);
            rand_prob_vec(rng, pr, 12.0);
            rand_prob_vec(rng, pb, 12.0);
            int16_t cx = (int16_t)(rng() % 100);
            int16_t cy = (int16_t)(rng() % 100);
            int16_t stride = strides[rng() % 3];
            DriveResult r = drive_one(sim, pl, pt, pr, pb, cx, cy, stride);
            compare_box(r, st, "peaked", budget);
        }
        printf("  peaked: max ULP dut-vs-ref=%d  dut-vs-shadow=%d  ref-vs-shadow=%d\n",
               st.max_ulp_dut_ref, st.max_ulp_dut_shadow, st.max_ulp_ref_shadow);
        sim.check(st.fails_dut_ref == 0,
                  "peaked: 0 fails dut-vs-ref");
        sim.check(st.fails_dut_shadow == 0,
                  "peaked: 0 fails dut-vs-shadow");
    }

    return sim.finish();
}
