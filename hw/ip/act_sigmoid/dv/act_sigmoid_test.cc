// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// act_sigmoid — Verilator test.
//
// Drives DUT (act_sigmoid) and REF (act_sigmoid_ref) at two
// parameterisations (variant A = non-saturating 1/16,1/128; variant B =
// upper-rail saturating 1/16,1/256) with identical stimulus and
// compares (a) DUT vs REF in SV and (b) DUT vs an independent C++
// shadow that recomputes sigmoid(x*InScale)/OutScale in double precision
// with round-half-to-even and int8 saturation.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

#include "Vact_sigmoid_tb.h"
#include "sim_ctrl.h"

using DUT = Vact_sigmoid_tb;

static int8_t sat_i8(long long q) {
    if (q >  127) return  127;
    if (q < -128) return -128;
    return static_cast<int8_t>(q);
}

// Round-half-to-even on a magnitude, then re-apply sign. Mirrors the
// rounding done in act_sigmoid / act_sigmoid_ref.
static long long rne_ll(double v) {
    bool neg = v < 0.0;
    double mag = neg ? -v : v;
    double fl = std::floor(mag);
    double frac = mag - fl;
    long long ip = static_cast<long long>(fl);
    long long q;
    if (frac > 0.5) {
        q = ip + 1;
    } else if (frac < 0.5) {
        q = ip;
    } else {
        // tie → magnitude rounded to even
        q = (ip & 1) ? (ip + 1) : ip;
    }
    return neg ? -q : q;
}

static int8_t shadow_sigmoid(int codepoint, double in_scale,
                             double out_scale) {
    double f = static_cast<double>(codepoint) * in_scale;
    double s;
    if (f >= 0.0) {
        s = 1.0 / (1.0 + std::exp(-f));
    } else {
        double ef = std::exp(f);
        s = ef / (1.0 + ef);
    }
    return sat_i8(rne_ll(s / out_scale));
}

static void apply(SimCtrl<DUT>& s, int8_t x) {
    s.dut->x_i = static_cast<uint8_t>(x);
    s.tick();
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 200000000ull;

    // Scales must match act_sigmoid_tb.sv.
    const double A_IN  = 1.0/16.0,  A_OUT = 1.0/128.0;
    const double B_IN  = 1.0/16.0,  B_OUT = 1.0/256.0;

    sim.dut->x_i = 0;
    // Drive rst_ni low manually and observe y_o == 0 BEFORE the post-
    // release tick fires. Doing a sim.reset() would clock one cycle
    // after release and immediately latch sigmoid(0)≠0, so to verify
    // the reset-clears-y behaviour we sample mid-reset.
    sim.dut->rst_ni = 0;
    sim.dut->eval();

    // ── Test 1: y == 0 while reset is asserted ──────
    printf("test 1: y zero under reset\n");
    sim.check(static_cast<int8_t>(sim.dut->yA_dut_o) == 0,
              "A: dut y == 0 under reset");
    sim.check(static_cast<int8_t>(sim.dut->yA_ref_o) == 0,
              "A: ref y == 0 under reset");
    sim.check(static_cast<int8_t>(sim.dut->yB_dut_o) == 0,
              "B: dut y == 0 under reset");
    sim.check(static_cast<int8_t>(sim.dut->yB_ref_o) == 0,
              "B: ref y == 0 under reset");
    sim.check(sim.dut->mismatchA_o == 0, "A: no mismatch under reset");
    sim.check(sim.dut->mismatchB_o == 0, "B: no mismatch under reset");

    // Now release reset properly.
    sim.reset();

    // ── Test 2: 1-cycle latency ─────────────────────
    printf("test 2: registered output (1-cycle latency)\n");
    // Drive a known non-zero value. After one tick the new value appears.
    sim.dut->x_i = static_cast<uint8_t>(int8_t(64));
    sim.tick();
    int8_t expA_64 = shadow_sigmoid(64, A_IN, A_OUT);
    sim.check(static_cast<int8_t>(sim.dut->yA_dut_o) == expA_64,
              "A: y(x=64) == shadow after 1 cycle");

    // ── Test 3: exhaustive sweep, both variants ─────
    printf("test 3: exhaustive 256-codepoint sweep\n");
    int mmA = 0, mmB = 0, shA = 0, shB = 0;
    int satB_pos = 0;
    int clampB_pos = 0;
    for (int i = -128; i <= 127; i++) {
        apply(sim, static_cast<int8_t>(i));
        int8_t a_dut = static_cast<int8_t>(sim.dut->yA_dut_o);
        int8_t a_ref = static_cast<int8_t>(sim.dut->yA_ref_o);
        int8_t b_dut = static_cast<int8_t>(sim.dut->yB_dut_o);
        int8_t b_ref = static_cast<int8_t>(sim.dut->yB_ref_o);
        // After apply(i): the posedge sampled x=i and the registered
        // y_o now equals lut[i] — compare against shadow(i).
        int8_t a_sh = shadow_sigmoid(i, A_IN, A_OUT);
        int8_t b_sh = shadow_sigmoid(i, B_IN, B_OUT);

        if (sim.dut->mismatchA_o) mmA++;
        if (sim.dut->mismatchB_o) mmB++;
        if (a_dut != a_sh) shA++;
        if (b_dut != b_sh) shB++;
        (void)a_ref; (void)b_ref;

        if (b_sh == 127) satB_pos++;
        // Pre-clamp scaled sigmoid for variant B.
        double f = static_cast<double>(i) * B_IN;
        double sB;
        if (f >= 0.0) sB = 1.0 / (1.0 + std::exp(-f));
        else { double ef = std::exp(f); sB = ef / (1.0 + ef); }
        double scaledB = sB / B_OUT;
        if (scaledB > 127.5) clampB_pos++;

        // Variant A is non-saturating — sigmoid(x*1/16)/1/128 ∈ [0, 128),
        // and the only value that would round to 128 is sigmoid → 1
        // which is never reached for finite x, so y is in [0, 127] and
        // never negative. Assert that here.
        sim.check(a_dut >= 0,
                  "A: sigmoid output non-negative (got negative at i=" +
                      std::to_string(i) + ")");
        sim.check(b_dut >= 0,
                  "B: sigmoid output non-negative (got negative at i=" +
                      std::to_string(i) + ")");
    }
    sim.check(mmA == 0, "A: 0 SV mismatches across sweep (got " +
                            std::to_string(mmA) + ")");
    sim.check(mmB == 0, "B: 0 SV mismatches across sweep (got " +
                            std::to_string(mmB) + ")");
    sim.check(shA == 0, "A: 0 C++ shadow mismatches (got " +
                            std::to_string(shA) + ")");
    sim.check(shB == 0, "B: 0 C++ shadow mismatches (got " +
                            std::to_string(shB) + ")");
    // Variant B is engineered to saturate at the upper rail.
    sim.check(clampB_pos > 0 && satB_pos > 0,
              "B: positive saturation actually clamps");

    // ── Test 4: monotonicity across all 256 codepoints ──
    printf("test 4: monotone non-decreasing across full input range\n");
    // sigmoid is monotone increasing in its float input, and the float
    // input f = i * InScale is monotone in i across i ∈ [-128, 127].
    // Hence the quantised output must be monotone non-decreasing for
    // both variants over the full codepoint range.
    apply(sim, 0);  // flush
    {
        int8_t prevA = -128, prevB = -128;
        bool first = true;
        bool monoA = true, monoB = true;
        for (int i = -128; i <= 127; i++) {
            apply(sim, static_cast<int8_t>(i));
            int8_t a = static_cast<int8_t>(sim.dut->yA_dut_o);
            int8_t b = static_cast<int8_t>(sim.dut->yB_dut_o);
            if (!first) {
                if (a < prevA) monoA = false;
                if (b < prevB) monoB = false;
            }
            prevA = a; prevB = b; first = false;
        }
        sim.check(monoA,
                  "A: sigmoid output monotone non-decreasing on [-128,127]");
        sim.check(monoB,
                  "B: sigmoid output monotone non-decreasing on [-128,127]");
    }

    // ── Test 5: spot-check anchor codepoints ────────
    printf("test 5: spot-check anchor codepoints\n");
    apply(sim, 0);
    apply(sim, 0);
    // sigmoid(0) = 0.5 → A: round(0.5/(1/128)) = round(64) = 64
    //                  → B: round(0.5/(1/256)) = round(128) → sat 127
    sim.check(static_cast<int8_t>(sim.dut->yA_dut_o) == 64,
              "A: sigmoid(0)=0.5 → 64");
    sim.check(static_cast<int8_t>(sim.dut->yB_dut_o) == 127,
              "B: sigmoid(0)=0.5 → 128 → sat 127");

    // ── Test 6: randomized stream (10k samples) ─────
    printf("test 6: random stream (10000 samples)\n");
    std::mt19937 rng(0xBADCAFEu);
    std::uniform_int_distribution<int> d8(-128, 127);
    int rmm_a = 0, rmm_b = 0, rsh_a = 0, rsh_b = 0;
    const int N = 10000;
    for (int i = 0; i < N; i++) {
        int cp = d8(rng);
        apply(sim, static_cast<int8_t>(cp));
        int8_t a_dut = static_cast<int8_t>(sim.dut->yA_dut_o);
        int8_t b_dut = static_cast<int8_t>(sim.dut->yB_dut_o);
        int8_t a_sh  = shadow_sigmoid(cp, A_IN, A_OUT);
        int8_t b_sh  = shadow_sigmoid(cp, B_IN, B_OUT);
        if (sim.dut->mismatchA_o) rmm_a++;
        if (sim.dut->mismatchB_o) rmm_b++;
        if (a_dut != a_sh) rsh_a++;
        if (b_dut != b_sh) rsh_b++;
    }
    sim.check(rmm_a == 0, "A rand: 0 SV mismatches (got " +
                              std::to_string(rmm_a) + ")");
    sim.check(rmm_b == 0, "B rand: 0 SV mismatches (got " +
                              std::to_string(rmm_b) + ")");
    sim.check(rsh_a == 0, "A rand: 0 shadow mismatches (got " +
                              std::to_string(rsh_a) + ")");
    sim.check(rsh_b == 0, "B rand: 0 shadow mismatches (got " +
                              std::to_string(rsh_b) + ")");

    // ── Test 7: mid-stream reset clears y ───────────
    printf("test 7: mid-stream reset clears y\n");
    apply(sim, 100);  // produce non-zero state
    apply(sim, 100);
    sim.check(static_cast<int8_t>(sim.dut->yA_dut_o) != 0,
              "A: y non-zero before reset");
    sim.check(static_cast<int8_t>(sim.dut->yB_dut_o) != 0,
              "B: y non-zero before reset");
    sim.dut->x_i = 0;
    // Assert reset and sample y_o while held — it must drop to 0
    // immediately (async-assert), even though x_i drives a non-zero
    // sigmoid value combinationally in the REF.
    sim.dut->rst_ni = 0;
    sim.dut->eval();
    sim.check(static_cast<int8_t>(sim.dut->yA_dut_o) == 0,
              "A: y==0 while re-reset asserted");
    sim.check(static_cast<int8_t>(sim.dut->yB_dut_o) == 0,
              "B: y==0 while re-reset asserted");
    sim.check(sim.dut->mismatchA_o == 0, "A: no mismatch under re-reset");
    sim.check(sim.dut->mismatchB_o == 0, "B: no mismatch under re-reset");
    // Complete the reset sequence.
    sim.reset();

    return sim.finish();
}
