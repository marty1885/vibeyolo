// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// act_silu — Verilator test.
//
// Drives DUT (act_silu) and REF (act_silu_ref) at two parameterisations
// (variant A = balanced 1/16,1/16; variant B = saturating 1/4,1/64) with
// identical stimulus and compares (a) DUT vs REF in SV and (b) DUT vs an
// independent C++ shadow that recomputes SiLU(x*InScale)/OutScale in
// double precision with round-half-to-even and int8 saturation.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

#include "Vact_silu_tb.h"
#include "sim_ctrl.h"

using DUT = Vact_silu_tb;

static int8_t sat_i8(long long q) {
    if (q >  127) return  127;
    if (q < -128) return -128;
    return static_cast<int8_t>(q);
}

// Round-half-to-even on a magnitude, then re-apply sign. Mirrors the
// rounding done in act_silu / act_silu_ref.
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

static int8_t shadow_silu(int codepoint, double in_scale, double out_scale) {
    double f = static_cast<double>(codepoint) * in_scale;
    double s;
    if (f >= 0.0) {
        s = f / (1.0 + std::exp(-f));
    } else {
        double ef = std::exp(f);
        s = (f * ef) / (1.0 + ef);
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

    // Scales must match act_silu_tb.sv.
    const double A_IN  = 1.0/16.0,  A_OUT = 1.0/16.0;
    const double B_IN  = 1.0/4.0,   B_OUT = 1.0/512.0;

    sim.dut->x_i = 0;
    sim.reset();

    // ── Test 1: y == 0 after reset ──────────────────
    printf("test 1: y zero after reset\n");
    sim.check(static_cast<int8_t>(sim.dut->yA_dut_o) == 0,
              "A: dut y == 0 after reset");
    sim.check(static_cast<int8_t>(sim.dut->yA_ref_o) == 0,
              "A: ref y == 0 after reset");
    sim.check(static_cast<int8_t>(sim.dut->yB_dut_o) == 0,
              "B: dut y == 0 after reset");
    sim.check(static_cast<int8_t>(sim.dut->yB_ref_o) == 0,
              "B: ref y == 0 after reset");
    sim.check(sim.dut->mismatchA_o == 0, "A: no mismatch after reset");
    sim.check(sim.dut->mismatchB_o == 0, "B: no mismatch after reset");

    // ── Test 2: 1-cycle latency ─────────────────────
    printf("test 2: registered output (1-cycle latency)\n");
    // Drive a known non-zero value; the *previous* y is still 0 because
    // the prior cycle had x=0 → silu(0)=0. After one tick the new value
    // appears.
    sim.dut->x_i = static_cast<uint8_t>(int8_t(64));
    sim.tick();
    int8_t expA_64 = shadow_silu(64, A_IN, A_OUT);
    sim.check(static_cast<int8_t>(sim.dut->yA_dut_o) == expA_64,
              "A: y(x=64) == shadow after 1 cycle");

    // ── Test 3: exhaustive sweep, both variants ─────
    printf("test 3: exhaustive 256-codepoint sweep\n");
    int mmA = 0, mmB = 0, shA = 0, shB = 0;
    int satB_pos = 0, satB_neg = 0;
    // Track pre-clamp magnitude to verify variant B genuinely saturates
    // (rather than just landing at ±127 by chance).
    int clampB_pos = 0, clampB_neg = 0;
    for (int i = -128; i <= 127; i++) {
        apply(sim, static_cast<int8_t>(i));
        int8_t a_dut = static_cast<int8_t>(sim.dut->yA_dut_o);
        int8_t a_ref = static_cast<int8_t>(sim.dut->yA_ref_o);
        int8_t b_dut = static_cast<int8_t>(sim.dut->yB_dut_o);
        int8_t b_ref = static_cast<int8_t>(sim.dut->yB_ref_o);
        // After apply(i): the posedge sampled x=i and the registered
        // y_o now equals lut[i] — compare against shadow(i).
        int8_t a_sh = shadow_silu(i, A_IN, A_OUT);
        int8_t b_sh = shadow_silu(i, B_IN, B_OUT);

        if (sim.dut->mismatchA_o) mmA++;
        if (sim.dut->mismatchB_o) mmB++;
        if (a_dut != a_sh) shA++;
        if (b_dut != b_sh) shB++;
        (void)a_ref; (void)b_ref;

        if (b_sh ==  127) satB_pos++;
        if (b_sh == -128) satB_neg++;
        // Pre-clamp scaled SiLU for variant B.
        double f = static_cast<double>(i) * B_IN;
        double sB;
        if (f >= 0.0) sB = f / (1.0 + std::exp(-f));
        else { double ef = std::exp(f); sB = (f * ef) / (1.0 + ef); }
        double scaledB = sB / B_OUT;
        if (scaledB >  127.5) clampB_pos++;
        if (scaledB < -128.5) clampB_neg++;
    }
    sim.check(mmA == 0, "A: 0 SV mismatches across sweep (got " +
                            std::to_string(mmA) + ")");
    sim.check(mmB == 0, "B: 0 SV mismatches across sweep (got " +
                            std::to_string(mmB) + ")");
    sim.check(shA == 0, "A: 0 C++ shadow mismatches (got " +
                            std::to_string(shA) + ")");
    sim.check(shB == 0, "B: 0 C++ shadow mismatches (got " +
                            std::to_string(shB) + ")");
    // Variant B is engineered to saturate at both rails.
    sim.check(clampB_pos > 0 && satB_pos > 0,
              "B: positive saturation actually clamps");
    sim.check(clampB_neg > 0 && satB_neg > 0,
              "B: negative saturation actually clamps");

    // ── Test 4: spot-check specific SiLU values ─────
    printf("test 4: spot-check anchor codepoints\n");
    // SiLU(0)=0 → y=0 at both variants.
    apply(sim, 0);
    apply(sim, 0);  // wait for the previous result to flush past test 3
    sim.check(static_cast<int8_t>(sim.dut->yA_dut_o) == 0, "A: silu(0)=0");
    sim.check(static_cast<int8_t>(sim.dut->yB_dut_o) == 0, "B: silu(0)=0");
    // SiLU is monotone non-decreasing in its *float* input. The int8
    // codepoint ordering interleaves positives [0..127] and negatives
    // [-128..-1], so traverse the codepoints in two ranges that ARE
    // monotone: 0..127 (rising) and -128..-1 (the float input is rising
    // f=-8.0..-0.0625, but SiLU has a minimum near f=-1.27 so monotone
    // only holds for f > -1.27, i.e. codepoints -20..-1 with A_IN=1/16).
    apply(sim, 0);  // flush
    int8_t y_prev = 0;
    bool monotone_pos = true;
    for (int i = 0; i <= 127; i++) {
        apply(sim, static_cast<int8_t>(i));
        int8_t y = static_cast<int8_t>(sim.dut->yA_dut_o);
        if (i > 0 && y < y_prev) monotone_pos = false;
        y_prev = y;
    }
    sim.check(monotone_pos,
              "A: SiLU output monotone non-decreasing over [0,127]");

    // ── Test 5: randomized stream (10k samples) ─────
    printf("test 5: random stream (10000 samples)\n");
    std::mt19937 rng(0xBADCAFEu);
    std::uniform_int_distribution<int> d8(-128, 127);
    int rmm_a = 0, rmm_b = 0, rsh_a = 0, rsh_b = 0;
    const int N = 10000;
    for (int i = 0; i < N; i++) {
        int cp = d8(rng);
        apply(sim, static_cast<int8_t>(cp));
        int8_t a_dut = static_cast<int8_t>(sim.dut->yA_dut_o);
        int8_t b_dut = static_cast<int8_t>(sim.dut->yB_dut_o);
        int8_t a_sh  = shadow_silu(cp, A_IN, A_OUT);
        int8_t b_sh  = shadow_silu(cp, B_IN, B_OUT);
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

    // ── Test 6: mid-stream reset clears y ───────────
    printf("test 6: mid-stream reset clears y\n");
    apply(sim, 100);  // produce non-zero state
    apply(sim, 100);
    sim.check(static_cast<int8_t>(sim.dut->yB_dut_o) != 0,
              "B: y non-zero before reset");
    sim.dut->x_i = 0;
    sim.reset();
    sim.check(static_cast<int8_t>(sim.dut->yA_dut_o) == 0,
              "A: y==0 after re-reset");
    sim.check(static_cast<int8_t>(sim.dut->yB_dut_o) == 0,
              "B: y==0 after re-reset");
    sim.check(sim.dut->mismatchA_o == 0, "A: no mismatch after re-reset");
    sim.check(sim.dut->mismatchB_o == 0, "B: no mismatch after re-reset");

    return sim.finish();
}
