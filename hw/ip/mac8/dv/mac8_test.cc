// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// mac8 — Verilator test.
//
// Drives DUT (mac8) and behavioral REF (mac8_ref) with identical stimulus
// via the mac8_tb wrapper. Every cycle we (a) assert the wrapper's
// mismatch_o is zero and (b) cross-check the DUT against an independent
// C++ shadow accumulator. This double-check guards against the case where
// DUT and REF share the same bug.

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

#include "Vmac8_tb.h"
#include "sim_ctrl.h"

using DUT = Vmac8_tb;

// Apply (a, b, en, clr) for one cycle and step the clock.
static void step(SimCtrl<DUT>& s, int8_t a, int8_t b, bool en, bool clr) {
    s.dut->a_i   = static_cast<uint8_t>(a);
    s.dut->b_i   = static_cast<uint8_t>(b);
    s.dut->en_i  = en ? 1 : 0;
    s.dut->clr_i = clr ? 1 : 0;
    s.tick();
}

static int32_t acc_dut(SimCtrl<DUT>& s) {
    return static_cast<int32_t>(s.dut->acc_dut_o);
}
static int32_t acc_ref(SimCtrl<DUT>& s) {
    return static_cast<int32_t>(s.dut->acc_ref_o);
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 200000000ull;  // 100M cycles cap; we use ~10k

    // ── Init ────────────────────────────────────────────
    sim.dut->a_i   = 0;
    sim.dut->b_i   = 0;
    sim.dut->en_i  = 0;
    sim.dut->clr_i = 0;
    sim.reset();

    // ── Test 1: acc==0 after reset ──────────────────────
    printf("test 1: acc zero after reset\n");
    sim.check(acc_dut(sim) == 0, "dut acc == 0 after reset");
    sim.check(acc_ref(sim) == 0, "ref acc == 0 after reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after reset");

    // Idle a few cycles with en=clr=0 — acc must hold at 0.
    for (int i = 0; i < 4; i++) {
        step(sim, 7, -3, false, false);
        sim.check(acc_dut(sim) == 0, "idle holds 0");
        sim.check(sim.dut->mismatch_o == 0, "idle: no mismatch");
    }

    // ── Test 2: clr_i loads product ─────────────────────
    printf("test 2: clr loads product (directed corners)\n");
    struct Pair { int8_t a, b; int32_t exp; };
    const Pair corners[] = {
        { 0,    0,        0},
        { 1,    1,        1},
        {-1,    1,       -1},
        { 127,  127,  16129},
        {-128, -128,  16384},
        {-128,  127, -16256},
        { 127, -128, -16256},
        {-128,  1,     -128},
    };
    for (auto& p : corners) {
        step(sim, p.a, p.b, false, true);  // clr loads a*b
        sim.check(acc_dut(sim) == p.exp,
                  "dut clr loads " + std::to_string(p.a) + "*" +
                  std::to_string(p.b) + " = " + std::to_string(p.exp));
        sim.check(acc_ref(sim) == p.exp, "ref matches expected");
        sim.check(sim.dut->mismatch_o == 0, "clr: no mismatch");
    }

    // ── Test 3: en_i accumulates ────────────────────────
    printf("test 3: en accumulates over a sequence\n");
    // Reset accumulator via clr with zero product.
    step(sim, 0, 0, false, true);
    sim.check(acc_dut(sim) == 0, "acc cleared to 0");
    int32_t shadow = 0;
    const int8_t seqA[] = { 3,  -5,  7,  -2, 11, -4,  6,  9};
    const int8_t seqB[] = { 4,   6, -8,  10,  2,  3, -7,  1};
    for (size_t i = 0; i < sizeof(seqA); i++) {
        step(sim, seqA[i], seqB[i], true, false);
        shadow += int32_t(seqA[i]) * int32_t(seqB[i]);
        sim.check(acc_dut(sim) == shadow,
                  "dut acc matches shadow @" + std::to_string(i));
        sim.check(sim.dut->mismatch_o == 0, "accum: no mismatch");
    }

    // ── Test 4: en=0,clr=0 holds ────────────────────────
    printf("test 4: hold when en=clr=0\n");
    int32_t held = acc_dut(sim);
    for (int i = 0; i < 5; i++) {
        // Apply non-zero inputs but neither en nor clr — must hold.
        step(sim, 42, -13, false, false);
        sim.check(acc_dut(sim) == held, "acc held");
        sim.check(sim.dut->mismatch_o == 0, "hold: no mismatch");
    }

    // ── Test 5: int32 wrap (not saturate) ───────────────
    printf("test 5: int32 wrap matches ref (no saturation)\n");
    // Force acc near INT32_MAX, then push it over via accumulation.
    // Strategy: clr with 127*127=16129; accumulate 127*127 many times so it
    // wraps. Track shadow in C++ with wrapping int32 semantics.
    step(sim, 127, 127, false, true);
    int32_t s5 = 16129;
    sim.check(acc_dut(sim) == s5, "wrap init");
    // 2^31 / 16129 ≈ 133143 — accumulate that many to definitely wrap.
    const int wrap_iters = 200000;
    for (int i = 0; i < wrap_iters; i++) {
        step(sim, 127, 127, true, false);
        // Use unsigned arithmetic to get well-defined wrap.
        s5 = int32_t(uint32_t(s5) + uint32_t(16129));
        if (sim.dut->mismatch_o) {
            sim.check(false, "wrap: mismatch at iter " + std::to_string(i));
            break;
        }
        if (acc_dut(sim) != s5) {
            sim.check(false, "wrap: dut!=shadow at iter " + std::to_string(i));
            break;
        }
    }
    sim.check(acc_dut(sim) == acc_ref(sim), "wrap end: dut == ref");
    sim.check(acc_dut(sim) == s5, "wrap end: dut == shadow");

    // ── Test 6: randomized stress ───────────────────────
    printf("test 6: randomized stress (10000 cycles)\n");
    step(sim, 0, 0, false, true);  // start from known state
    int32_t rshadow = 0;
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<int> d8(-128, 127);
    std::uniform_int_distribution<int> d1(0, 1);
    // Bias en/clr to keep them mostly active so we exercise both paths.
    std::uniform_int_distribution<int> dprob(0, 99);

    int n_mismatch = 0;
    int n_shadow_fail = 0;
    const int N = 10000;
    for (int i = 0; i < N; i++) {
        int8_t a = int8_t(d8(rng));
        int8_t b = int8_t(d8(rng));
        // 10% clr, otherwise 70% en, else hold.
        int p = dprob(rng);
        bool clr = (p < 10);
        bool en  = (!clr) && (p < 80);

        step(sim, a, b, en, clr);

        int32_t prod = int32_t(a) * int32_t(b);
        if (clr) {
            rshadow = prod;
        } else if (en) {
            rshadow = int32_t(uint32_t(rshadow) + uint32_t(prod));
        }

        if (sim.dut->mismatch_o) n_mismatch++;
        if (acc_dut(sim) != rshadow) n_shadow_fail++;
    }
    sim.check(n_mismatch == 0,
              "rand: 0 cycle-by-cycle mismatches (was " +
              std::to_string(n_mismatch) + ")");
    sim.check(n_shadow_fail == 0,
              "rand: 0 dut-vs-C++-shadow mismatches (was " +
              std::to_string(n_shadow_fail) + ")");

    // ── Test 7: reset re-asserts to 0 mid-stream ────────
    printf("test 7: mid-stream reset clears acc\n");
    step(sim, 50, 50, true, false);  // produce non-zero state
    sim.check(acc_dut(sim) != 0, "acc non-zero before reset");
    // Quiesce inputs so the post-reset settle cycle doesn't re-accumulate.
    sim.dut->en_i  = 0;
    sim.dut->clr_i = 0;
    sim.reset();
    sim.check(acc_dut(sim) == 0, "dut acc==0 after re-reset");
    sim.check(acc_ref(sim) == 0, "ref acc==0 after re-reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after re-reset");

    return sim.finish();
}
