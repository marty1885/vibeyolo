// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// maxpool_kxk — Verilator test.
//
// Drives DUT (maxpool_kxk) and behavioral REF (maxpool_kxk_ref) with
// identical stimulus via the maxpool_kxk_tb wrapper. Every cycle we
// (a) assert the wrapper's mismatch_o is zero and (b) cross-check the
// DUT against an independent C++ shadow max (std::max_element). This
// double-check guards against the case where DUT and REF share the
// same bug.
//
// K is selected at compile time via -DK=<n> (the Makefile passes the
// matching value to both Verilator and g++).

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "Vmaxpool_kxk_tb.h"
#include "sim_ctrl.h"

#ifndef MAXPOOL_K
#error "MAXPOOL_K must be defined on the command line (e.g. -DMAXPOOL_K=5)"
#endif

static constexpr int kK = MAXPOOL_K;
static constexpr int kN = MAXPOOL_K * MAXPOOL_K;
static constexpr int kBits = kN * 8;

using DUT = Vmaxpool_kxk_tb;

// Pack kN int8 lanes into the DUT's x_i port. Verilator represents the
// packed-array port as a single wide signal — for kBits <= 32 it's a
// uint32_t scalar, for 33..64 a uint64_t scalar, and for > 64 a CData
// array of 32-bit words. We discover the type via sizeof at compile
// time and write each lane to the corresponding byte position.
static void set_x(DUT* dut, const std::array<int8_t, kN>& v) {
    using PortT = decltype(dut->x_i);
    constexpr size_t port_bytes = sizeof(PortT);
    static_assert(port_bytes * 8 >= static_cast<size_t>(kBits),
                  "x_i port narrower than expected");

    // Zero the entire backing storage so any padding bits above the
    // top byte stay clean cycle-to-cycle.
    std::memset(&dut->x_i, 0, port_bytes);

    // Write each lane as an unsigned byte at byte offset i in the
    // little-endian layout Verilator uses for packed signals.
    uint8_t* raw = reinterpret_cast<uint8_t*>(&dut->x_i);
    for (int i = 0; i < kN; i++) {
        raw[i] = static_cast<uint8_t>(v[i]);
    }
}

// Apply (vector, en) for one cycle and step the clock. Returns the
// expected post-tick max if en is true, else the prior expected.
static void step(SimCtrl<DUT>& s, const std::array<int8_t, kN>& v, bool en) {
    set_x(s.dut.get(), v);
    s.dut->en_i = en ? 1 : 0;
    s.tick();
}

static int8_t y_dut(SimCtrl<DUT>& s) {
    return static_cast<int8_t>(s.dut->y_dut_o);
}
static int8_t y_ref(SimCtrl<DUT>& s) {
    return static_cast<int8_t>(s.dut->y_ref_o);
}

static int8_t shadow_max(const std::array<int8_t, kN>& v) {
    return *std::max_element(v.begin(), v.end());
}

static std::array<int8_t, kN> filled(int8_t val) {
    std::array<int8_t, kN> v;
    v.fill(val);
    return v;
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 200000000ull;

    printf("==== maxpool_kxk K=%d (N=%d) ====\n", kK, kN);

    // ── Init ────────────────────────────────────────────
    sim.dut->en_i = 0;
    set_x(sim.dut.get(), filled(0));
    sim.reset();

    // ── Test 1: y_o == 0 after reset ────────────────────
    printf("test 1: y zero after reset\n");
    sim.check(y_dut(sim) == 0, "dut y == 0 after reset");
    sim.check(y_ref(sim) == 0, "ref y == 0 after reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after reset");

    // Idle a few cycles with en=0 — y must hold at 0 even with stimulus
    // changing on x_i.
    for (int i = 0; i < 4; i++) {
        step(sim, filled(static_cast<int8_t>(50 + i)), false);
        sim.check(y_dut(sim) == 0, "idle: y holds 0");
        sim.check(sim.dut->mismatch_o == 0, "idle: no mismatch");
    }

    // ── Test 2: directed corners ────────────────────────
    printf("test 2: directed corner patches\n");

    struct Case {
        std::array<int8_t, kN> v;
        int8_t exp;
        const char* name;
    };
    std::vector<Case> cases;

    cases.push_back({filled(0), 0, "all zero"});
    cases.push_back({filled(-128), -128, "all -128"});
    cases.push_back({filled(127), 127, "all +127"});

    // One +127 among -128s.
    {
        auto v = filled(static_cast<int8_t>(-128));
        v[kN / 2] = 127;
        cases.push_back({v, 127, "one +127 among -128s"});
    }
    // One -128 among +127s -> still +127.
    {
        auto v = filled(static_cast<int8_t>(127));
        v[0] = -128;
        cases.push_back({v, 127, "one -128 among +127s"});
    }
    // One +50 in an otherwise -100 patch.
    {
        auto v = filled(static_cast<int8_t>(-100));
        v[kN - 1] = 50;
        cases.push_back({v, 50, "one +50 in -100s"});
    }
    // Descending sequence starting from a positive seed.
    {
        std::array<int8_t, kN> v;
        for (int i = 0; i < kN; i++) {
            v[i] = static_cast<int8_t>(120 - i);
        }
        cases.push_back({v, 120, "descending sequence"});
    }
    // Ascending sequence ending at a positive cap.
    {
        std::array<int8_t, kN> v;
        for (int i = 0; i < kN; i++) {
            v[i] = static_cast<int8_t>(-50 + i);
        }
        int8_t exp = static_cast<int8_t>(-50 + kN - 1);
        cases.push_back({v, exp, "ascending sequence"});
    }
    // Mixed signs, max at the last position.
    {
        std::array<int8_t, kN> v;
        for (int i = 0; i < kN; i++) {
            v[i] = static_cast<int8_t>(((i & 1) ? -1 : 1) * (10 + i));
        }
        v[kN - 1] = 99;
        cases.push_back({v, 99, "max at last position"});
    }
    // Mixed signs, max at the first position.
    {
        std::array<int8_t, kN> v;
        for (int i = 0; i < kN; i++) {
            v[i] = static_cast<int8_t>(-i);
        }
        v[0] = 77;
        cases.push_back({v, 77, "max at first position"});
    }

    for (const auto& c : cases) {
        step(sim, c.v, true);
        sim.check(y_dut(sim) == c.exp,
                  std::string("dut: ") + c.name);
        sim.check(y_ref(sim) == c.exp,
                  std::string("ref: ") + c.name);
        sim.check(sim.dut->mismatch_o == 0,
                  std::string("no mismatch: ") + c.name);
    }

    // ── Test 3: en=0 holds prior value through input changes ────
    printf("test 3: hold when en=0\n");
    // First load a known value.
    step(sim, filled(static_cast<int8_t>(42)), true);
    sim.check(y_dut(sim) == 42, "loaded 42");
    int8_t held = y_dut(sim);

    std::mt19937 hold_rng(0xDEADBEEFu);
    std::uniform_int_distribution<int> d8(-128, 127);
    for (int i = 0; i < 8; i++) {
        std::array<int8_t, kN> v;
        for (int j = 0; j < kN; j++) v[j] = static_cast<int8_t>(d8(hold_rng));
        step(sim, v, false);
        sim.check(y_dut(sim) == held, "y held while en=0");
        sim.check(sim.dut->mismatch_o == 0, "hold: no mismatch");
    }

    // ── Test 4: randomized stress (10000 patches) ───────
    printf("test 4: randomized stress (10000 patches)\n");
    std::mt19937 rng(0xC0FFEEu + kK);
    std::uniform_int_distribution<int> dprob(0, 99);

    int n_mismatch = 0;
    int n_shadow_fail = 0;
    int8_t expected_y = 42;  // matches the post-test-3 loaded value
    // After hold tests above, en was last 0 — so y still equals 42.

    const int Nrand = 10000;
    for (int i = 0; i < Nrand; i++) {
        std::array<int8_t, kN> v;
        for (int j = 0; j < kN; j++) v[j] = static_cast<int8_t>(d8(rng));
        bool en = (dprob(rng) < 80);

        int8_t this_max = shadow_max(v);
        step(sim, v, en);
        if (en) expected_y = this_max;

        if (sim.dut->mismatch_o) n_mismatch++;
        if (y_dut(sim) != expected_y) n_shadow_fail++;
    }
    sim.check(n_mismatch == 0,
              "rand: 0 cycle-by-cycle mismatches (was " +
              std::to_string(n_mismatch) + ")");
    sim.check(n_shadow_fail == 0,
              "rand: 0 dut-vs-C++-shadow mismatches (was " +
              std::to_string(n_shadow_fail) + ")");

    // ── Test 5: mid-stream reset clears y ───────────────
    printf("test 5: mid-stream reset clears y\n");
    step(sim, filled(static_cast<int8_t>(99)), true);
    sim.check(y_dut(sim) == 99, "y == 99 before reset");
    sim.dut->en_i = 0;
    set_x(sim.dut.get(), filled(0));
    sim.reset();
    sim.check(y_dut(sim) == 0, "dut y == 0 after re-reset");
    sim.check(y_ref(sim) == 0, "ref y == 0 after re-reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after re-reset");

    return sim.finish();
}
