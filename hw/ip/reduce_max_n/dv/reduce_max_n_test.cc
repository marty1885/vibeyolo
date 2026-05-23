// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// reduce_max_n — Verilator test.
//
// Drives DUT (reduce_max_n) and behavioral REF (reduce_max_n_ref) with
// identical stimulus via reduce_max_n_tb. Every cycle we (a) assert the
// wrapper's mismatch_o is zero and (b) cross-check the DUT against an
// independent C++ shadow max (std::max_element). N is selected at compile
// time via -DRM_N=<n> (the Makefile passes the matching value to both
// Verilator and g++).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "Vreduce_max_n_tb.h"
#include "sim_ctrl.h"

#ifndef RM_N
#error "RM_N must be defined on the command line (e.g. -DRM_N=80)"
#endif

static constexpr int kN = RM_N;
static constexpr int kBits = kN * 8;

using DUT = Vreduce_max_n_tb;

// Pack kN int8 lanes into the DUT's x_i port. Verilator backs the packed
// signal with a scalar (<=64b) or a 32-bit word array (>64b); either way
// the little-endian byte layout lets us write lane i at byte i.
static void set_x(DUT* dut, const std::vector<int8_t>& v) {
    constexpr size_t port_bytes = sizeof(dut->x_i);
    static_assert(port_bytes * 8 >= static_cast<size_t>(kBits),
                  "x_i port narrower than expected");
    std::memset(&dut->x_i, 0, port_bytes);
    uint8_t* raw = reinterpret_cast<uint8_t*>(&dut->x_i);
    for (int i = 0; i < kN; i++) {
        raw[i] = static_cast<uint8_t>(v[i]);
    }
}

static void step(SimCtrl<DUT>& s, const std::vector<int8_t>& v, bool en) {
    set_x(s.dut.get(), v);
    s.dut->en_i = en ? 1 : 0;
    s.tick();
}

static int8_t y_dut(SimCtrl<DUT>& s) { return static_cast<int8_t>(s.dut->y_dut_o); }
static int8_t y_ref(SimCtrl<DUT>& s) { return static_cast<int8_t>(s.dut->y_ref_o); }

static int8_t shadow_max(const std::vector<int8_t>& v) {
    return *std::max_element(v.begin(), v.end());
}
static std::vector<int8_t> filled(int8_t val) {
    return std::vector<int8_t>(kN, val);
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 200000000ull;

    printf("==== reduce_max_n N=%d ====\n", kN);

    sim.dut->en_i = 0;
    set_x(sim.dut.get(), filled(0));
    sim.reset();

    // ── Test 1: reset state ─────────────────────────────
    printf("test 1: y zero / valid low after reset\n");
    sim.check(y_dut(sim) == 0, "dut y == 0 after reset");
    sim.check(sim.dut->valid_dut_o == 0, "dut valid low after reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after reset");
    for (int i = 0; i < 4; i++) {
        step(sim, filled(static_cast<int8_t>(50 + i)), false);
        sim.check(y_dut(sim) == 0, "idle: y holds 0");
        sim.check(sim.dut->valid_dut_o == 0, "idle: valid low");
        sim.check(sim.dut->mismatch_o == 0, "idle: no mismatch");
    }

    // ── Test 2: directed corners ────────────────────────
    printf("test 2: directed corner vectors\n");
    struct Case { std::vector<int8_t> v; int8_t exp; const char* name; };
    std::vector<Case> cases;
    cases.push_back({filled(0), 0, "all zero"});
    cases.push_back({filled(-128), -128, "all -128"});
    cases.push_back({filled(127), 127, "all +127"});
    { auto v = filled(-128); v[kN/2] = 127; cases.push_back({v,127,"one +127 among -128s"}); }
    { auto v = filled(127);  v[0] = -128;   cases.push_back({v,127,"one -128 among +127s"}); }
    { auto v = filled(-100); v[kN-1] = 50;  cases.push_back({v,50, "one +50 in -100s"}); }
    { std::vector<int8_t> v(kN); for (int i=0;i<kN;i++) v[i]=static_cast<int8_t>(120-i);
      cases.push_back({v,120,"descending"}); }
    { std::vector<int8_t> v(kN); for (int i=0;i<kN;i++) v[i]=static_cast<int8_t>(-50+i);
      cases.push_back({v,static_cast<int8_t>(-50+kN-1),"ascending (clamped <=127)"}); }
    { std::vector<int8_t> v(kN); for (int i=0;i<kN;i++) v[i]=static_cast<int8_t>(-i);
      v[0]=77; cases.push_back({v,77,"max at first position"}); }
    for (auto& c : cases) {
        // recompute clamp-safe expected via shadow to avoid arithmetic surprises
        int8_t exp = shadow_max(c.v);
        step(sim, c.v, true);
        sim.check(y_dut(sim) == exp, std::string("dut: ") + c.name);
        sim.check(y_ref(sim) == exp, std::string("ref: ") + c.name);
        sim.check(sim.dut->valid_dut_o == 1, std::string("valid: ") + c.name);
        sim.check(sim.dut->mismatch_o == 0, std::string("no mismatch: ") + c.name);
    }

    // ── Test 3: hold when en=0 ──────────────────────────
    printf("test 3: hold when en=0\n");
    step(sim, filled(42), true);
    sim.check(y_dut(sim) == 42, "loaded 42");
    int8_t held = y_dut(sim);
    std::mt19937 hold_rng(0xDEADBEEFu);
    std::uniform_int_distribution<int> d8(-128, 127);
    for (int i = 0; i < 8; i++) {
        std::vector<int8_t> v(kN);
        for (int j = 0; j < kN; j++) v[j] = static_cast<int8_t>(d8(hold_rng));
        step(sim, v, false);
        sim.check(y_dut(sim) == held, "y held while en=0");
        sim.check(sim.dut->valid_dut_o == 0, "valid low while en=0");
        sim.check(sim.dut->mismatch_o == 0, "hold: no mismatch");
    }

    // ── Test 4: randomized stress ───────────────────────
    printf("test 4: randomized stress (10000 vectors)\n");
    std::mt19937 rng(0xC0FFEEu + kN);
    std::uniform_int_distribution<int> dprob(0, 99);
    int n_mismatch = 0, n_shadow_fail = 0;
    int8_t expected_y = 42;  // post-test-3 held value
    for (int i = 0; i < 10000; i++) {
        std::vector<int8_t> v(kN);
        for (int j = 0; j < kN; j++) v[j] = static_cast<int8_t>(d8(rng));
        bool en = (dprob(rng) < 80);
        int8_t this_max = shadow_max(v);
        step(sim, v, en);
        if (en) expected_y = this_max;
        if (sim.dut->mismatch_o) n_mismatch++;
        if (y_dut(sim) != expected_y) n_shadow_fail++;
    }
    sim.check(n_mismatch == 0,
              "rand: 0 cycle mismatches (was " + std::to_string(n_mismatch) + ")");
    sim.check(n_shadow_fail == 0,
              "rand: 0 dut-vs-shadow (was " + std::to_string(n_shadow_fail) + ")");

    // ── Test 5: mid-stream reset ────────────────────────
    printf("test 5: mid-stream reset clears y\n");
    step(sim, filled(99), true);
    sim.check(y_dut(sim) == 99, "y == 99 before reset");
    sim.dut->en_i = 0;
    set_x(sim.dut.get(), filled(0));
    sim.reset();
    sim.check(y_dut(sim) == 0, "dut y == 0 after re-reset");
    sim.check(sim.dut->valid_dut_o == 0, "valid low after re-reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after re-reset");

    return sim.finish();
}
