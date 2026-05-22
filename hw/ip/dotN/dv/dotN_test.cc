// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// dotN — Verilator test.
//
// Drives DUT (dotN) and behavioral REF (dotN_ref) with identical
// stimulus via the dotN_tb wrapper. Every cycle we (a) assert the
// wrapper's mismatch_o is zero and (b) cross-check the DUT against an
// independent C++ shadow that mirrors the pipeline latency. This double
// check guards against the case where DUT and REF share the same bug.
//
// N is selected at compile time via -DDOTN_N=<n> (the Makefile passes
// the matching value to both Verilator and g++).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <random>
#include <string>
#include <vector>

#include "VdotN_tb.h"
#include "sim_ctrl.h"

#ifndef DOTN_N
#error "DOTN_N must be defined on the command line (e.g. -DDOTN_N=16)"
#endif

static constexpr int kN    = DOTN_N;
static constexpr int kBits = kN * 8;

// ceil(log2(N)) for N>=1.
static constexpr int clog2_n(int n) {
    int l = 0;
    int v = 1;
    while (v < n) { v <<= 1; l++; }
    return l;
}
static constexpr int kLatency = 1 + clog2_n(kN);

using DUT = VdotN_tb;

// Pack kN int8 lanes into a Verilator packed-array port. The same byte
// layout as the maxpool tests works here.
template <typename PortT>
static void set_lane(PortT& port, const std::vector<int8_t>& v) {
    constexpr size_t port_bytes = sizeof(PortT);
    static_assert(port_bytes * 8 >= static_cast<size_t>(kBits),
                  "port narrower than expected");
    std::memset(&port, 0, port_bytes);
    uint8_t* raw = reinterpret_cast<uint8_t*>(&port);
    for (int i = 0; i < kN; i++) {
        raw[i] = static_cast<uint8_t>(v[i]);
    }
}

static void set_a(DUT* dut, const std::vector<int8_t>& v) {
    set_lane(dut->a_i, v);
}
static void set_b(DUT* dut, const std::vector<int8_t>& v) {
    set_lane(dut->b_i, v);
}

static int32_t shadow_dot(const std::vector<int8_t>& a,
                          const std::vector<int8_t>& b) {
    int32_t s = 0;
    for (int i = 0; i < kN; i++) {
        s += int32_t(a[i]) * int32_t(b[i]);
    }
    return s;
}

static std::vector<int8_t> zeros() {
    return std::vector<int8_t>(kN, 0);
}
static std::vector<int8_t> filled(int8_t v) {
    return std::vector<int8_t>(kN, v);
}

static void step(SimCtrl<DUT>& s,
                 const std::vector<int8_t>& a,
                 const std::vector<int8_t>& b,
                 bool en,
                 bool clr) {
    set_a(s.dut.get(), a);
    set_b(s.dut.get(), b);
    s.dut->en_i  = en ? 1 : 0;
    s.dut->clr_i = clr ? 1 : 0;
    s.tick();
}

static int32_t y_dut(SimCtrl<DUT>& s) {
    return static_cast<int32_t>(s.dut->y_dut_o);
}
static bool valid_dut(SimCtrl<DUT>& s) {
    return s.dut->valid_dut_o != 0;
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 200000000ull;

    printf("==== dotN N=%d (latency=%d) ====\n", kN, kLatency);

    // ── Init ────────────────────────────────────────────
    sim.dut->en_i  = 0;
    sim.dut->clr_i = 0;
    set_a(sim.dut.get(), zeros());
    set_b(sim.dut.get(), zeros());
    sim.reset();

    // ── Test 1: post-reset state ────────────────────────
    printf("test 1: y/valid zero after reset\n");
    sim.check(y_dut(sim) == 0, "dut y == 0 after reset");
    sim.check(!valid_dut(sim), "dut valid == 0 after reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after reset");

    // Idle a few cycles with en=0 — outputs must stay zero.
    for (int i = 0; i < kLatency + 2; i++) {
        step(sim, filled(static_cast<int8_t>(i + 1)),
                  filled(static_cast<int8_t>(i + 2)),
                  /*en=*/false, /*clr=*/false);
        sim.check(y_dut(sim) == 0, "idle: y stays 0");
        sim.check(!valid_dut(sim), "idle: valid stays 0");
        sim.check(sim.dut->mismatch_o == 0, "idle: no mismatch");
    }

    // ── Test 2: pipeline fill — single en pulse ─────────
    printf("test 2: single en pulse appears at t=latency\n");
    // Drive en for one cycle with known inputs, then idle.
    {
        auto a = filled(static_cast<int8_t>(1));
        auto b = filled(static_cast<int8_t>(1));
        int32_t expected = shadow_dot(a, b);  // == N
        // After step 1 (above), valid stays low through the next
        // (LATENCY-2) idle cycles; only after the LATENCY-th step
        // does y_o reflect the data and valid_o rise. Apply LATENCY-1
        // idle cycles total: the first LATENCY-2 should still see
        // valid==0, then the (LATENCY-1)-th is the result cycle.
        step(sim, a, b, /*en=*/true, /*clr=*/false);
        for (int i = 1; i < kLatency - 1; i++) {
            step(sim, zeros(), zeros(), /*en=*/false, /*clr=*/false);
            sim.check(!valid_dut(sim),
                      "valid still 0 at cycle " + std::to_string(i));
            sim.check(sim.dut->mismatch_o == 0,
                      "fill: no mismatch @" + std::to_string(i));
        }
        // Final cycle — valid should rise on this edge.
        step(sim, zeros(), zeros(), /*en=*/false, /*clr=*/false);
        sim.check(valid_dut(sim), "valid asserts at t=latency");
        sim.check(y_dut(sim) == expected,
                  "y == N after single pulse (got " +
                  std::to_string(y_dut(sim)) + ", exp " +
                  std::to_string(expected) + ")");
        sim.check(sim.dut->mismatch_o == 0, "fill: no mismatch at exit");
    }

    // Drain.
    for (int i = 0; i < kLatency + 2; i++) {
        step(sim, zeros(), zeros(), false, false);
    }

    // ── Test 3: directed corners with sustained en ──────
    printf("test 3: directed corner vectors\n");
    struct Case {
        std::vector<int8_t> a;
        std::vector<int8_t> b;
        int32_t exp;
        const char* name;
    };
    std::vector<Case> cases;

    cases.push_back({zeros(), zeros(), 0, "all zeros"});
    cases.push_back({filled(127), filled(127),
                     int32_t(kN) * 127 * 127, "all +127 * +127"});
    cases.push_back({filled(-128), filled(-128),
                     int32_t(kN) * (-128) * (-128), "all -128 * -128"});
    cases.push_back({filled(-128), filled(127),
                     int32_t(kN) * (-128) * 127, "all -128 * +127"});

    // One non-zero lane only.
    {
        auto a = zeros(); auto b = zeros();
        a[0] = 127; b[0] = -128;
        cases.push_back({a, b, 127 * -128, "one lane 127 * -128"});
    }
    {
        auto a = zeros(); auto b = zeros();
        a[kN - 1] = 127; b[kN - 1] = -128;
        cases.push_back({a, b, 127 * -128, "last lane 127 * -128"});
    }
    // Alternating signs that cancel.
    {
        std::vector<int8_t> a(kN), b(kN);
        for (int i = 0; i < kN; i++) {
            a[i] = static_cast<int8_t>(((i & 1) ? -1 : 1) * 10);
            b[i] = 10;
        }
        cases.push_back({a, b, shadow_dot(a, b), "alternating-sign a"});
    }
    // Ramp.
    {
        std::vector<int8_t> a(kN), b(kN);
        for (int i = 0; i < kN; i++) {
            a[i] = static_cast<int8_t>(i - kN / 2);
            b[i] = static_cast<int8_t>(i + 1);
        }
        cases.push_back({a, b, shadow_dot(a, b), "ramp"});
    }

    for (const auto& c : cases) {
        // Apply on cycle 0, then wait latency cycles draining with en=0
        // and zero inputs so only the test pulse propagates.
        step(sim, c.a, c.b, /*en=*/true, /*clr=*/false);
        for (int i = 1; i < kLatency; i++) {
            step(sim, zeros(), zeros(), false, false);
        }
        sim.check(valid_dut(sim),
                  std::string("directed: valid @ latency: ") + c.name);
        sim.check(y_dut(sim) == c.exp,
                  std::string("directed: y match: ") + c.name +
                  " (got " + std::to_string(y_dut(sim)) +
                  ", exp " + std::to_string(c.exp) + ")");
        sim.check(sim.dut->mismatch_o == 0,
                  std::string("directed: no mismatch: ") + c.name);

        // Drain one more cycle so valid drops.
        step(sim, zeros(), zeros(), false, false);
        sim.check(!valid_dut(sim),
                  std::string("directed: valid clears: ") + c.name);
    }

    // ── Test 4: sustained throughput ────────────────────
    printf("test 4: sustained en — one valid result per cycle\n");
    {
        std::mt19937 rng(0x5EED1234u + kN);
        std::uniform_int_distribution<int> d8(-128, 127);
        std::deque<int32_t> shadow_q;

        const int kStream = 200;
        for (int t = 0; t < kStream + kLatency; t++) {
            std::vector<int8_t> a(kN), b(kN);
            bool en = (t < kStream);  // stop pushing past kStream
            int32_t exp = 0;
            if (en) {
                for (int i = 0; i < kN; i++) {
                    a[i] = static_cast<int8_t>(d8(rng));
                    b[i] = static_cast<int8_t>(d8(rng));
                }
                exp = shadow_dot(a, b);
            } else {
                std::fill(a.begin(), a.end(), 0);
                std::fill(b.begin(), b.end(), 0);
            }
            shadow_q.push_back(en ? exp : INT32_MIN);  // sentinel for !en

            step(sim, a, b, en, false);

            if (static_cast<int>(shadow_q.size()) >= kLatency) {
                int32_t want = shadow_q.front();
                shadow_q.pop_front();
                if (want == INT32_MIN) {
                    // we pushed en=0 LATENCY cycles ago — valid_o low
                    sim.check(!valid_dut(sim),
                              "stream: valid low when en was low @t=" +
                              std::to_string(t));
                } else {
                    sim.check(valid_dut(sim),
                              "stream: valid high @t=" + std::to_string(t));
                    sim.check(y_dut(sim) == want,
                              "stream: y match @t=" + std::to_string(t) +
                              " got " + std::to_string(y_dut(sim)) +
                              " exp " + std::to_string(want));
                }
                sim.check(sim.dut->mismatch_o == 0,
                          "stream: no mismatch @t=" + std::to_string(t));
            }
        }
    }

    // Drain.
    for (int i = 0; i < kLatency + 2; i++) {
        step(sim, zeros(), zeros(), false, false);
    }

    // ── Test 5: sparse en — token tracking ──────────────
    printf("test 5: sparse en pulses propagate one-for-one\n");
    {
        std::mt19937 rng(0xABCD0123u + kN);
        std::uniform_int_distribution<int> d8(-128, 127);
        std::uniform_int_distribution<int> dprob(0, 99);
        std::deque<std::pair<bool,int32_t>> q;  // (en, expected)

        const int kCycles = 500;
        for (int t = 0; t < kCycles + kLatency; t++) {
            bool en = (t < kCycles) && (dprob(rng) < 40);  // ~40% busy
            std::vector<int8_t> a(kN), b(kN);
            int32_t exp = 0;
            if (en) {
                for (int i = 0; i < kN; i++) {
                    a[i] = static_cast<int8_t>(d8(rng));
                    b[i] = static_cast<int8_t>(d8(rng));
                }
                exp = shadow_dot(a, b);
            }
            q.push_back({en, exp});

            step(sim, a, b, en, false);

            if (static_cast<int>(q.size()) >= kLatency) {
                auto [want_v, want_y] = q.front();
                q.pop_front();
                sim.check(valid_dut(sim) == want_v,
                          "sparse: valid match @t=" + std::to_string(t));
                if (want_v) {
                    sim.check(y_dut(sim) == want_y,
                              "sparse: y match @t=" + std::to_string(t) +
                              " got " + std::to_string(y_dut(sim)) +
                              " exp " + std::to_string(want_y));
                }
                sim.check(sim.dut->mismatch_o == 0,
                          "sparse: no mismatch @t=" + std::to_string(t));
            }
        }
    }

    // Drain.
    for (int i = 0; i < kLatency + 2; i++) {
        step(sim, zeros(), zeros(), false, false);
    }

    // ── Test 6: clr_i flushes the pipeline ──────────────
    printf("test 6: clr flushes pipeline regs to 0\n");
    {
        // Push a few en pulses, then assert clr mid-pipeline.
        for (int i = 0; i < kLatency; i++) {
            step(sim, filled(static_cast<int8_t>(50)),
                      filled(static_cast<int8_t>(50)),
                      /*en=*/true, /*clr=*/false);
        }
        // Pipeline is full of valid 50*50*N values now — assert clr.
        step(sim, zeros(), zeros(), /*en=*/false, /*clr=*/true);
        // After clr, valid should be 0 and y should be 0 within one
        // cycle. (clr is synchronous, so y_o updates next posedge.)
        sim.check(!valid_dut(sim), "valid==0 after clr cycle");
        sim.check(y_dut(sim) == 0, "y==0 after clr cycle");
        sim.check(sim.dut->mismatch_o == 0, "clr: no mismatch");

        // Idle a few cycles and re-check.
        for (int i = 0; i < kLatency + 2; i++) {
            step(sim, zeros(), zeros(), false, false);
            sim.check(!valid_dut(sim), "post-clr: valid stays 0");
            sim.check(y_dut(sim) == 0, "post-clr: y stays 0");
            sim.check(sim.dut->mismatch_o == 0, "post-clr: no mismatch");
        }
    }

    // ── Test 7: randomized stress, 10000 vectors ────────
    printf("test 7: randomized stress (10000 vectors)\n");
    {
        std::mt19937 rng(0xC0FFEEu + kN);
        std::uniform_int_distribution<int> d8(-128, 127);
        std::uniform_int_distribution<int> dprob(0, 99);
        std::deque<std::pair<bool,int32_t>> q;

        int n_mismatch = 0;
        int n_shadow_fail = 0;
        const int Nrand = 10000;

        for (int t = 0; t < Nrand + kLatency; t++) {
            bool en = (t < Nrand) && (dprob(rng) < 85);
            std::vector<int8_t> a(kN), b(kN);
            int32_t exp = 0;
            if (en) {
                for (int i = 0; i < kN; i++) {
                    a[i] = static_cast<int8_t>(d8(rng));
                    b[i] = static_cast<int8_t>(d8(rng));
                }
                exp = shadow_dot(a, b);
            }
            q.push_back({en, exp});

            step(sim, a, b, en, /*clr=*/false);

            if (sim.dut->mismatch_o) n_mismatch++;

            if (static_cast<int>(q.size()) >= kLatency) {
                auto [want_v, want_y] = q.front();
                q.pop_front();
                if (valid_dut(sim) != want_v) {
                    n_shadow_fail++;
                } else if (want_v && y_dut(sim) != want_y) {
                    n_shadow_fail++;
                }
            }
        }
        sim.check(n_mismatch == 0,
                  "rand: 0 cycle-by-cycle mismatches (was " +
                  std::to_string(n_mismatch) + ")");
        sim.check(n_shadow_fail == 0,
                  "rand: 0 dut-vs-C++-shadow mismatches (was " +
                  std::to_string(n_shadow_fail) + ")");
    }

    // ── Test 8: mid-stream reset ────────────────────────
    printf("test 8: mid-stream reset clears pipeline\n");
    step(sim, filled(static_cast<int8_t>(100)),
              filled(static_cast<int8_t>(100)),
              /*en=*/true, false);
    sim.dut->en_i  = 0;
    sim.dut->clr_i = 0;
    set_a(sim.dut.get(), zeros());
    set_b(sim.dut.get(), zeros());
    sim.reset();
    sim.check(y_dut(sim) == 0, "y==0 after re-reset");
    sim.check(!valid_dut(sim), "valid==0 after re-reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after re-reset");

    return sim.finish();
}
