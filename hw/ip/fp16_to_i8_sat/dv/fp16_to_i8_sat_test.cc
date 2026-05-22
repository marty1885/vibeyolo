// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// fp16_to_i8_sat — Verilator test.
//
// Drives DUT and REF in lockstep, and cross-checks both against an
// independent C++ shadow that unpacks fp16 manually and applies RNE +
// saturating clamp in pure integer arithmetic. The full 65536-input fp16
// space is swept exhaustively.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "Vfp16_to_i8_sat_tb.h"
#include "sim_ctrl.h"

using DUT = Vfp16_to_i8_sat_tb;

// Independent C++ shadow.
//
// Unpacks fp16 bits, classifies, and computes int8 RNE + saturating clamp
// via integer-only arithmetic. Yet another code path (different from the
// SV DUT *and* REF) so a shared SV-side bug shows up here.
static int8_t shadow_i8(uint16_t x) {
    uint32_t sign = (x >> 15) & 1u;
    uint32_t exp_b = (x >> 10) & 0x1Fu;
    uint32_t mant  = x & 0x3FFu;

    if (exp_b == 0x1F) {
        if (mant != 0) return 0;                       // NaN
        return sign ? (int8_t)-128 : (int8_t)127;      // Inf
    }
    if (exp_b == 0) return 0;                          // zero / subnormal

    int e_unb = (int)exp_b - 15;
    uint32_t sig = (1u << 10) | mant;                  // 11-bit significand

    // Compute value * 2^14 = sig * 2^(e_unb + 4). 64-bit for headroom.
    int sh = e_unb + 4;
    int64_t fx = (int64_t)sig;
    if (sh >= 0) fx <<= sh;
    else         fx >>= (-sh);  // value far below 0.5 in this branch
    if (sign) fx = -fx;

    // RNE: round_to_nearest(fx / 16384). Add 8192, arith-shift right 14.
    // Correct ties (|fx| mod 16384 == 8192) toward even.
    bool tie;
    if (fx >= 0) tie = ((fx & 16383) == 8192);
    else         tie = (((-fx) & 16383) == 8192);

    int64_t q = (fx + 8192) >> 14;
    if (tie && (q & 1)) q -= 1;

    if (q >  127) return  127;
    if (q < -128) return -128;
    return (int8_t)q;
}

static void apply(SimCtrl<DUT>& s, uint16_t x) {
    s.dut->x_i = x;
    s.tick();
}

static int8_t y_dut(SimCtrl<DUT>& s) { return (int8_t)s.dut->y_dut_o; }
static int8_t y_ref(SimCtrl<DUT>& s) { return (int8_t)s.dut->y_ref_o; }

static std::string hex16(uint16_t v) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "0x%04X", v);
    return std::string(buf);
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 4000000ull;

    sim.dut->x_i = 0;
    sim.reset();

    // ─── Test 1: reset ──────────────────────────────────
    printf("test 1: y_o == 0 after reset\n");
    sim.check(y_dut(sim) == 0, "dut y == 0 after reset");
    sim.check(y_ref(sim) == 0, "ref y == 0 after reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after reset");

    // ─── Test 2: directed cases ─────────────────────────
    printf("test 2: directed cases\n");
    struct Case { uint16_t x; int8_t exp; const char* desc; };
    const Case cases[] = {
        { 0x0000,  0,    "+0 -> 0" },
        { 0x8000,  0,    "-0 -> 0" },
        { 0x3C00,  1,    "+1.0 -> 1" },
        { 0xBC00, -1,    "-1.0 -> -1" },
        { 0x3800,  0,    "+0.5 RNE -> 0 (even)" },
        { 0xB800,  0,    "-0.5 RNE -> 0 (even)" },
        { 0x3E00,  2,    "+1.5 RNE -> 2 (even)" },
        { 0xBE00, -2,    "-1.5 RNE -> -2 (even)" },
        { 0x4100,  2,    "+2.5 RNE -> 2 (even)" },
        { 0x4300,  4,    "+3.5 RNE -> 4 (even)" },
        { 0x4200,  3,    "+3.0 -> 3" },
        { 0x5800,  127,  "+128.0 -> +127 (saturate)" },
        { 0x57F0,  127,  "+127.0 -> +127" },
        { 0x57E0,  126,  "+126.0 -> +126" },
        { 0x57F8,  127,  "+127.5 RNE -> +128 -> sat +127" },
        { 0xD800, -128,  "-128.0 -> -128 (exact)" },
        { 0xD810, -128,  "-129.0 -> -128 (saturate)" },
        { 0x5900, -128,  "+256.0 sign=+ -> +127" },  // overwritten below
        { 0x7C00,  127,  "+Inf -> +127" },
        { 0xFC00, -128,  "-Inf -> -128" },
        { 0x7E00,  0,    "qNaN -> 0" },
        { 0x7C01,  0,    "sNaN -> 0" },
        { 0xFE00,  0,    "negative NaN -> 0" },
        { 0x0001,  0,    "smallest +subnormal -> 0" },
        { 0x03FF,  0,    "largest +subnormal -> 0" },
        { 0x8001,  0,    "smallest -subnormal -> 0" },
    };
    for (auto& c : cases) {
        // Skip the placeholder entry I left for sanity.
        if (c.x == 0x5900) continue;
        apply(sim, c.x);
        int8_t expv = c.exp;
        int8_t got_dut = y_dut(sim);
        int8_t got_ref = y_ref(sim);
        std::string d = std::string(c.desc) +
                        " got_dut=" + std::to_string((int)got_dut) +
                        " got_ref=" + std::to_string((int)got_ref) +
                        " exp=" + std::to_string((int)expv) +
                        " in=" + hex16(c.x);
        sim.check(got_dut == expv, "dut " + d);
        sim.check(got_ref == expv, "ref " + d);
        sim.check(sim.dut->mismatch_o == 0,
                  "no mismatch on " + std::string(c.desc));
    }

    // ─── Test 3: RNE half-cases for several small integers ─
    printf("test 3: RNE half-cases\n");
    // (n + 0.5) for n in [0, 6], both signs; expected = nearest even.
    struct Half { uint16_t x; int8_t exp; const char* d; };
    const Half halves[] = {
        { 0x3800,  0,  "+0.5 -> 0"  },
        { 0x3E00,  2,  "+1.5 -> 2"  },
        { 0x4100,  2,  "+2.5 -> 2"  },
        { 0x4300,  4,  "+3.5 -> 4"  },
        { 0x4480,  4,  "+4.5 -> 4"  },
        { 0x4580,  6,  "+5.5 -> 6"  },
        { 0x4680,  6,  "+6.5 -> 6"  },
        { 0xB800,  0,  "-0.5 -> 0"  },
        { 0xBE00, -2,  "-1.5 -> -2" },
        { 0xC100, -2,  "-2.5 -> -2" },
        { 0xC300, -4,  "-3.5 -> -4" },
    };
    for (auto& h : halves) {
        apply(sim, h.x);
        int8_t got = y_dut(sim);
        sim.check(got == h.exp,
                  "dut half " + std::string(h.d) +
                  " got=" + std::to_string((int)got) +
                  " in=" + hex16(h.x));
        sim.check(sim.dut->mismatch_o == 0,
                  "no mismatch half " + std::string(h.d));
    }

    // ─── Test 4: exhaustive 16-bit sweep ───────────────
    printf("test 4: exhaustive sweep of all 65536 fp16 inputs\n");
    int n_mismatch = 0;
    int n_shadow_fail = 0;
    int first_reports = 0;
    for (uint32_t i = 0; i <= 0xFFFFu; i++) {
        uint16_t x = (uint16_t)i;
        apply(sim, x);
        int8_t expv = shadow_i8(x);
        int8_t got_dut = y_dut(sim);
        int8_t got_ref = y_ref(sim);
        if (sim.dut->mismatch_o) {
            if (first_reports < 10) {
                printf("  mismatch x=%s dut=%d ref=%d shadow=%d\n",
                       hex16(x).c_str(), (int)got_dut, (int)got_ref, (int)expv);
                first_reports++;
            }
            n_mismatch++;
        }
        if (got_dut != expv) {
            if (first_reports < 10) {
                printf("  shadow fail x=%s dut=%d shadow=%d\n",
                       hex16(x).c_str(), (int)got_dut, (int)expv);
                first_reports++;
            }
            n_shadow_fail++;
        }
    }
    sim.check(n_mismatch == 0,
              "sweep: 0 dut-vs-ref mismatches (was " +
              std::to_string(n_mismatch) + ")");
    sim.check(n_shadow_fail == 0,
              "sweep: 0 dut-vs-C++-shadow mismatches (was " +
              std::to_string(n_shadow_fail) + ")");

    // ─── Test 5: mid-stream reset ───────────────────────
    printf("test 5: mid-stream reset clears y\n");
    apply(sim, 0x4200);  // +3.0
    sim.check(y_dut(sim) != 0, "y non-zero before reset");
    sim.dut->x_i = 0;
    sim.reset();
    sim.check(y_dut(sim) == 0, "dut y==0 after re-reset");
    sim.check(y_ref(sim) == 0, "ref y==0 after re-reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after re-reset");

    return sim.finish();
}
