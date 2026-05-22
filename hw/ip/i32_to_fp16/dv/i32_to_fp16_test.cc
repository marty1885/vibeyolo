// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// i32_to_fp16 — Verilator test.
//
// Drives DUT (i32_to_fp16) and behavioral REF (i32_to_fp16_ref) with the
// same int32 stimulus via i32_to_fp16_tb. Every cycle we (a) assert the
// wrapper's mismatch_o is zero and (b) cross-check the DUT against an
// independent C++ shadow that performs the int32->fp16 RNE conversion in
// pure integer arithmetic. The shadow doesn't go through host floats, so
// a shared SV bug in DUT+REF (both of which lean on similar bit-tricks)
// would still be flagged.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "Vi32_to_fp16_tb.h"
#include "sim_ctrl.h"

using DUT = Vi32_to_fp16_tb;

// Independent C++ reference: signed int32 -> (fp16, shift) where the IP
// auto-prescales by 2^shift to keep the fp16 representable. For |x| < 2^16
// shift==0 and the conversion is the legacy bit-exact RNE int32→fp16.
// Pure integer; no float involvement.
struct ShadowOut { uint16_t fp16; uint8_t shift; };

static ShadowOut shadow_fp16(int32_t x) {
    if (x == 0) return {0x0000, 0};
    uint16_t sign = (x < 0) ? 0x8000 : 0x0000;
    uint32_t mag = (x < 0) ? (uint32_t)(-(int64_t)x) : (uint32_t)x;

    // Find MSB position (0..31).
    int msb = 31;
    while (((mag >> msb) & 1u) == 0u) msb--;

    int pre_shift = (msb <= 14) ? 0 : (msb - 14);
    uint32_t mag_sh        = mag >> pre_shift;
    uint32_t shifted_lobits = (pre_shift > 0) ? (mag & ((1u << pre_shift) - 1u)) : 0u;
    uint32_t extra_sticky  = shifted_lobits ? 1u : 0u;
    int msb_post = msb - pre_shift;

    int biased = msb_post + 15;

    // Align so implicit-1 lives at bit 31.
    uint32_t aligned = (msb_post >= 31) ? mag_sh : (mag_sh << (31 - msb_post));
    uint32_t mant   = (aligned >> 21) & 0x3FFu;
    uint32_t guard  = (aligned >> 20) & 0x1u;
    uint32_t sticky = ((aligned & 0xFFFFFu) ? 1u : 0u) | extra_sticky;
    uint32_t lsb    = mant & 1u;

    uint32_t round_up = (guard && (sticky || lsb)) ? 1u : 0u;
    uint32_t mant_r   = mant + round_up;
    if (mant_r & 0x400u) {
        biased += 1;
        mant_r  = 0;
    }
    uint16_t y;
    if (biased >= 31) y = (uint16_t)(sign | 0x7C00);
    else              y = (uint16_t)(sign | ((biased & 0x1F) << 10) | (mant_r & 0x3FF));
    return { y, (uint8_t)pre_shift };
}

// Convenience wrapper that returns just the fp16 value (legacy callers).
static uint16_t shadow_fp16_y(int32_t x) { return shadow_fp16(x).fp16; }

static void apply(SimCtrl<DUT>& s, int32_t x) {
    s.dut->x_i = (uint32_t)x;
    s.tick();
}

static uint16_t y_dut(SimCtrl<DUT>& s) { return (uint16_t)s.dut->y_dut_o; }
static uint16_t y_ref(SimCtrl<DUT>& s) { return (uint16_t)s.dut->y_ref_o; }
static uint8_t  sh_dut(SimCtrl<DUT>& s) { return (uint8_t)s.dut->shift_dut_o; }
static uint8_t  sh_ref(SimCtrl<DUT>& s) { return (uint8_t)s.dut->shift_ref_o; }

// Reconstruct (approx) the magnitude of acc from (fp16_y, shift). For
// |acc| < 2^16 this is exact (ignoring RNE last bit). For larger acc
// it should be within fp16 ULP at exponent `shift`.
static double reconstruct(uint16_t fp16, uint8_t shift) {
    uint16_t sign_b = (fp16 >> 15) & 1;
    uint16_t bexp   = (fp16 >> 10) & 0x1F;
    uint16_t frac   = fp16 & 0x3FF;
    double  val;
    if (bexp == 0x1F) {
        return sign_b ? -1e300 : 1e300;  // ±Inf-ish sentinel
    } else if (bexp == 0) {
        val = (double)frac * (1.0 / (1 << 24));
    } else {
        val = (1024.0 + frac) * std::ldexp(1.0, (int)bexp - 25);
    }
    if (sign_b) val = -val;
    return std::ldexp(val, shift);
}

static std::string hex16(uint16_t v) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "0x%04X", v);
    return std::string(buf);
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 400000000ull;

    sim.dut->x_i = 0;
    sim.reset();

    // ─── Test 1: reset ──────────────────────────────────
    printf("test 1: y_o == 0 after reset\n");
    sim.check(y_dut(sim) == 0x0000, "dut y == 0 after reset");
    sim.check(y_ref(sim) == 0x0000, "ref y == 0 after reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after reset");

    // ─── Test 2: directed values ────────────────────────
    printf("test 2: directed cases\n");
    // Directed cases that fit in fp16 unaided: shift must be 0 and the
    // fp16 output must match the legacy behavior bit-exactly.
    struct Case { int32_t x; uint16_t exp; const char* desc; };
    const Case cases[] = {
        { 0,           0x0000, "0 -> +0" },
        { 1,           0x3C00, "1 -> 1.0" },
        { -1,          0xBC00, "-1 -> -1.0" },
        { 2,           0x4000, "2" },
        { -2,          0xC000, "-2" },
        { 3,           0x4200, "3" },
        { 1024,        0x6400, "1024" },
        { -1024,       0xE400, "-1024" },
        { 2048,        0x6800, "2048 (exact)" },
        { 2049,        0x6800, "2049 RNE -> 2048 (even)" },
        { 2050,        0x6801, "2050 exact at bit10" },
        { 2051,        0x6802, "2051 RNE -> 2052 (even)" },
        { 32767,       0x7800, "max no-shift (msb=14): 32767 RNE -> 32768" },
        { -32767,      0xF800, "-32767 RNE -> -32768" },
        // |x| in [32768, 65535] (msb=15) now PRESCALES (shift=1)
        // instead of fitting at shift=0 — tested in 2b below.
    };
    for (auto& c : cases) {
        apply(sim, c.x);
        std::string d = std::string(c.desc) + " got=" + hex16(y_dut(sim)) +
                        " exp=" + hex16(c.exp);
        sim.check(y_dut(sim) == c.exp, "dut " + d);
        sim.check(y_ref(sim) == c.exp, "ref " + d);
        sim.check(sh_dut(sim) == 0, "shift==0 for fit-case: " + std::string(c.desc));
        sim.check(sim.dut->mismatch_o == 0, "no mismatch: " + std::string(c.desc));
    }

    // Directed cases that require prescale: shift > 0 and the
    // reconstructed value (fp16 * 2^shift) must match acc within ULP.
    printf("test 2b: directed prescale cases\n");
    struct PCase { int32_t x; uint8_t shift; const char* desc; };
    const PCase pcases[] = {
        { 65536,        2, "2^16 -> shift=2 (msb=16)" },
        { 100000,       2, "100000 -> shift=2 (msb=16)" },
        { -100000,      2, "-100000 -> shift=2" },
        { 1 << 17,      3, "2^17 -> shift=3" },
        { 1 << 20,      6, "2^20 -> shift=6" },
        { 1 << 25,     11, "2^25 -> shift=11" },
        { 1 << 30,     16, "2^30 -> shift=16" },
        { INT32_MAX,   16, "INT32_MAX -> shift=16 (msb=30)" },
        { INT32_MIN,   17, "INT32_MIN -> shift=17 (msb=31)" },
        { 10'000'000,   9, "10M -> shift=9 (msb=23)" },
    };
    for (auto& c : pcases) {
        apply(sim, c.x);
        std::string d = std::string(c.desc) + " got_y=" + hex16(y_dut(sim)) +
                        " got_sh=" + std::to_string((int)sh_dut(sim)) +
                        " exp_sh=" + std::to_string((int)c.shift);
        sim.check(sh_dut(sim) == c.shift, "dut shift " + d);
        sim.check(sh_ref(sim) == c.shift, "ref shift " + d);
        sim.check(sim.dut->mismatch_o == 0, "no mismatch: " + std::string(c.desc));
        // Reconstructed magnitude must be within one fp16 ULP at the
        // shifted exponent: ULP_total = 2^(fp16_exp - 25 + shift).
        double recon = reconstruct(y_dut(sim), sh_dut(sim));
        double truth = (double)c.x;
        int fp16_exp = (y_dut(sim) >> 10) & 0x1F;
        double ulp_at = std::ldexp(1.0, fp16_exp - 25 + (int)c.shift);
        sim.check(std::fabs(recon - truth) <= ulp_at,
                  "reconstruct within ulp: " + std::string(c.desc) +
                  " recon=" + std::to_string(recon) +
                  " truth=" + std::to_string(truth) +
                  " ulp=" + std::to_string(ulp_at));
    }

    // ─── Test 3: RNE boundary sweep around bit 10 ───────
    // For x in [2^11, 2^12) the ulp is 2; check tie behavior.
    printf("test 3: RNE tie sweep\n");
    for (int32_t x = 2048; x <= 2060; x++) {
        apply(sim, x);
        uint16_t exp = shadow_fp16_y(x);
        sim.check(y_dut(sim) == exp,
                  "dut RNE x=" + std::to_string(x) +
                  " got=" + hex16(y_dut(sim)) + " exp=" + hex16(exp));
        sim.check(sim.dut->mismatch_o == 0, "RNE no mismatch");
    }

    // ─── Test 4: overflow boundary sweep ────────────────
    printf("test 4: overflow boundary\n");
    for (int32_t x = 65500; x <= 65540; x++) {
        apply(sim, x);
        uint16_t exp = shadow_fp16_y(x);
        sim.check(y_dut(sim) == exp,
                  "dut ovf x=" + std::to_string(x) +
                  " got=" + hex16(y_dut(sim)) + " exp=" + hex16(exp));
        sim.check(sim.dut->mismatch_o == 0, "ovf no mismatch");
    }

    // ─── Test 4b: large-magnitude boundary sweep ────────
    printf("test 4b: large magnitude sweep (shift path)\n");
    {
        int32_t xs[] = { (int32_t)(1<<17), (int32_t)(1<<20), (int32_t)(1<<25),
                         (int32_t)(1<<30), INT32_MAX-1, INT32_MAX, INT32_MIN+1,
                         -(int32_t)(1<<17), -(int32_t)(1<<20), -(int32_t)(1<<25),
                         -(int32_t)(1<<30) };
        for (int32_t x : xs) {
            apply(sim, x);
            ShadowOut sh = shadow_fp16(x);
            sim.check(y_dut(sim) == sh.fp16,
                      "large dut y matches shadow x=" + std::to_string((long long)x));
            sim.check(sh_dut(sim) == sh.shift,
                      "large dut shift matches shadow x=" + std::to_string((long long)x));
            sim.check(sim.dut->mismatch_o == 0, "no mismatch large x");
        }
    }

    // ─── Test 5: randomized stress ──────────────────────
    printf("test 5: 50000 random int32 values\n");
    std::mt19937 rng(0xBADC0FFEu);
    std::uniform_int_distribution<int64_t> d32(INT32_MIN, INT32_MAX);
    // Also throw in a high-density sweep around small magnitudes where
    // mantissa rounding is most interesting.
    std::uniform_int_distribution<int>     dsmall(-200000, 200000);
    std::uniform_int_distribution<int>     bucket(0, 9);

    int n_mismatch    = 0;
    int n_shadow_fail = 0;
    const int N = 50000;
    for (int i = 0; i < N; i++) {
        int32_t x;
        if (bucket(rng) < 4) x = (int32_t)dsmall(rng);
        else                 x = (int32_t)d32(rng);

        apply(sim, x);
        ShadowOut sho = shadow_fp16(x);
        uint16_t exp = sho.fp16;
        if (sh_dut(sim) != sho.shift) {
            if (n_shadow_fail < 5) {
                printf("  shift mismatch @i=%d x=%d dut_sh=%d sh_sh=%d\n",
                       i, x, (int)sh_dut(sim), (int)sho.shift);
            }
            n_shadow_fail++;
        }
        if (sim.dut->mismatch_o) {
            if (n_mismatch < 5) {
                printf("  mismatch @i=%d x=%d dut=%s ref=%s\n",
                       i, x, hex16(y_dut(sim)).c_str(), hex16(y_ref(sim)).c_str());
            }
            n_mismatch++;
        }
        if (y_dut(sim) != exp) {
            if (n_shadow_fail < 5) {
                printf("  shadow @i=%d x=%d dut=%s shadow=%s\n",
                       i, x, hex16(y_dut(sim)).c_str(), hex16(exp).c_str());
            }
            n_shadow_fail++;
        }
    }
    sim.check(n_mismatch == 0,
              "rand: 0 dut-vs-ref mismatches (was " +
              std::to_string(n_mismatch) + ")");
    sim.check(n_shadow_fail == 0,
              "rand: 0 dut-vs-C++-shadow mismatches (was " +
              std::to_string(n_shadow_fail) + ")");

    // ─── Test 6: mid-stream reset ───────────────────────
    printf("test 6: mid-stream reset clears y\n");
    apply(sim, 12345);
    sim.check(y_dut(sim) != 0, "y non-zero before reset");
    sim.dut->x_i = 0;
    sim.reset();
    sim.check(y_dut(sim) == 0, "dut y==0 after re-reset");
    sim.check(y_ref(sim) == 0, "ref y==0 after re-reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after re-reset");

    return sim.finish();
}
