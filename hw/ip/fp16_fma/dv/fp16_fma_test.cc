// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// fp16_fma — Verilator test. Drives DUT and REF with the same fp16
// triples and cross-checks both against an independent C++ shadow that
// computes the fp16 fused multiply-add (single rounding, RNE) using
// pure integer arithmetic with a __int128 accumulator. The shadow uses
// yet a third layout (different anchor, different normalisation) so
// shared bugs across the two SV implementations and the C++ shadow are
// implausible.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include "Vfp16_fma_tb.h"
#include "sim_ctrl.h"

using DUT = Vfp16_fma_tb;

// ─── unpack fp16 ──────────────────────────────────────────────
struct Unpack {
    int sign;
    int kind;       // 0 zero, 1 normal, 2 subnormal, 3 inf, 4 nan
    // value = sig * 2^exp; sig is non-negative.
    // For normal:    sig in [2^23, 2^24-1] (we shift implicit-1 up to bit 23)
    // For subnormal: sig in [0, 2^23) at bits [22:13]
    // Both forms set exp consistently so total magnitude is correct.
    __int128 sig;
    int      exp;
};

static Unpack unpack16(uint16_t x) {
    Unpack u;
    u.sign = (x >> 15) & 1;
    int bexp = (x >> 10) & 0x1F;
    int frac = x & 0x3FF;
    if (bexp == 0 && frac == 0) {
        u.kind = 0; u.sig = 0; u.exp = 0;
    } else if (bexp == 31 && frac == 0) {
        u.kind = 3; u.sig = 0; u.exp = 0;
    } else if (bexp == 31) {
        u.kind = 4; u.sig = 0; u.exp = 0;
    } else if (bexp == 0) {
        // subnormal: value = frac * 2^-24
        // place frac at bits [22:13] of a 24-bit sig → sig = frac<<13,
        // exp = -24 - 13 = -37
        u.kind = 2;
        u.sig  = (__int128)(uint64_t)frac << 13;
        u.exp  = -37;
    } else {
        // normal: value = (1024+frac) * 2^(bexp-25)
        // place (1024+frac) at bits [23:13] → sig = (1024+frac)<<13,
        // exp = (bexp-25) - 13 = bexp - 38
        u.kind = 1;
        u.sig  = (__int128)(uint64_t)(1024 + frac) << 13;
        u.exp  = bexp - 38;
    }
    return u;
}

static uint16_t pack_inf(int sign)  { return (uint16_t)((sign << 15) | 0x7C00); }
static uint16_t pack_nan()          { return 0x7E00; }
static uint16_t pack_zero(int sign) { return (uint16_t)(sign << 15); }

// Compute fp16 FMA via pure integer arithmetic in __int128.
//
// Strategy (deliberately different from DUT and REF):
//   - Pre-compute product (sig24*sig24 = 48-bit unsigned) and product exp.
//   - Pick anchor exp = MIN(product_exp, c_exp). Shift both up to anchor
//     (no precision loss).
//   - Add/sub magnitudes.
//   - Find MSB of the 128-bit magnitude with __builtin_clzll-style scan.
//   - Place leading-1 at bit 10 of a result window; capture guard, round,
//     sticky.
//   - RNE round; handle subnormal/overflow on output.
static uint16_t fp16_fma_shadow(uint16_t a, uint16_t b, uint16_t c) {
    Unpack ua = unpack16(a), ub = unpack16(b), uc = unpack16(c);

    // NaN propagation, Inf*0, Inf+(-Inf).
    if (ua.kind == 4 || ub.kind == 4 || uc.kind == 4) return pack_nan();
    bool prod_inf  = (ua.kind == 3 || ub.kind == 3);
    bool prod_zero = (ua.kind == 0 || ub.kind == 0);
    if (prod_inf && prod_zero) return pack_nan();
    int sp = ua.sign ^ ub.sign;
    if (prod_inf && uc.kind == 3 && sp != uc.sign) return pack_nan();
    if (prod_inf) return pack_inf(sp);
    if (uc.kind == 3) return pack_inf(uc.sign);

    // Product (48-bit exact).
    __int128 sigprod = ua.sig * ub.sig;  // 24*24 → 48 bit at most
    int      eprod   = ua.exp + ub.exp;

    bool p_z = prod_zero || (sigprod == 0);
    bool c_z = (uc.kind == 0);

    if (p_z && c_z) return 0x0000;  // +0

    int anchor;
    if (p_z)        anchor = uc.exp;
    else if (c_z)   anchor = eprod;
    else            anchor = (eprod < uc.exp) ? eprod : uc.exp;

    __int128 big_p = p_z ? (__int128)0 : (sigprod << (eprod - anchor));
    __int128 big_c = c_z ? (__int128)0 : (uc.sig  << (uc.exp - anchor));

    int  result_sign;
    __int128 mag;
    if (sp == uc.sign) {
        mag         = big_p + big_c;
        result_sign = sp;
    } else {
        if (big_p >= big_c) {
            mag         = big_p - big_c;
            result_sign = sp;
        } else {
            mag         = big_c - big_p;
            result_sign = uc.sign;
        }
    }

    if (mag == 0) return 0x0000;  // exact zero, RNE → +0

    // Find MSB position (0..127).
    int msb = 0;
    for (int i = 127; i >= 0; i--) {
        if ((mag >> i) & 1) { msb = i; break; }
    }

    // Unbiased exponent of result = anchor + msb. fp16 bias = 15.
    int biased = anchor + msb + 15;

    if (biased >= 31) return pack_inf(result_sign);

    // Place leading-1 at bit 10 → shift right by (msb - 10).
    // For subnormal output (biased <= 0) we shift by additional (1 - biased).
    int shift = msb - 10;
    int eff_biased;
    if (biased <= 0) {
        shift += (1 - biased);
        eff_biased = 0;
    } else {
        eff_biased = biased;
    }

    uint64_t mant10;
    int guard, round_bit, sticky;
    if (shift <= 0) {
        // can't really happen (msb < 10 means biased very negative which
        // we'd already have flushed below). Treat as direct.
        mant10    = (uint64_t)(mag << (-shift)) & 0x3FF;
        guard     = 0;
        round_bit = 0;
        sticky    = 0;
    } else if (shift == 1) {
        mant10    = (uint64_t)(mag >> 1) & 0x3FF;
        guard     = (int)(mag & 1);
        round_bit = 0;
        sticky    = 0;
    } else if (shift == 2) {
        mant10    = (uint64_t)(mag >> 2) & 0x3FF;
        guard     = (int)((mag >> 1) & 1);
        round_bit = (int)(mag & 1);
        sticky    = 0;
    } else {
        mant10    = (uint64_t)(mag >> shift) & 0x3FF;
        guard     = (int)((mag >> (shift - 1)) & 1);
        round_bit = (int)((mag >> (shift - 2)) & 1);
        // sticky = OR of bits below bit (shift-2)
        __int128 below_mask = ((__int128)1 << (shift - 2)) - 1;
        sticky = ((mag & below_mask) != 0) ? 1 : 0;
    }

    int lsb      = (int)(mant10 & 1);
    int round_up = (guard && ((round_bit | sticky) || lsb)) ? 1 : 0;
    uint32_t mant_r = (uint32_t)mant10 + round_up;
    if (mant_r & 0x400) {
        eff_biased += 1;
        mant_r = 0;
    }

    if (eff_biased >= 31) return pack_inf(result_sign);
    if (eff_biased <= 0) {
        // subnormal output (or flush to zero if mant_r==0)
        return (uint16_t)((result_sign << 15) | (mant_r & 0x3FF));
    }
    return (uint16_t)((result_sign << 15) | ((eff_biased & 0x1F) << 10) |
                      (mant_r & 0x3FF));
}

// ─── driver ───────────────────────────────────────────────────
// Pipeline latency of fp16_fma (== fp16_lat_pkg::FP16_FMA_LAT). apply()
// holds the inputs constant for LAT ticks so the pipeline is fully drained
// and y_o reflects the triple just applied — letting every directed /
// shadow check below read the result inline. Overlapped streaming (one
// triple per tick) is exercised separately in test 5b.
static constexpr int LAT = 5;

static void apply(SimCtrl<DUT>& s, uint16_t a, uint16_t b, uint16_t c) {
    s.dut->a_i = a;
    s.dut->b_i = b;
    s.dut->c_i = c;
    for (int i = 0; i < LAT; i++) s.tick();
}

static uint16_t y_dut(SimCtrl<DUT>& s) { return (uint16_t)s.dut->y_dut_o; }
static uint16_t y_ref(SimCtrl<DUT>& s) { return (uint16_t)s.dut->y_ref_o; }

static std::string hex16(uint16_t v) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "0x%04X", v);
    return std::string(buf);
}

// Treat two fp16 NaN encodings as equivalent (any qNaN ≡ any other qNaN
// for the purposes of pass/fail). Our DUT and REF always emit 0x7E00 so
// this only really matters when comparing against the shadow.
static bool fp16_equal(uint16_t x, uint16_t y) {
    auto is_nan = [](uint16_t v) {
        return ((v >> 10) & 0x1F) == 0x1F && (v & 0x3FF) != 0;
    };
    if (is_nan(x) && is_nan(y)) return true;
    return x == y;
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 1200000000ull;

    sim.dut->a_i = 0;
    sim.dut->b_i = 0;
    sim.dut->c_i = 0;
    sim.reset();

    // ─── Test 1: reset ──────────────────────────────────
    printf("test 1: y_o == 0 after reset\n");
    sim.check(y_dut(sim) == 0x0000, "dut y == 0 after reset");
    sim.check(y_ref(sim) == 0x0000, "ref y == 0 after reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after reset");

    // ─── Test 2: directed cases ─────────────────────────
    printf("test 2: directed cases\n");
    struct Case { uint16_t a, b, c, exp; const char* desc; };
    const Case cases[] = {
        { 0x3C00, 0x3C00, 0x0000, 0x3C00, "1*1+0 = 1" },
        { 0x3C00, 0x3C00, 0x3C00, 0x4000, "1*1+1 = 2" },
        { 0x3800, 0x3800, 0x3400, 0x3800, "0.5*0.5 + 0.25 = 0.5" },
        { 0xBC00, 0x3C00, 0x3C00, 0x0000, "-1*1 + 1 = +0" },
        { 0x3C00, 0x7BFF, 0xFBFF, 0x0000, "1*65504 + (-65504) = +0" },
        { 0x0001, 0x3C00, 0x0000, 0x0001, "min subnormal * 1 + 0 = min subnormal" },
        { 0x7BFF, 0x4000, 0x0000, 0x7C00, "65504 * 2 + 0 = +Inf" },
        { 0x7E00, 0x3C00, 0x3C00, 0x7E00, "NaN * 1 + 1 = NaN" },
        { 0x7C00, 0x0000, 0x0000, 0x7E00, "Inf * 0 + 0 = NaN" },
        { 0x7C00, 0x3C00, 0xFC00, 0x7E00, "Inf - Inf = NaN" },
        { 0x0000, 0x0000, 0x0000, 0x0000, "0 + 0 = +0" },
        { 0x8000, 0x0000, 0x0000, 0x0000, "-0 * 0 + 0 = +0" },
        { 0xBC00, 0x3C00, 0x0000, 0xBC00, "-1 * 1 + 0 = -1" },
        { 0x3C00, 0x3C00, 0xBC00, 0x0000, "1*1 + (-1) = +0" },
    };
    for (auto& cs : cases) {
        apply(sim, cs.a, cs.b, cs.c);
        std::string d = std::string(cs.desc) + " got=" + hex16(y_dut(sim)) +
                        " exp=" + hex16(cs.exp);
        sim.check(fp16_equal(y_dut(sim), cs.exp), "dut " + d);
        sim.check(fp16_equal(y_ref(sim), cs.exp), "ref " + d);
        sim.check(sim.dut->mismatch_o == 0, "no mismatch: " + std::string(cs.desc));
        uint16_t sh = fp16_fma_shadow(cs.a, cs.b, cs.c);
        sim.check(fp16_equal(sh, cs.exp), "shadow " + std::string(cs.desc) +
                                          " sh=" + hex16(sh));
    }

    // ─── Test 3: signed-zero corners ────────────────────
    printf("test 3: signed-zero corners\n");
    struct ZCase { uint16_t a, b, c; const char* desc; };
    const ZCase zcases[] = {
        { 0x8000, 0x3C00, 0x0000, "-0 * 1 + 0" },     // a*b = -0, c = +0 → +0 RNE
        { 0x8000, 0x3C00, 0x8000, "-0 * 1 + (-0)" },  // -0 + -0 = -0
        { 0x3C00, 0x8000, 0x0000, "1 * -0 + 0" },
        { 0x0000, 0xBC00, 0x0000, "0 * -1 + 0" },
    };
    for (auto& z : zcases) {
        apply(sim, z.a, z.b, z.c);
        uint16_t sh = fp16_fma_shadow(z.a, z.b, z.c);
        sim.check(fp16_equal(y_dut(sim), sh),
                  std::string("zsign ") + z.desc +
                  " dut=" + hex16(y_dut(sim)) + " sh=" + hex16(sh));
        sim.check(sim.dut->mismatch_o == 0,
                  std::string("zsign no mismatch: ") + z.desc);
    }

    // ─── Test 4: RNE tie-break corners ───────────────────
    printf("test 4: RNE tie-break corners\n");
    // 1 + 2^-11 = 1.0 + 0.5*ulp(1) → tie → round to even (1.0).
    // 1 + 3*2^-11 = 1.5*ulp tie → round to nearest even at next bit.
    {
        // 2^-11 = 0x1400 (bexp=4, frac=0) — actually fp16 2^-11:
        // unbiased exp -11 → biased 4 → 0x1000. Let me just generate via
        // a couple of constructed values.
        struct R { uint16_t a, b, c; const char* desc; };
        R rs[] = {
            { 0x3C00, 0x3C00, 0x1000, "1 + 2^-11 (tie → even = 1.0)" },
            // 2^-10 (smallest representable ulp at 1.0) = 0x1400
            { 0x3C00, 0x3C00, 0x1400, "1 + 2^-10 = 0x3C01 (no tie)" },
        };
        for (auto& r : rs) {
            apply(sim, r.a, r.b, r.c);
            uint16_t sh = fp16_fma_shadow(r.a, r.b, r.c);
            sim.check(fp16_equal(y_dut(sim), sh),
                      std::string("rne ") + r.desc +
                      " dut=" + hex16(y_dut(sim)) + " sh=" + hex16(sh));
            sim.check(sim.dut->mismatch_o == 0, std::string("rne nomis: ") + r.desc);
        }
    }

    // ─── Test 5: randomized stress ──────────────────────
    printf("test 5: 100000 random fp16 triples\n");
    std::mt19937 rng(0xFA1A4F00u);
    std::uniform_int_distribution<uint32_t> d16(0, 0xFFFF);

    int n_mismatch = 0;
    int n_shadow_fail = 0;
    int first_print = 0;
    const int N = 100000;
    for (int i = 0; i < N; i++) {
        uint16_t a = (uint16_t)d16(rng);
        uint16_t b = (uint16_t)d16(rng);
        uint16_t c = (uint16_t)d16(rng);
        apply(sim, a, b, c);
        uint16_t sh = fp16_fma_shadow(a, b, c);
        uint16_t yd = y_dut(sim);
        uint16_t yr = y_ref(sim);
        if (sim.dut->mismatch_o) {
            if (first_print < 10) {
                printf("  mismatch @i=%d a=%s b=%s c=%s dut=%s ref=%s sh=%s\n",
                       i, hex16(a).c_str(), hex16(b).c_str(), hex16(c).c_str(),
                       hex16(yd).c_str(), hex16(yr).c_str(), hex16(sh).c_str());
                first_print++;
            }
            n_mismatch++;
        }
        if (!fp16_equal(yd, sh)) {
            if (first_print < 10) {
                printf("  shadow-mismatch @i=%d a=%s b=%s c=%s dut=%s sh=%s\n",
                       i, hex16(a).c_str(), hex16(b).c_str(), hex16(c).c_str(),
                       hex16(yd).c_str(), hex16(sh).c_str());
                first_print++;
            }
            n_shadow_fail++;
        }
    }
    sim.check(n_mismatch == 0,
              "rand: 0 dut-vs-ref mismatches (was " +
              std::to_string(n_mismatch) + ")");
    sim.check(n_shadow_fail == 0,
              "rand: 0 dut-vs-shadow mismatches (was " +
              std::to_string(n_shadow_fail) + ")");

    // ─── Test 5b: overlapped streaming (full throughput) ─
    // Feed one fresh triple per tick so distinct values occupy distinct
    // pipeline stages simultaneously, and verify DUT == REF every cycle.
    // This is the real test of the pipelined behaviour (apply() above
    // flushes between triples and so never overlaps stages).
    printf("test 5b: overlapped streaming, DUT==REF every cycle\n");
    {
        std::mt19937 srng(0x0FF1CE17u);
        std::uniform_int_distribution<uint32_t> sd16(0, 0xFFFF);
        int n_stream_mismatch = 0;
        const int NS = 40000;
        for (int i = 0; i < NS; i++) {
            sim.dut->a_i = (uint16_t)sd16(srng);
            sim.dut->b_i = (uint16_t)sd16(srng);
            sim.dut->c_i = (uint16_t)sd16(srng);
            sim.tick();  // single tick → values pipeline through, stages overlap
            if (sim.dut->mismatch_o) {
                if (n_stream_mismatch < 5) {
                    printf("  stream mismatch @i=%d dut=%s ref=%s\n",
                           i, hex16(y_dut(sim)).c_str(), hex16(y_ref(sim)).c_str());
                }
                n_stream_mismatch++;
            }
        }
        sim.check(n_stream_mismatch == 0,
                  "stream: 0 dut-vs-ref mismatches (was " +
                  std::to_string(n_stream_mismatch) + ")");
    }

    // ─── Test 6: mid-stream reset ───────────────────────
    printf("test 6: mid-stream reset\n");
    apply(sim, 0x3C00, 0x3C00, 0x3C00);
    sim.check(y_dut(sim) != 0, "y non-zero before reset");
    sim.dut->a_i = 0; sim.dut->b_i = 0; sim.dut->c_i = 0;
    sim.reset();
    sim.check(y_dut(sim) == 0, "dut y==0 after re-reset");
    sim.check(y_ref(sim) == 0, "ref y==0 after re-reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after re-reset");

    return sim.finish();
}
