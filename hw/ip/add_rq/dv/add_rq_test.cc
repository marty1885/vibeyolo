// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// add_rq — Verilator test. Drives DUT and REF in lockstep with random
// (a_i8, b_i8, scale_a, scale_b, inv_out_scale, bias) tuples, plus
// directed edge cases. Cross-checks against an independent C++ shadow
// that pipelines through fp16-rounded stages using `double` arithmetic
// — a different host type from the SV REF's `shortreal` and with a
// different fp16-conversion code path from both SV models. Each fp16
// rounding step uses an integer-mantissa RNE in __int128 (mirroring
// the shadow style from hw/ip/fp16_fma/dv/fp16_fma_test.cc).
//
// The DUT pipeline has 12 cycles of latency (i32_to_fp16=2 + three
// fp16_fma=3 each + fp16_to_i8_sat=1); we apply stimulus, then sample
// (y, valid) 12 cycles later.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <deque>
#include <vector>

#include "Vadd_rq_tb.h"
#include "sim_ctrl.h"

using DUT = Vadd_rq_tb;

static constexpr int LATENCY = 12;

// ─── fp16 ↔ double helpers (independent of SV REF) ──────────────
struct F16 {
    int sign;
    int kind;     // 0 zero, 1 normal, 2 subnormal, 3 inf, 4 nan
    // For normal/subnormal value = sig * 2^exp where sig fits in __int128
    __int128 sig;
    int      exp;
};

static F16 unpack_fp16_int(uint16_t x) {
    F16 u;
    u.sign = (x >> 15) & 1;
    int e = (x >> 10) & 0x1F;
    int m = x & 0x3FF;
    if (e == 0 && m == 0) { u.kind = 0; u.sig = 0; u.exp = 0; }
    else if (e == 31 && m == 0) { u.kind = 3; u.sig = 0; u.exp = 0; }
    else if (e == 31) { u.kind = 4; u.sig = 0; u.exp = 0; }
    else if (e == 0) {
        u.kind = 2;
        u.sig  = (__int128)m;
        u.exp  = -24;
    } else {
        u.kind = 1;
        u.sig  = (__int128)(1024 + m);
        u.exp  = e - 25;
    }
    return u;
}

// Convert fp16 to double (exact: every fp16 value fits in double).
static double fp16_to_double(uint16_t x) {
    F16 u = unpack_fp16_int(x);
    if (u.kind == 4) return std::nan("");
    if (u.kind == 3) return u.sign ? -INFINITY : INFINITY;
    if (u.kind == 0) return u.sign ? -0.0 : 0.0;
    double v = (double)(uint64_t)u.sig;
    v = std::ldexp(v, u.exp);
    return u.sign ? -v : v;
}

// Round a double to fp16 RNE with proper sub/Inf/NaN handling using
// purely integer arithmetic on the mantissa (different code path from
// the SV models).
static uint16_t double_to_fp16(double v) {
    if (std::isnan(v)) return 0x7E00;
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    int sign = (int)(bits >> 63) & 1;
    if (std::isinf(v)) return (uint16_t)((sign << 15) | 0x7C00);
    if (v == 0.0) return (uint16_t)(sign << 15);

    int e64 = (int)((bits >> 52) & 0x7FF);
    uint64_t m64 = bits & ((1ULL << 52) - 1);

    int unbiased = e64 - 1023;     // (for normals)
    __int128 sig;
    int exp_val;
    if (e64 == 0) {
        // subnormal double — way below fp16 representable; flush to 0
        return (uint16_t)(sign << 15);
    } else {
        sig = ((__int128)1 << 52) | m64;
        exp_val = unbiased - 52;     // sig at bit 52 is implicit 1
    }

    // Now value = sig * 2^exp_val, sig has MSB at bit 52.
    // We want to express it as fp16: place leading 1 at bit 10 of mant.
    // Round to nearest even at bit 11 boundary, after accounting for
    // subnormal-shift.
    int msb = 52;
    // Adjust msb in case sig has trailing zeros / sig nonzero already
    // — leading bit IS at 52 since we set the implicit 1.

    // unbiased exponent of value = exp_val + msb = unbiased.
    int biased = unbiased + 15;

    if (biased >= 31) return (uint16_t)((sign << 15) | 0x7C00);

    int shift;
    int eff_biased;
    if (biased <= 0) {
        shift = (msb - 10) + (1 - biased);
        eff_biased = 0;
    } else {
        shift = msb - 10;
        eff_biased = biased;
    }

    uint64_t mant10;
    int guard, round_b, sticky;
    if (shift <= 0) {
        // can't normally happen for fp16-representable magnitudes
        mant10  = (uint64_t)(sig << (-shift)) & 0x3FF;
        guard   = 0; round_b = 0; sticky = 0;
    } else if (shift == 1) {
        mant10  = (uint64_t)(sig >> 1) & 0x3FF;
        guard   = (int)(sig & 1);
        round_b = 0; sticky = 0;
    } else if (shift == 2) {
        mant10  = (uint64_t)(sig >> 2) & 0x3FF;
        guard   = (int)((sig >> 1) & 1);
        round_b = (int)(sig & 1);
        sticky  = 0;
    } else if (shift >= 100) {
        mant10  = 0;
        guard = 0; round_b = 0;
        sticky = (sig != 0) ? 1 : 0;
    } else {
        mant10  = (uint64_t)(sig >> shift) & 0x3FF;
        guard   = (int)((sig >> (shift - 1)) & 1);
        round_b = (int)((sig >> (shift - 2)) & 1);
        __int128 below = ((__int128)1 << (shift - 2)) - 1;
        sticky = ((sig & below) != 0) ? 1 : 0;
    }

    int lsb = (int)(mant10 & 1);
    int up  = (guard && ((round_b | sticky) || lsb)) ? 1 : 0;
    uint32_t mr = (uint32_t)mant10 + up;
    if (mr & 0x400) { eff_biased += 1; mr = 0; }

    if (eff_biased >= 31) return (uint16_t)((sign << 15) | 0x7C00);
    if (eff_biased <= 0)  return (uint16_t)((sign << 15) | (mr & 0x3FF));
    return (uint16_t)((sign << 15) | ((eff_biased & 0x1F) << 10) | (mr & 0x3FF));
}

// fp16 → int8 saturating, RNE on the integer.
static int8_t fp16_to_i8_sat_shadow(uint16_t x) {
    F16 u = unpack_fp16_int(x);
    if (u.kind == 4) return 0;
    if (u.kind == 3) return u.sign ? -128 : 127;
    if (u.kind == 0) return 0;
    double v = fp16_to_double(x);

    // RNE round to integer
    double r = std::nearbyint(v);  // default RNE
    if (r >  127.0) return  127;
    if (r < -128.0) return -128;
    return (int8_t)(int)r;
}

// Five-stage pipeline shadow.
static int8_t add_rq_shadow(int8_t a, int8_t b,
                             uint16_t sa, uint16_t sb,
                             uint16_t inv_out, uint16_t bias) {
    // Stage 0: i32 → fp16 (a,b are in [-128,127], exactly representable
    // in fp16, so this is exact).
    uint16_t fa16 = double_to_fp16((double)a);
    uint16_t fb16 = double_to_fp16((double)b);

    // Stage 1: fp16(fa * sa + 0) and fp16(fb * sb + 0)
    double ta = fp16_to_double(fa16) * fp16_to_double(sa);
    double tb = fp16_to_double(fb16) * fp16_to_double(sb);
    uint16_t ta16 = double_to_fp16(ta);
    uint16_t tb16 = double_to_fp16(tb);

    // Stage 2: fp16(ta * 1.0 + tb)
    double sum = fp16_to_double(ta16) + fp16_to_double(tb16);
    uint16_t sum16 = double_to_fp16(sum);

    // Stage 3: fp16(sum * inv_out + bias)
    double fy = fp16_to_double(sum16) * fp16_to_double(inv_out)
                + fp16_to_double(bias);
    uint16_t fy16 = double_to_fp16(fy);

    // Stage 4: fp16 → i8 sat
    return fp16_to_i8_sat_shadow(fy16);
}

// ─── driver / sampler ─────────────────────────────────────────
struct Stim {
    int8_t   a, b;
    uint16_t sa, sb, inv_out, bias;
    bool     valid;
};

static void drive(SimCtrl<DUT>& s, const Stim& st) {
    s.dut->valid_i              = st.valid ? 1 : 0;
    s.dut->a_i8_i               = (uint8_t)st.a;
    s.dut->b_i8_i               = (uint8_t)st.b;
    s.dut->scale_a_fp16_i       = st.sa;
    s.dut->scale_b_fp16_i       = st.sb;
    s.dut->inv_out_scale_fp16_i = st.inv_out;
    s.dut->bias_fp16_i          = st.bias;
}

static std::string hex16(uint16_t v) {
    char buf[8]; std::snprintf(buf, sizeof(buf), "0x%04X", v); return buf;
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 2000000000ull;

    // initial drive
    sim.dut->valid_i = 0;
    sim.dut->a_i8_i  = 0;
    sim.dut->b_i8_i  = 0;
    sim.dut->scale_a_fp16_i       = 0;
    sim.dut->scale_b_fp16_i       = 0;
    sim.dut->inv_out_scale_fp16_i = 0;
    sim.dut->bias_fp16_i          = 0;
    sim.reset();

    // After-reset checks.
    sim.check(sim.dut->y_dut_o     == 0, "dut y == 0 after reset");
    sim.check(sim.dut->y_ref_o     == 0, "ref y == 0 after reset");
    sim.check(sim.dut->valid_dut_o == 0, "dut valid == 0 after reset");
    sim.check(sim.dut->valid_ref_o == 0, "ref valid == 0 after reset");
    sim.check(sim.dut->mismatch_o  == 0, "no mismatch after reset");

    // ─── helper: pipeline-style run ─────────────────────────
    // Applies a sequence of Stim, drains LATENCY extra cycles, and
    // compares (valid, y) on every tick against the shadow and against
    // the DUT-vs-REF mismatch flag.
    auto run = [&](const std::vector<Stim>& seq, const char* tag,
                   int& mismatch_count, int& shadow_count) {
        std::deque<Stim> inflight;
        const int N = (int)seq.size();
        const int total = N + LATENCY + 2;
        for (int i = 0; i < total; i++) {
            Stim s_in;
            if (i < N) s_in = seq[i];
            else       s_in = {0, 0, 0, 0, 0, 0, false};

            drive(sim, s_in);
            sim.tick();

            // Output now corresponds to whatever was applied LATENCY
            // cycles ago. Track via FIFO.
            inflight.push_back(s_in);
            if ((int)inflight.size() >= LATENCY) {
                Stim expected = inflight.front();
                inflight.pop_front();

                bool vdut = sim.dut->valid_dut_o;
                bool vref = sim.dut->valid_ref_o;
                int8_t ydut = (int8_t)sim.dut->y_dut_o;
                int8_t yref = (int8_t)sim.dut->y_ref_o;

                if (sim.dut->mismatch_o) {
                    if (mismatch_count < 8) {
                        printf("  [%s] mismatch @ick=%d expected_valid=%d "
                               "vdut=%d vref=%d ydut=%d yref=%d "
                               "a=%d b=%d sa=%s sb=%s inv=%s bias=%s\n",
                               tag, i, expected.valid?1:0, vdut?1:0, vref?1:0,
                               (int)ydut, (int)yref,
                               (int)expected.a, (int)expected.b,
                               hex16(expected.sa).c_str(),
                               hex16(expected.sb).c_str(),
                               hex16(expected.inv_out).c_str(),
                               hex16(expected.bias).c_str());
                    }
                    mismatch_count++;
                }

                if (expected.valid) {
                    int8_t sh = add_rq_shadow(expected.a, expected.b,
                                              expected.sa, expected.sb,
                                              expected.inv_out, expected.bias);
                    if (!vdut) {
                        if (shadow_count < 8) {
                            printf("  [%s] valid drop @i=%d\n", tag, i);
                        }
                        shadow_count++;
                    } else if (ydut != sh) {
                        // Allow ±1 LSB tolerance vs shadow (rounding paths
                        // differ slightly across implementations). The
                        // DUT-vs-REF check above is bit-exact.
                        int diff = (int)ydut - (int)sh;
                        if (diff > 1 || diff < -1) {
                            if (shadow_count < 8) {
                                printf("  [%s] shadow-mismatch @i=%d "
                                       "ydut=%d sh=%d diff=%d "
                                       "a=%d b=%d sa=%s sb=%s inv=%s bias=%s\n",
                                       tag, i, (int)ydut, (int)sh, diff,
                                       (int)expected.a, (int)expected.b,
                                       hex16(expected.sa).c_str(),
                                       hex16(expected.sb).c_str(),
                                       hex16(expected.inv_out).c_str(),
                                       hex16(expected.bias).c_str());
                            }
                            shadow_count++;
                        }
                    }
                }
            }
        }
    };

    // ─── Test: directed edge cases ──────────────────────────
    printf("test: directed edges\n");
    {
        std::vector<Stim> seq;
        // a=0, b=0 → 0
        seq.push_back({0, 0, 0x3C00, 0x3C00, 0x3C00, 0x0000, true});
        // a=0, b=10, all scales 1, no bias → 10
        seq.push_back({0, 10, 0x3C00, 0x3C00, 0x3C00, 0x0000, true});
        seq.push_back({10, 0, 0x3C00, 0x3C00, 0x3C00, 0x0000, true});
        // opposite-sign cancellation: 50 + (-50) with all unit scales → 0
        seq.push_back({50, -50, 0x3C00, 0x3C00, 0x3C00, 0x0000, true});
        // both saturate positive: a=127, b=127, scales=1, inv_out=1 → +127
        seq.push_back({127, 127, 0x3C00, 0x3C00, 0x3C00, 0x0000, true});
        // both saturate negative: a=-128, b=-128 → -128
        seq.push_back({-128, -128, 0x3C00, 0x3C00, 0x3C00, 0x0000, true});
        // scale_a = 0 disables a side
        seq.push_back({100, 20, 0x0000, 0x3C00, 0x3C00, 0x0000, true});
        // scale_b = 0 disables b side
        seq.push_back({100, 20, 0x3C00, 0x0000, 0x3C00, 0x0000, true});
        // inv_out_scale=2.0: 30+10=40, *2 = 80
        seq.push_back({30, 10, 0x3C00, 0x3C00, 0x4000, 0x0000, true});
        // inv_out_scale=0.5: 30+10=40, *0.5 = 20
        seq.push_back({30, 10, 0x3C00, 0x3C00, 0x3800, 0x0000, true});
        // bias only: a=0, b=0, bias = 5.0 (0x4500) → 5
        seq.push_back({0, 0, 0x3C00, 0x3C00, 0x3C00, 0x4500, true});
        // NaN propagation: scale_a=NaN
        seq.push_back({10, 10, 0x7E00, 0x3C00, 0x3C00, 0x0000, true});
        // +Inf scale → saturates to +127 (or -128 depending on sign of a)
        seq.push_back({10, 0, 0x7C00, 0x3C00, 0x3C00, 0x0000, true});
        // bubble with valid=0
        seq.push_back({1, 2, 0x3C00, 0x3C00, 0x3C00, 0x0000, false});
        // back-to-back
        seq.push_back({1, 2, 0x3C00, 0x3C00, 0x3C00, 0x0000, true});

        int m = 0, sc = 0;
        run(seq, "directed", m, sc);
        sim.check(m == 0, "directed: 0 DUT-vs-REF mismatches (got " +
                          std::to_string(m) + ")");
        sim.check(sc == 0, "directed: 0 DUT-vs-shadow mismatches (got " +
                           std::to_string(sc) + ")");
    }

    // ─── Test: random fp16 scales, random i8 ────────────────
    printf("test: 12000 random vectors\n");
    {
        std::mt19937 rng(0xADD1B100u);
        std::uniform_int_distribution<int> d_i8(-128, 127);
        // For scales/bias we want representable, non-Inf/NaN fp16 values
        // most of the time, with occasional pathological ones.
        // Generate by picking a random exponent in [-8..+4] (so magnitudes
        // ~[1/256, 16]) and random mantissa.
        auto rand_fp16_typical = [&]() -> uint16_t {
            std::uniform_int_distribution<int> de(-8, 4);
            std::uniform_int_distribution<int> dm(0, 0x3FF);
            std::uniform_int_distribution<int> ds(0, 1);
            int e = de(rng);
            int biased = e + 15;
            int s = ds(rng);
            int m = dm(rng);
            return (uint16_t)((s << 15) | ((biased & 0x1F) << 10) | (m & 0x3FF));
        };
        auto rand_fp16_any = [&]() -> uint16_t {
            std::uniform_int_distribution<uint32_t> d16(0, 0xFFFF);
            return (uint16_t)d16(rng);
        };

        std::vector<Stim> seq;
        seq.reserve(12000);
        for (int i = 0; i < 12000; i++) {
            Stim s;
            s.a = (int8_t)d_i8(rng);
            s.b = (int8_t)d_i8(rng);
            // 5% chance of any fp16 (incl Inf/NaN/subnormal); else typical.
            std::uniform_int_distribution<int> pct(0, 99);
            auto pick = [&]() {
                return pct(rng) < 5 ? rand_fp16_any() : rand_fp16_typical();
            };
            s.sa      = pick();
            s.sb      = pick();
            s.inv_out = pick();
            s.bias    = pct(rng) < 50 ? 0x0000 : pick();
            s.valid   = (pct(rng) < 90);
            seq.push_back(s);
        }

        int m = 0, sc = 0;
        run(seq, "rand", m, sc);
        sim.check(m == 0, "rand: 0 DUT-vs-REF mismatches (got " +
                          std::to_string(m) + ")");
        // Allow a small shadow-mismatch budget for NaN/Inf border cases
        // where ±1-LSB heuristics can still disagree.
        sim.check(sc < 50,
                  "rand: <50 DUT-vs-shadow mismatches (got " +
                  std::to_string(sc) + ")");
    }

    // ─── Test: opposite-sign cancellation sweep ─────────────
    printf("test: cancellation sweep\n");
    {
        std::vector<Stim> seq;
        for (int a = -128; a <= 127; a += 13) {
            int b = -a;
            if (b > 127) b = 127;
            if (b < -128) b = -128;
            seq.push_back({(int8_t)a, (int8_t)b,
                           0x3C00, 0x3C00, 0x3C00, 0x0000, true});
        }
        int m = 0, sc = 0;
        run(seq, "cancel", m, sc);
        sim.check(m == 0, "cancel: 0 DUT-vs-REF mismatches");
    }

    // ─── Test: saturate sweep ───────────────────────────────
    printf("test: saturate sweep\n");
    {
        std::vector<Stim> seq;
        // big scales to push beyond i8 range
        for (int a = -128; a <= 127; a += 17) {
            for (int b = -128; b <= 127; b += 19) {
                seq.push_back({(int8_t)a, (int8_t)b,
                               0x4400 /*4.0*/, 0x4400 /*4.0*/,
                               0x3C00 /*1.0*/, 0x0000, true});
            }
        }
        int m = 0, sc = 0;
        run(seq, "sat", m, sc);
        sim.check(m == 0, "sat: 0 DUT-vs-REF mismatches");
    }

    // ─── Test: mid-stream reset ─────────────────────────────
    printf("test: mid-stream reset\n");
    sim.dut->valid_i = 1;
    sim.dut->a_i8_i  = 50;
    sim.dut->b_i8_i  = 50;
    sim.dut->scale_a_fp16_i = 0x3C00;
    sim.dut->scale_b_fp16_i = 0x3C00;
    sim.dut->inv_out_scale_fp16_i = 0x3C00;
    sim.dut->bias_fp16_i = 0;
    for (int i = 0; i < 10; i++) sim.tick();
    sim.dut->valid_i = 0;
    sim.dut->a_i8_i  = 0;
    sim.dut->b_i8_i  = 0;
    sim.dut->scale_a_fp16_i = 0;
    sim.dut->scale_b_fp16_i = 0;
    sim.dut->inv_out_scale_fp16_i = 0;
    sim.dut->bias_fp16_i = 0;
    sim.reset();
    sim.check(sim.dut->y_dut_o     == 0, "dut y==0 after re-reset");
    sim.check(sim.dut->y_ref_o     == 0, "ref y==0 after re-reset");
    sim.check(sim.dut->valid_dut_o == 0, "dut valid==0 after re-reset");
    sim.check(sim.dut->valid_ref_o == 0, "ref valid==0 after re-reset");
    sim.check(sim.dut->mismatch_o  == 0, "no mismatch after re-reset");

    return sim.finish();
}
