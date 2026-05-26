// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// fp16_macw — Verilator test. Drives the wide-accumulate MAC (a,b fp16;
// c,y a 30-bit float: 8 exp, 21 mantissa) and cross-checks against an
// independent long-double reference: y = round_wide(a*b + c). Also runs a
// long streaming reduction (y fed back as c) and checks the wide result is
// dramatically closer to the exact dot product than fp16 accumulation —
// which is the whole reason this block exists.

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "Vfp16_macw_tb.h"
#include "sim_ctrl.h"

using DUT = Vfp16_macw_tb;

static constexpr int LAT      = 5;
static constexpr int ACC_EXP  = 8;
static constexpr int ACC_MANT = 21;
static constexpr int ACC_BIAS = (1 << (ACC_EXP - 1)) - 1;   // 127
static constexpr uint32_t ACC_W = 1 + ACC_EXP + ACC_MANT;   // 30

// ─────────────────────── fp16 helpers ───────────────────────
static long double fp16_to_ld(uint16_t x) {
    int s = (x >> 15) & 1, e = (x >> 10) & 31, f = x & 1023;
    long double v;
    if (e == 31) v = (f ? NAN : INFINITY);
    else if (e == 0) v = ldexpl((long double)f, -24);
    else v = ldexpl(1.0L + (long double)f / 1024.0L, e - 15);
    return s ? -v : v;
}
static uint16_t ld_to_fp16(long double v) {
    if (std::isnan(v)) return 0x7E00;
    int s = std::signbit(v) ? 1 : 0;
    long double av = std::fabsl(v);
    if (std::isinf(av) || av >= 65520.0L) return (uint16_t)((s << 15) | 0x7C00);
    if (av == 0.0L) return (uint16_t)(s << 15);
    int e; long double m = frexpl(av, &e);      // av = m*2^e, m in [0.5,1)
    e -= 1; m *= 2.0L;                            // av = m*2^e, m in [1,2)
    int biased = e + 15;
    if (biased >= 31) return (uint16_t)((s << 15) | 0x7C00);
    long double scaled; long long mant;
    if (biased <= 0) {
        scaled = av * ldexpl(1.0L, 24);
        mant = llroundl(scaled);
        if (mant >= 1024) return (uint16_t)((s << 15) | (1 << 10));
        return (uint16_t)((s << 15) | (mant & 0x3FF));
    }
    scaled = (m - 1.0L) * 1024.0L;
    mant = llroundl(scaled);
    if (mant >= 1024) { biased++; mant = 0; if (biased >= 31) return (uint16_t)((s << 15) | 0x7C00); }
    return (uint16_t)((s << 15) | (biased << 10) | (mant & 0x3FF));
}

// ─────────────────────── wide float helpers ─────────────────
static long double wide_to_ld(uint32_t x) {
    int s = (x >> (ACC_W - 1)) & 1;
    int e = (x >> ACC_MANT) & ((1 << ACC_EXP) - 1);
    uint32_t f = x & ((1u << ACC_MANT) - 1);
    long double v;
    if (e == (1 << ACC_EXP) - 1) v = (f ? NAN : INFINITY);
    else if (e == 0) v = ldexpl((long double)f, 1 - ACC_BIAS - ACC_MANT);
    else v = ldexpl(1.0L + (long double)f / (long double)(1u << ACC_MANT), e - ACC_BIAS);
    return s ? -v : v;
}
static uint32_t ld_to_wide(long double v) {
    if (std::isnan(v)) return (uint32_t)((( (1u<<ACC_EXP)-1) << ACC_MANT) | (1u << (ACC_MANT-1)));
    int s = std::signbit(v) ? 1 : 0;
    long double av = std::fabsl(v);
    uint32_t emax = (1u << ACC_EXP) - 1;
    if (std::isinf(av)) return (uint32_t)((s << (ACC_W-1)) | (emax << ACC_MANT));
    if (av == 0.0L) return (uint32_t)(s << (ACC_W - 1));
    int e; long double m = frexpl(av, &e); e -= 1; m *= 2.0L;   // m in [1,2)
    int biased = e + ACC_BIAS;
    long double scaled; long long mant;
    if (biased >= (int)emax) return (uint32_t)((s << (ACC_W-1)) | (emax << ACC_MANT)); // Inf
    if (biased <= 0) {
        scaled = av * ldexpl(1.0L, ACC_BIAS + ACC_MANT - 1);
        mant = llroundl(scaled);
        if (mant >= (1LL << ACC_MANT)) return (uint32_t)((s << (ACC_W-1)) | (1u << ACC_MANT));
        return (uint32_t)((s << (ACC_W-1)) | (uint32_t)(mant & ((1u<<ACC_MANT)-1)));
    }
    scaled = (m - 1.0L) * (long double)(1u << ACC_MANT);
    mant = llroundl(scaled);
    if (mant >= (1LL << ACC_MANT)) { biased++; mant = 0; if (biased >= (int)emax) return (uint32_t)((s << (ACC_W-1)) | (emax << ACC_MANT)); }
    return (uint32_t)((s << (ACC_W-1)) | ((uint32_t)biased << ACC_MANT) | (uint32_t)(mant & ((1u<<ACC_MANT)-1)));
}

// ordered key for wide-ULP distance
static int64_t wide_ordered(uint32_t a) {
    int64_t v = a & ((1u << (ACC_W - 1)) - 1);   // magnitude bits
    return (a >> (ACC_W - 1)) & 1 ? -v : v;
}
static int64_t wide_ulp(uint32_t a, uint32_t b) {
    int64_t d = wide_ordered(a) - wide_ordered(b);
    return d < 0 ? -d : d;
}

// ─────────────────────── driver ─────────────────────────────
static uint32_t step(SimCtrl<DUT>& s, uint16_t a, uint16_t b, uint32_t c) {
    s.dut->a_i = a; s.dut->b_i = b; s.dut->c_i = c;
    for (int i = 0; i < LAT; i++) s.tick();
    return (uint32_t)s.dut->y_o;
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 1200000000ull;
    sim.dut->a_i = 0; sim.dut->b_i = 0; sim.dut->c_i = 0;
    sim.reset();

    printf("test 1: reset → y==0\n");
    sim.check((uint32_t)sim.dut->y_o == 0, "y==0 after reset");

    // ─── Test 2: directed ───
    printf("test 2: directed cases\n");
    struct C { uint16_t a, b; long double c; const char* d; };
    const C cases[] = {
        { 0x3C00, 0x3C00, 0.0L,  "1*1+0 = 1" },
        { 0x3C00, 0x3C00, 1.0L,  "1*1+1 = 2" },
        { 0x3800, 0x3800, 0.25L, "0.5*0.5+0.25 = 0.5" },
        { 0xBC00, 0x3C00, 1.0L,  "-1*1+1 = 0" },
        { 0x3C00, 0x3C00, -1.0L, "1*1-1 = 0" },
    };
    for (auto& c : cases) {
        uint32_t cw = ld_to_wide(c.c);
        uint32_t got = step(sim, c.a, c.b, cw);
        long double ref = fp16_to_ld(c.a) * fp16_to_ld(c.b) + wide_to_ld(cw);
        uint32_t exp = ld_to_wide(ref);
        sim.check(wide_ulp(got, exp) == 0,
                  std::string(c.d) + " got=" + std::to_string(wide_to_ld(got)) +
                  " exp=" + std::to_string((double)ref));
    }

    // ─── Test 3: randomized single ops (bounded so long double is exact) ───
    printf("test 3: 200000 random single MACs\n");
    std::mt19937 rng(0xACC0u);
    std::uniform_real_distribution<double> da(-4.0, 4.0), dc(-64.0, 64.0);
    int64_t worst = 0; int errs = 0;
    for (int i = 0; i < 200000; i++) {
        uint16_t a = ld_to_fp16((long double)da(rng));
        uint16_t b = ld_to_fp16((long double)da(rng));
        uint32_t c = ld_to_wide((long double)dc(rng));
        uint32_t got = step(sim, a, b, c);
        long double ref = fp16_to_ld(a) * fp16_to_ld(b) + wide_to_ld(c);
        uint32_t exp = ld_to_wide(ref);
        int64_t u = wide_ulp(got, exp);
        if (u > worst) worst = u;
        if (u > 1) {                          // allow ≤1 ULP (deep-shift sticky)
            if (errs < 8)
                printf("  mismatch a=%04x b=%04x c=%08x got=%08x(%.6f) exp=%08x(%.6f) ulp=%lld\n",
                       a, b, c, got, (double)wide_to_ld(got), exp, (double)ref, (long long)u);
            errs++;
        }
    }
    printf("  worst single-op ulp=%lld errs=%d\n", (long long)worst, errs);
    sim.check(errs == 0, "all single MACs within 1 wide-ULP");

    // ─── Test 4: streaming reduction — accuracy vs fp16 accumulation ───
    // Sum of N fp16 products. Compare: wide-acc DUT vs fp16-acc vs exact.
    printf("test 4: streaming reductions (wide acc should beat fp16 acc)\n");
    std::uniform_real_distribution<double> dv(-1.0, 1.0);
    int wide_wins = 0; const int TRIALS = 200, N = 400;
    long double wide_err_sum = 0, f16_err_sum = 0;
    for (int t = 0; t < TRIALS; t++) {
        std::vector<uint16_t> A(N), B(N);
        for (int k = 0; k < N; k++) { A[k] = ld_to_fp16((long double)dv(rng)); B[k] = ld_to_fp16((long double)dv(rng)); }
        // exact (long double) dot product
        long double exact = 0;
        for (int k = 0; k < N; k++) exact += fp16_to_ld(A[k]) * fp16_to_ld(B[k]);
        // fp16 accumulation (what flash_attn used to do)
        uint16_t f16acc = 0x0000;
        for (int k = 0; k < N; k++)
            f16acc = ld_to_fp16(fp16_to_ld(A[k]) * fp16_to_ld(B[k]) + fp16_to_ld(f16acc));
        // wide accumulation through the DUT (c fed back)
        uint32_t wacc = 0;
        for (int k = 0; k < N; k++) wacc = step(sim, A[k], B[k], wacc);
        long double we = std::fabsl(wide_to_ld(wacc) - exact);
        long double fe = std::fabsl(fp16_to_ld(f16acc) - exact);
        wide_err_sum += we; f16_err_sum += fe;
        if (we <= fe) wide_wins++;
    }
    printf("  mean|err| wide=%.3e  fp16=%.3e  wide_wins=%d/%d\n",
           (double)(wide_err_sum / TRIALS), (double)(f16_err_sum / TRIALS), wide_wins, TRIALS);
    sim.check(wide_err_sum * 10.0L < f16_err_sum,
              "wide accumulation ≥10x more accurate than fp16 accumulation");

    return sim.finish();
}
