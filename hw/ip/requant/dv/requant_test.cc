// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// requant_test — drives DUT (composed) and REF (flat behavioral) with
// identical (acc, scale, bias) triples and cross-checks both against an
// independent C++ shadow that implements int32→fp16, fp16 FMA, and fp16→
// int8 saturation from scratch using __int128 integer arithmetic.
//
// The lesson from fp16_fma's test: fp16 FMA alignment can need ~76 bits;
// fp32 is not enough. We use __int128 everywhere the magnitude lives.
//
// Pipeline latency: 7 cycles — i32_to_fp16(2) + scale-bump(1) + fp16_fma(3)
// + fp16_to_i8_sat(1). We push a triple into the pipe, then sample valid_o
// + y_o LATENCY cycles later (queue-aligned below).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <random>
#include <string>

#include "Vrequant_tb.h"
#include "sim_ctrl.h"

using DUT = Vrequant_tb;

static constexpr int LATENCY = 7;

// ─── int32 → (fp16, shift) with auto-prescale ───────────────
struct CvtOut { uint16_t fp16; uint8_t shift; };

static CvtOut i32_to_fp16_shadow(int32_t x) {
    if (x == 0) return {0x0000, 0};
    int sign = (x < 0) ? 1 : 0;
    uint32_t mag = sign ? (uint32_t)(-(int64_t)x) : (uint32_t)x;
    int msb = 31;
    while (msb > 0 && !((mag >> msb) & 1u)) msb--;
    int pre_shift = (msb <= 15) ? 0 : (msb - 14);
    uint32_t mag_sh = mag >> pre_shift;
    uint32_t shifted_lo = (pre_shift > 0) ? (mag & ((1u << pre_shift) - 1u)) : 0u;
    int extra_sticky = shifted_lo ? 1 : 0;
    int e_post = msb - pre_shift;

    int biased = e_post + 15;
    uint32_t mant_pre;
    int guard = 0, sticky = extra_sticky;
    if (e_post <= 10) {
        mant_pre = mag_sh << (10 - e_post);
    } else {
        int sh = e_post - 10;
        mant_pre = mag_sh >> sh;
        guard    = (int)((mag_sh >> (sh - 1)) & 1u);
        uint32_t mask = (sh - 1 > 0) ? ((1u << (sh - 1)) - 1u) : 0u;
        if (mag_sh & mask) sticky = 1;
    }
    int lsb = (int)(mant_pre & 1u);
    int round_up = (guard && (sticky || lsb)) ? 1 : 0;
    uint32_t mant_r = mant_pre + (uint32_t)round_up;
    if (mant_r & 0x800u) {
        biased += 1;
        mant_r = 0;
    }
    uint16_t fp16;
    if (biased >= 31) fp16 = (uint16_t)((sign << 15) | 0x7C00);
    else              fp16 = (uint16_t)((sign << 15) | ((biased & 0x1F) << 10) | (mant_r & 0x3FFu));
    return { fp16, (uint8_t)pre_shift };
}

static uint16_t bump_scale_shadow(uint16_t scale, uint8_t shift) {
    uint16_t s_sign = scale & 0x8000;
    uint8_t  s_exp  = (scale >> 10) & 0x1F;
    uint16_t s_frac = scale & 0x3FF;
    if (s_exp == 0 || s_exp == 31) return scale;
    int e_sum = (int)s_exp + (int)shift;
    uint8_t e_new = (e_sum >= 30) ? 30 : (uint8_t)e_sum;
    return (uint16_t)(s_sign | ((uint16_t)e_new << 10) | s_frac);
}

// ─── fp16 FMA shadow (lifted from fp16_fma_test.cc, __int128 path) ──
struct Unpack {
    int sign;
    int kind;       // 0 zero, 1 normal, 2 subnormal, 3 inf, 4 nan
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
        u.kind = 2;
        u.sig  = (__int128)(uint64_t)frac << 13;
        u.exp  = -37;
    } else {
        u.kind = 1;
        u.sig  = (__int128)(uint64_t)(1024 + frac) << 13;
        u.exp  = bexp - 38;
    }
    return u;
}

static uint16_t pack_inf(int sign)  { return (uint16_t)((sign << 15) | 0x7C00); }
static uint16_t pack_nan()          { return 0x7E00; }

static uint16_t fp16_fma_shadow(uint16_t a, uint16_t b, uint16_t c) {
    Unpack ua = unpack16(a), ub = unpack16(b), uc = unpack16(c);

    if (ua.kind == 4 || ub.kind == 4 || uc.kind == 4) return pack_nan();
    bool prod_inf  = (ua.kind == 3 || ub.kind == 3);
    bool prod_zero = (ua.kind == 0 || ub.kind == 0);
    if (prod_inf && prod_zero) return pack_nan();
    int sp = ua.sign ^ ub.sign;
    if (prod_inf && uc.kind == 3 && sp != uc.sign) return pack_nan();
    if (prod_inf) return pack_inf(sp);
    if (uc.kind == 3) return pack_inf(uc.sign);

    __int128 sigprod = ua.sig * ub.sig;
    int      eprod   = ua.exp + ub.exp;

    bool p_z = prod_zero || (sigprod == 0);
    bool c_z = (uc.kind == 0);
    if (p_z && c_z) return 0x0000;

    int anchor;
    if (p_z)        anchor = uc.exp;
    else if (c_z)   anchor = eprod;
    else            anchor = (eprod < uc.exp) ? eprod : uc.exp;

    __int128 big_p = p_z ? (__int128)0 : (sigprod << (eprod - anchor));
    __int128 big_c = c_z ? (__int128)0 : (uc.sig  << (uc.exp - anchor));

    int result_sign;
    __int128 mag;
    if (sp == uc.sign) {
        mag = big_p + big_c;
        result_sign = sp;
    } else if (big_p >= big_c) {
        mag = big_p - big_c;
        result_sign = sp;
    } else {
        mag = big_c - big_p;
        result_sign = uc.sign;
    }
    if (mag == 0) return 0x0000;

    int msb = 0;
    for (int i = 127; i >= 0; i--) {
        if ((mag >> i) & 1) { msb = i; break; }
    }
    int biased = anchor + msb + 15;
    if (biased >= 31) return pack_inf(result_sign);

    int shift = msb - 10;
    int eff_biased;
    if (biased <= 0) { shift += (1 - biased); eff_biased = 0; }
    else             { eff_biased = biased; }

    uint64_t mant10;
    int guard, round_bit, sticky;
    if (shift <= 0) {
        mant10 = (uint64_t)(mag << (-shift)) & 0x3FF;
        guard = 0; round_bit = 0; sticky = 0;
    } else if (shift == 1) {
        mant10 = (uint64_t)(mag >> 1) & 0x3FF;
        guard = (int)(mag & 1); round_bit = 0; sticky = 0;
    } else if (shift == 2) {
        mant10 = (uint64_t)(mag >> 2) & 0x3FF;
        guard = (int)((mag >> 1) & 1); round_bit = (int)(mag & 1); sticky = 0;
    } else {
        mant10 = (uint64_t)(mag >> shift) & 0x3FF;
        guard  = (int)((mag >> (shift - 1)) & 1);
        round_bit = (int)((mag >> (shift - 2)) & 1);
        __int128 below_mask = ((__int128)1 << (shift - 2)) - 1;
        sticky = ((mag & below_mask) != 0) ? 1 : 0;
    }

    int lsb = (int)(mant10 & 1);
    int round_up = (guard && ((round_bit | sticky) || lsb)) ? 1 : 0;
    uint32_t mant_r = (uint32_t)mant10 + round_up;
    if (mant_r & 0x400) { eff_biased += 1; mant_r = 0; }

    if (eff_biased >= 31) return pack_inf(result_sign);
    if (eff_biased <= 0)  return (uint16_t)((result_sign << 15) | (mant_r & 0x3FF));
    return (uint16_t)((result_sign << 15) | ((eff_biased & 0x1F) << 10) | (mant_r & 0x3FF));
}

// ─── fp16 → int8 sat shadow ─────────────────────────────────
static int8_t fp16_to_i8_sat_shadow(uint16_t x) {
    int sign = (x >> 15) & 1;
    int bexp = (x >> 10) & 0x1F;
    int frac = x & 0x3FF;
    if (bexp == 0x1F && frac != 0) return 0;
    if (bexp == 0x1F) return sign ? (int8_t)-128 : (int8_t)127;
    if (bexp == 0)    return 0;

    int e_unb = bexp - 15;
    if (e_unb >= 7)  return sign ? (int8_t)-128 : (int8_t)127;
    if (e_unb <= -2) return 0;

    uint32_t sig11 = (1u << 10) | frac;
    int sh = 10 - e_unb;   // 4..11
    uint32_t mag_int = (sig11 >> sh) & 0xFFu;
    int guard = (int)((sig11 >> (sh - 1)) & 1u);
    uint32_t mask = (sh >= 2) ? ((1u << (sh - 1)) - 1u) : 0u;
    int sticky = (sig11 & mask) ? 1 : 0;
    int round_up = (guard && (sticky || (mag_int & 1))) ? 1 : 0;
    uint32_t mag_rnd = mag_int + round_up;

    if (!sign && mag_rnd >= 128) return (int8_t)127;
    if ( sign && mag_rnd >  128) return (int8_t)-128;
    if ( sign && mag_rnd == 128) return (int8_t)-128;
    return sign ? (int8_t)(-(int)mag_rnd) : (int8_t)mag_rnd;
}

// ─── full requant shadow ────────────────────────────────────
static int8_t requant_shadow(int32_t acc, uint16_t scale, uint16_t bias) {
    CvtOut c   = i32_to_fp16_shadow(acc);
    uint16_t sb = bump_scale_shadow(scale, c.shift);
    uint16_t f2 = fp16_fma_shadow(c.fp16, sb, bias);
    return fp16_to_i8_sat_shadow(f2);
}

// ─── driver ─────────────────────────────────────────────────
struct Stim {
    int32_t  acc;
    uint16_t scale;
    uint16_t bias;
    int8_t   expected;
    bool     valid;
};

static std::string hex16(uint16_t v) {
    char b[8];
    std::snprintf(b, sizeof(b), "0x%04X", v);
    return std::string(b);
}
static std::string hex32(uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof(b), "0x%08X", v);
    return std::string(b);
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 1200000000ull;

    sim.dut->valid_i      = 0;
    sim.dut->acc_i        = 0;
    sim.dut->scale_fp16_i = 0;
    sim.dut->bias_fp16_i  = 0;
    sim.reset();

    // ─── Test 1: reset ────────────────────────────────────
    printf("test 1: post-reset state\n");
    sim.check(sim.dut->y_dut_o == 0,         "dut y == 0 after reset");
    sim.check(sim.dut->y_ref_o == 0,         "ref y == 0 after reset");
    sim.check(sim.dut->valid_dut_o == 0,     "dut valid == 0 after reset");
    sim.check(sim.dut->valid_ref_o == 0,     "ref valid == 0 after reset");
    sim.check(sim.dut->mismatch_o == 0,      "no mismatch after reset");

    // ─── helper: stream model with built-in LATENCY alignment ───
    //
    // After the K-th tick (counting from reset), the outputs reflect the
    // value applied on the (K - LATENCY)-th tick (since each pipe stage is
    // a single posedge register). We model this with a queue: enqueue on
    // every apply(), and once the queue depth exceeds LATENCY the front
    // entry is the one whose result is on the wires *right now*.
    std::queue<Stim> pipe;
    int n_mismatch_total   = 0;
    int n_shadow_fail_total = 0;
    int first_print = 0;
    const char* phase = "";

    auto consume_head = [&]() {
        // After K ticks, queue depth is K and y_o reflects the entry that
        // was pushed at tick (K - LATENCY + 1), which is the front of the
        // queue once depth >= LATENCY.
        if ((int)pipe.size() < LATENCY) return;
        Stim head = pipe.front();
        pipe.pop();
        int8_t yd = (int8_t)sim.dut->y_dut_o;
        int8_t yr = (int8_t)sim.dut->y_ref_o;
        bool vd = sim.dut->valid_dut_o;
        bool vr = sim.dut->valid_ref_o;
        if (sim.dut->mismatch_o) {
            if (first_print < 10) {
                printf("  [%s] dut-vs-ref mismatch: acc=%s sc=%s bs=%s "
                       "dut=%d ref=%d sh=%d vd=%d vr=%d valid_q=%d\n",
                       phase,
                       hex32((uint32_t)head.acc).c_str(),
                       hex16(head.scale).c_str(),
                       hex16(head.bias).c_str(),
                       (int)yd, (int)yr, (int)head.expected,
                       (int)vd, (int)vr, (int)head.valid);
                first_print++;
            }
            n_mismatch_total++;
        }
        if (head.valid && yd != head.expected) {
            if (first_print < 10) {
                printf("  [%s] dut-vs-shadow mismatch: acc=%s sc=%s bs=%s "
                       "dut=%d sh=%d\n",
                       phase,
                       hex32((uint32_t)head.acc).c_str(),
                       hex16(head.scale).c_str(),
                       hex16(head.bias).c_str(),
                       (int)yd, (int)head.expected);
                first_print++;
            }
            n_shadow_fail_total++;
        }
    };

    auto apply = [&](int32_t acc, uint16_t scale, uint16_t bias, bool valid) {
        sim.dut->valid_i      = valid;
        sim.dut->acc_i        = acc;
        sim.dut->scale_fp16_i = scale;
        sim.dut->bias_fp16_i  = bias;
        int8_t exp = requant_shadow(acc, scale, bias);
        pipe.push(Stim{acc, scale, bias, exp, valid});
        sim.tick();
        consume_head();
    };

    // Drain remaining entries by pushing LATENCY invalid-no-op cycles
    // (so every in-flight result reaches the output and gets popped), then
    // flushing what's still queued.
    auto drain = [&]() {
        for (int i = 0; i < LATENCY; i++) apply(0, 0, 0, false);
        while (!pipe.empty()) pipe.pop();
    };

    // ─── Test 2: directed ───────────────────────────────
    printf("test 2: directed edge cases\n");
    struct Case { int32_t acc; uint16_t scale; uint16_t bias; const char* desc; };
    const uint16_t FP16_1   = 0x3C00;
    const uint16_t FP16_0   = 0x0000;
    const uint16_t FP16_N0  = 0x8000;
    const uint16_t FP16_INF = 0x7C00;
    const uint16_t FP16_NINF = 0xFC00;
    const uint16_t FP16_NAN = 0x7E00;
    const Case cases[] = {
        {        0, FP16_1, FP16_0,   "0*1+0 = 0" },
        {        5, FP16_1, FP16_0,   "5*1+0 = 5" },
        {     1000, FP16_0, FP16_1,   "acc*0 + 1 = 1" },
        {  INT32_MAX, FP16_1, FP16_0, "INT32_MAX*1 = +Inf → +127" },
        {  INT32_MIN, FP16_1, FP16_0, "INT32_MIN*1 = -Inf → -128" },
        {      127, FP16_1, FP16_0,   "127 → +127" },
        {      128, FP16_1, FP16_0,   "128 saturates to +127" },
        {     -128, FP16_1, FP16_0,   "-128 → -128" },
        {     -129, FP16_1, FP16_0,   "-129 saturates to -128" },
        {       50, FP16_0, FP16_0,   "scale +0 → 0" },
        {       50, FP16_N0, FP16_0,  "scale -0 → 0" },
        {       50, FP16_1, FP16_INF, "bias +Inf → +127" },
        {       50, FP16_1, FP16_NINF,"bias -Inf → -128" },
        {       50, FP16_1, FP16_NAN, "bias NaN → 0 (NaN propagates)" },
        {       50, FP16_NAN, FP16_0, "scale NaN → 0" },
        {  INT32_MAX, FP16_0, FP16_0, "huge*0+0 → 0" },
    };
    phase = "directed";
    // Make sure pipe is empty so each case can be flushed end-to-end.
    drain();
    for (auto& cs : cases) {
        // Apply the case for one cycle, then push LATENCY-1 no-op cycles
        // so queue depth reaches LATENCY and consume_head pops the case
        // when the DUT output reflects this case (3 ticks after drive).
        apply(cs.acc, cs.scale, cs.bias, true);
        for (int i = 0; i < LATENCY - 1; i++) apply(0, 0, 0, false);
    }
    drain();
    // Per-case correctness is captured by the global counters; assert now.
    sim.check(n_mismatch_total == 0,
              "directed: DUT vs REF mismatches = " +
              std::to_string(n_mismatch_total));
    sim.check(n_shadow_fail_total == 0,
              "directed: DUT vs shadow mismatches = " +
              std::to_string(n_shadow_fail_total));

    // ─── Test 3: randomized stress ───────────────────────
    printf("test 3: 20000 random (acc, scale, bias) triples\n");
    std::mt19937 rng(0xC0FFEE42u);
    std::uniform_int_distribution<uint64_t> d32(0, 0xFFFFFFFFull);
    std::uniform_int_distribution<uint64_t> d16(0, 0xFFFFull);

    phase = "rand1";
    int base_mis = n_mismatch_total, base_sh = n_shadow_fail_total;
    const int N = 20000;
    for (int i = 0; i < N; i++) {
        int32_t  acc  = (int32_t)d32(rng);
        uint16_t scl  = (uint16_t)d16(rng);
        uint16_t bs   = (uint16_t)d16(rng);
        if ((i & 0x1F) == 0)  acc = 0;
        if ((i & 0x3F) == 0)  acc = INT32_MAX;
        if ((i & 0x3F) == 1)  acc = INT32_MIN;
        if ((i & 0x1F) == 1)  scl = 0x3C00;
        if ((i & 0x1F) == 2)  bs  = 0x0000;
        if ((i & 0xFF) == 3)  scl = 0x7C00;
        if ((i & 0xFF) == 4)  bs  = 0x7E00;
        apply(acc, scl, bs, true);
    }
    drain();
    sim.check(n_mismatch_total - base_mis == 0,
              "rand: 0 dut-vs-ref mismatches (was " +
              std::to_string(n_mismatch_total - base_mis) + ")");
    sim.check(n_shadow_fail_total - base_sh == 0,
              "rand: 0 dut-vs-shadow mismatches (was " +
              std::to_string(n_shadow_fail_total - base_sh) + ")");

    // ─── Test 3b: big-acc + small-scale prescale scenarios ───
    // Validates that the in-IP autoscale lets layer code skip the old
    // ACC_SHIFT workaround: feed a huge accumulator with a sub-unit scale
    // and check the result against the integer-precision __int128 shadow.
    printf("test 3b: big-acc * small-scale (autoscale path)\n");
    base_mis = n_mismatch_total;
    base_sh  = n_shadow_fail_total;
    phase = "bigacc";
    drain();
    struct PCase {
        int32_t  acc;
        uint16_t scale;
        uint16_t bias;
        const char* desc;
    };
    // fp16 encodings for reference:
    //   1.0     = 0x3C00     2^-10  = 0x1400     2^-6 = 0x2400
    //   2^-20 ≈ 0x1000 (denormal) — exercises the no-bump path.
    //   1/1024 = 0x2400 (=2^-10? no, 2^-10 = 0x1400). Let me list normals:
    //   1.0    = 0x3C00 ( exp=15, mant=0 )
    //   0.5    = 0x3800
    //   2^-6   = 0x2400  (exp=9)
    //   2^-10  = 0x1400  (exp=5)
    //   2^-14  = 0x0400  (smallest normal, exp=1)
    const PCase pcases[] = {
        { 10'000'000,   0x2400, 0x0000, "10M  * 2^-6  + 0" },     // ≈ 156250
        { -10'000'000,  0x2400, 0x0000, "-10M * 2^-6  + 0" },
        { 100'000'000,  0x1400, 0x0000, "100M * 2^-10 + 0" },     // ≈ 97656
        { 1 << 25,      0x1400, 0x3C00, "2^25 * 2^-10 + 1.0" },   // ≈ 32769
        { 1 << 28,      0x0400, 0x0000, "2^28 * 2^-14 + 0" },     // ≈ 16384
        { INT32_MAX,    0x1400, 0x0000, "INT32_MAX * 2^-10" },    // huge / 1024
        { INT32_MIN,    0x1400, 0x0000, "INT32_MIN * 2^-10" },
        { 1'000'000,    0x3C00, 0x0000, "1M  * 1.0  -> saturate to +127" },
        { -1'000'000,   0x3C00, 0x0000, "-1M * 1.0  -> saturate to -128" },
        { 50'000'000,   0x2400, 0x0000, "50M * 2^-6 -> ~781250 -> +127" },
        { 12345,        0x2400, 0x4000, "small acc * 2^-6 + 2.0" },
    };
    for (auto& cs : pcases) {
        apply(cs.acc, cs.scale, cs.bias, true);
        for (int i = 0; i < LATENCY - 1; i++) apply(0, 0, 0, false);
    }
    drain();
    sim.check(n_mismatch_total - base_mis == 0,
              "bigacc: 0 dut-vs-ref mismatches (was " +
              std::to_string(n_mismatch_total - base_mis) + ")");
    sim.check(n_shadow_fail_total - base_sh == 0,
              "bigacc: 0 dut-vs-shadow mismatches (was " +
              std::to_string(n_shadow_fail_total - base_sh) + ")");

    // ─── Test 3c: random big-acc + bounded-scale stress ───
    // Exhaustively exercises the autoscale + exponent-bump path with
    // scales restricted to normal fp16 in the range ~[2^-14, 2^0].
    printf("test 3c: 30000 random big-acc + small-scale triples\n");
    base_mis = n_mismatch_total;
    base_sh  = n_shadow_fail_total;
    phase = "bigacc-rand";
    {
        std::uniform_int_distribution<int32_t>  d_acc(INT32_MIN/4, INT32_MAX/4);
        std::uniform_int_distribution<uint16_t> d_exp(1, 15);     // 2^-14 .. 2^0
        std::uniform_int_distribution<uint16_t> d_frac(0, 0x3FF);
        std::uniform_int_distribution<uint16_t> d_sgn(0, 1);
        for (int i = 0; i < 30000; i++) {
            int32_t acc = d_acc(rng);
            // mix in some really huge accs
            if ((i & 0xF) == 0) acc *= 8;
            uint16_t scl = ((uint16_t)d_sgn(rng) << 15) |
                           ((uint16_t)d_exp(rng) << 10) |
                           d_frac(rng);
            // small bias in normal range, occasionally zero
            uint16_t bs = ((i & 7) == 0) ? 0x0000 :
                          (((uint16_t)d_sgn(rng) << 15) |
                           ((uint16_t)(d_exp(rng) + 5) << 10) |
                           d_frac(rng));
            apply(acc, scl, bs, true);
        }
    }
    drain();
    sim.check(n_mismatch_total - base_mis == 0,
              "bigacc-rand: 0 dut-vs-ref mismatches (was " +
              std::to_string(n_mismatch_total - base_mis) + ")");
    sim.check(n_shadow_fail_total - base_sh == 0,
              "bigacc-rand: 0 dut-vs-shadow mismatches (was " +
              std::to_string(n_shadow_fail_total - base_sh) + ")");

    // ─── Test 4: heavier random with bounded exponents ───
    printf("test 4: 10000 more random vectors, broad coverage\n");
    base_mis = n_mismatch_total;
    base_sh  = n_shadow_fail_total;
    phase = "rand2";
    for (int i = 0; i < 10000; i++) {
        int32_t  acc  = (int32_t)d32(rng);
        uint16_t scl  = (uint16_t)d16(rng);
        uint16_t bs   = (uint16_t)d16(rng);
        if ((i % 5) != 0) {
            scl = (uint16_t)((scl & 0xF3FF) | ((scl & 0x1C00) ? (scl & 0x1C00) : 0x1000));
            bs  = (uint16_t)((bs  & 0xF3FF) | ((bs  & 0x1C00) ? (bs  & 0x1C00) : 0x1000));
        }
        apply(acc, scl, bs, true);
    }
    drain();
    sim.check(n_mismatch_total - base_mis == 0,
              "rand2: 0 dut-vs-ref mismatches (was " +
              std::to_string(n_mismatch_total - base_mis) + ")");
    sim.check(n_shadow_fail_total - base_sh == 0,
              "rand2: 0 dut-vs-shadow mismatches (was " +
              std::to_string(n_shadow_fail_total - base_sh) + ")");

    // ─── Test 5: valid-bit gating ───────────────────────
    printf("test 5: valid bit propagates correctly\n");
    drain();
    // Pipe is empty and last LATENCY+1 cycles were invalid no-ops, so
    // valid_o is low.
    for (int i = 0; i < 3; i++) {
        sim.dut->valid_i = 0;
        sim.dut->acc_i   = 0;
        sim.dut->scale_fp16_i = 0;
        sim.dut->bias_fp16_i  = 0;
        sim.tick();
    }
    sim.check(sim.dut->valid_dut_o == 0, "valid_o low when no valid_i");
    sim.check(sim.dut->valid_ref_o == 0, "ref valid_o low when no valid_i");

    // Pulse valid_i high for one tick. The LATENCY-deep shift register
    // carries the bit so valid_o asserts exactly LATENCY ticks after the
    // pulse-tick (counting the pulse-tick itself).
    sim.dut->valid_i = 1;
    sim.dut->acc_i   = 100;
    sim.dut->scale_fp16_i = 0x3C00; // 1.0
    sim.dut->bias_fp16_i  = 0x0000;
    sim.tick();                          // pulse-tick: valid enters stage 0
    sim.dut->valid_i = 0;
    sim.dut->acc_i   = 0;
    // valid_o must stay low for LATENCY-1 ticks after the pulse-tick…
    for (int i = 0; i < LATENCY - 1; i++) {
        sim.check(sim.dut->valid_dut_o == 0,
                  "valid_o low " + std::to_string(i) + " ticks after pulse");
        sim.tick();
    }
    // …and assert on the LATENCY-th tick.
    sim.check(sim.dut->valid_dut_o == 1,
              "valid_o high LATENCY ticks after pulse");
    sim.check(sim.dut->valid_ref_o == 1,
              "ref valid_o high LATENCY ticks after pulse");
    sim.check((int8_t)sim.dut->y_dut_o == 100, "y_o == 100 for 100*1+0");

    // ─── Test 6: mid-stream reset ───────────────────────
    printf("test 6: mid-stream reset\n");
    sim.dut->valid_i = 1;
    sim.dut->acc_i = 42;
    sim.dut->scale_fp16_i = 0x3C00;
    sim.dut->bias_fp16_i  = 0x0000;
    sim.tick();
    sim.dut->valid_i = 0; sim.dut->acc_i = 0;
    sim.dut->scale_fp16_i = 0; sim.dut->bias_fp16_i = 0;
    sim.reset();
    sim.check(sim.dut->y_dut_o == 0, "dut y==0 after re-reset");
    sim.check(sim.dut->y_ref_o == 0, "ref y==0 after re-reset");
    sim.check(sim.dut->valid_dut_o == 0, "dut valid==0 after re-reset");
    sim.check(sim.dut->valid_ref_o == 0, "ref valid==0 after re-reset");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after re-reset");

    return sim.finish();
}
