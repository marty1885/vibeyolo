// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// topk_fp16 — Verilator test. Drives an fp16 candidate stream into the
// DUT/REF lockstep TB, then verifies the K returned (value, index) pairs
// are exactly the top-K of the input stream (compared as multisets — heap
// order is unspecified).
//
// Two configs are built and run:
//   • big   : N=8400, K=300, IDX_W=14 (production shape)
//   • small : N=32,   K=4,   IDX_W=5  (fast iteration)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "sim_ctrl.h"

// Both TB tops share the same port set (just different widths). Build the
// driver as a template over the Verilated TB type.

// ───────────────────────── fp16 helpers ───────────────────────────

static double fp16_to_double(uint16_t x) {
    int s = (x >> 15) & 1;
    int e = (x >> 10) & 0x1F;
    int f = x & 0x3FF;
    double v;
    if (e == 0x1F) {
        v = (f == 0) ? std::numeric_limits<double>::infinity()
                     : std::numeric_limits<double>::quiet_NaN();
    } else if (e == 0) {
        v = std::ldexp((double)f, -24);
    } else {
        v = std::ldexp(1.0 + (double)f / 1024.0, e - 15);
    }
    return s ? -v : v;
}

static uint16_t double_to_fp16(double v) {
    if (std::isnan(v)) return 0x7E00;
    int s = std::signbit(v) ? 1 : 0;
    double av = std::fabs(v);
    if (std::isinf(av) || av >= 65520.0) return (uint16_t)((s << 15) | 0x7C00);
    if (av == 0.0) return (uint16_t)(s << 15);
    int e;
    double m = std::frexp(av, &e);
    int unbiased = e - 1;
    int biased = unbiased + 15;
    if (biased >= 31) return (uint16_t)((s << 15) | 0x7C00);
    if (biased <= 0) {
        double scaled = av * (double)(1 << 24);
        double floor_v = std::floor(scaled);
        double frac = scaled - floor_v;
        long mant_int = (long)floor_v;
        if (frac > 0.5)                            mant_int += 1;
        else if (frac == 0.5 && (mant_int & 1))    mant_int += 1;
        if (mant_int >= 1024) return (uint16_t)((s << 15) | (1 << 10));
        return (uint16_t)((s << 15) | (mant_int & 0x3FF));
    }
    double mant_d = (m * 2.0 - 1.0) * 1024.0;
    double floor_v = std::floor(mant_d);
    double frac = mant_d - floor_v;
    long mant_int = (long)floor_v;
    if (frac > 0.5)                            mant_int += 1;
    else if (frac == 0.5 && (mant_int & 1))    mant_int += 1;
    if (mant_int >= 1024) {
        biased += 1;
        mant_int = 0;
        if (biased >= 31) return (uint16_t)((s << 15) | 0x7C00);
    }
    return (uint16_t)((s << 15) | ((biased & 0x1F) << 10) | (mant_int & 0x3FF));
}

// ───────────────────────── flat-port helpers ──────────────────────

// Read a bit range from a wide Verilator port. Ports may appear as a
// scalar integer (uint32_t / uint64_t for ≤64-bit wide signals) or as
// a VlWide<N> array of 32-bit words (for >64-bit wide signals). We unify
// both via a generic `read_word(port, w)` helper.

#include <type_traits>

template <typename PortT>
static uint32_t read_word(PortT& port, int w) {
    if constexpr (std::is_integral_v<PortT>) {
        // Scalar: shift down by w*32, mask 32 bits.
        if (w * 32 >= (int)(sizeof(PortT) * 8)) return 0u;
        return (uint32_t)((port >> (w * 32)) & 0xFFFFFFFFu);
    } else {
        return (uint32_t)port.at(w);
    }
}

template <typename PortT>
static uint64_t read_bits(PortT& port, int bitpos, int nbits) {
    int w  = bitpos / 32;
    int sh = bitpos % 32;
    uint64_t lo = read_word(port, w);
    uint64_t hi = (sh + nbits > 32) ? read_word(port, w + 1) : 0u;
    uint64_t combined = (hi << 32) | lo;
    uint64_t mask = (nbits >= 64) ? ~0ull : ((1ull << nbits) - 1ull);
    return (combined >> sh) & mask;
}

template <typename PortT>
static uint16_t read_val_lane(PortT& port, int lane) {
    return (uint16_t)read_bits(port, lane * 16, 16);
}

template <typename PortT>
static uint32_t read_idx_lane(PortT& port, int lane, int idx_w) {
    return (uint32_t)read_bits(port, lane * idx_w, idx_w);
}

// ───────────────────────── canonical compare ──────────────────────

// Compare two multisets of (value, index) pairs. fp16 ordering: convert
// to double — exact since fp16 has only 11 significand bits.

struct Item {
    uint16_t v;
    uint32_t idx;
    bool operator==(const Item& o) const { return v == o.v && idx == o.idx; }
    // sort: descending by value, then ascending by index (canonical)
    bool operator<(const Item& o) const {
        double a = fp16_to_double(v);
        double b = fp16_to_double(o.v);
        if (a != b) return a > b;
        return idx < o.idx;
    }
};

// Compute the gold top-K from the full input stream.
static std::vector<Item> compute_topk(const std::vector<uint16_t>& vals,
                                      int K) {
    std::vector<Item> items;
    items.reserve(vals.size());
    for (size_t i = 0; i < vals.size(); i++) {
        items.push_back({vals[i], (uint32_t)i});
    }
    std::sort(items.begin(), items.end());
    if ((int)items.size() > K) items.resize(K);
    return items;
}

// ───────────────────────── per-config driver ─────────────────────

template <typename DUT>
struct Runner {
    SimCtrl<DUT>& sim;
    int N;
    int K;
    int IDX_W;

    Runner(SimCtrl<DUT>& s, int N_, int K_, int IDX_W_)
        : sim(s), N(N_), K(K_), IDX_W(IDX_W_) {}

    // Run one frame; returns observed cycle count for completion (from
    // start_i deassert to done_dut_o pulse).
    struct FrameResult {
        std::vector<Item> dut_set;
        std::vector<Item> ref_set;
        std::vector<Item> gold_set;
        uint64_t cycles;
        bool done_seen;
    };

    FrameResult run_frame(const std::vector<uint16_t>& vals,
                          const std::vector<uint32_t>& idxs) {
        FrameResult r{};
        r.done_seen = false;

        // Start pulse
        sim.dut->start_i    = 1;
        sim.dut->in_valid_i = 0;
        sim.dut->in_value_i = 0;
        sim.dut->in_index_i = 0;
        sim.tick();
        sim.dut->start_i = 0;

        uint64_t t0 = sim.sim_time;
        size_t   feed = 0;
        uint64_t safety = (uint64_t)N * 20 + 1000;

        while (sim.sim_time - t0 < safety * 2) {
            if (sim.dut->done_dut_o) {
                r.done_seen = true;
                break;
            }
            if (sim.dut->in_ready_dut_o && feed < vals.size()) {
                sim.dut->in_valid_i = 1;
                sim.dut->in_value_i = vals[feed];
                sim.dut->in_index_i = idxs[feed];
                feed++;
            } else {
                sim.dut->in_valid_i = 0;
            }
            sim.tick();
        }
        sim.dut->in_valid_i = 0;
        // SimCtrl bumps sim_time twice per tick (clk_low + clk_high), so
        // divide by 2 to report cycles.
        r.cycles = (sim.sim_time - t0) / 2;

        // Snapshot outputs
        r.dut_set.resize(K);
        r.ref_set.resize(K);
        for (int i = 0; i < K; i++) {
            r.dut_set[i].v   = read_val_lane(sim.dut->out_value_dut_o, i);
            r.dut_set[i].idx = read_idx_lane(sim.dut->out_index_dut_o, i, IDX_W);
            r.ref_set[i].v   = read_val_lane(sim.dut->out_value_ref_o, i);
            r.ref_set[i].idx = read_idx_lane(sim.dut->out_index_ref_o, i, IDX_W);
        }
        // gold
        r.gold_set = compute_topk(vals, K);

        // Drop a few cycles between frames
        for (int k = 0; k < 5; k++) sim.tick();
        return r;
    }
};

// ───────────────────────── stimulus generators ───────────────────

static uint16_t rand_fp16_unit(std::mt19937& rng) {
    // Generate a fp16 in [0, 1] — matches what sigmoid emits.
    std::uniform_real_distribution<double> u(0.0, 1.0);
    return double_to_fp16(u(rng));
}

// ───────────────────────── set comparator ────────────────────────

// Compare two multisets of values (sorted by value only); duplicate-tolerant.
static bool same_value_multiset(std::vector<Item> a, std::vector<Item> b) {
    auto by_v = [](const Item& x, const Item& y) {
        return fp16_to_double(x.v) > fp16_to_double(y.v);
    };
    std::sort(a.begin(), a.end(), by_v);
    std::sort(b.begin(), b.end(), by_v);
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].v != b[i].v) return false;
    }
    return true;
}

// Validate every (v, idx) in s actually corresponds to vals[idx] == v.
static bool indices_valid(const std::vector<Item>& s,
                          const std::vector<uint16_t>& vals) {
    for (auto& it : s) {
        if (it.idx >= vals.size()) return false;
        if (vals[it.idx] != it.v)  return false;
    }
    return true;
}

// ───────────────────────── per-config test body ──────────────────

template <typename DUT>
static void run_tests(SimCtrl<DUT>& sim, int N, int K, int IDX_W,
                      const char* tag, uint64_t& worst_cycles_out) {
    sim.dut->start_i    = 0;
    sim.dut->in_valid_i = 0;
    sim.dut->in_value_i = 0;
    sim.dut->in_index_i = 0;
    sim.reset();

    sim.check(sim.dut->done_dut_o == 0,
              std::string(tag) + ": done_dut low after reset");
    sim.check(sim.dut->done_ref_o == 0,
              std::string(tag) + ": done_ref low after reset");
    sim.check(sim.dut->in_ready_dut_o == 0,
              std::string(tag) + ": in_ready_dut low before start");

    Runner<DUT> runner(sim, N, K, IDX_W);

    // Helper: drive a single frame and assert.
    auto check_frame = [&](const std::vector<uint16_t>& vals,
                           const std::string& name) {
        std::vector<uint32_t> idxs(vals.size());
        for (size_t i = 0; i < vals.size(); i++) idxs[i] = (uint32_t)i;
        auto r = runner.run_frame(vals, idxs);
        sim.check(r.done_seen,
                  std::string(tag) + "/" + name + ": done_dut_o pulsed");
        sim.check(same_value_multiset(r.dut_set, r.gold_set),
                  std::string(tag) + "/" + name + ": DUT matches gold top-K");
        sim.check(same_value_multiset(r.ref_set, r.gold_set),
                  std::string(tag) + "/" + name + ": REF matches gold top-K");
        sim.check(indices_valid(r.dut_set, vals),
                  std::string(tag) + "/" + name + ": DUT indices map back");
        if (r.cycles > worst_cycles_out) worst_cycles_out = r.cycles;
        printf("  %s/%s cycles=%lu done=%d\n",
               tag, name.c_str(), (unsigned long)r.cycles, (int)r.done_seen);
    };

    // ── Edge: sorted ascending (every input replaces root) ──
    {
        std::vector<uint16_t> vals(N);
        for (int i = 0; i < N; i++)
            vals[i] = double_to_fp16((double)(i + 1) / (double)(N + 1));
        check_frame(vals, "sorted_asc");
    }

    // ── Edge: sorted descending (only first K matter) ──
    {
        std::vector<uint16_t> vals(N);
        for (int i = 0; i < N; i++)
            vals[i] = double_to_fp16((double)(N - i) / (double)(N + 1));
        check_frame(vals, "sorted_desc");
    }

    // ── Edge: all equal ──
    {
        std::vector<uint16_t> vals(N, double_to_fp16(0.5));
        check_frame(vals, "all_equal");
    }

    // ── Edge: single peak in the middle ──
    {
        std::vector<uint16_t> vals(N, double_to_fp16(0.001));
        vals[N / 2] = double_to_fp16(0.999);
        check_frame(vals, "single_peak");
    }

    // ── Edge: K duplicates of max, rest small ──
    {
        std::vector<uint16_t> vals(N, double_to_fp16(0.01));
        // place K duplicates of 0.9 at scattered positions
        std::mt19937 rng(0xAB1Au);
        std::vector<int> pos(N);
        for (int i = 0; i < N; i++) pos[i] = i;
        std::shuffle(pos.begin(), pos.end(), rng);
        for (int i = 0; i < K; i++) vals[pos[i]] = double_to_fp16(0.9);
        check_frame(vals, "k_duplicates");
    }

    // ── Random frames ──
    int nrand = (N > 1000) ? 3 : 8;
    std::mt19937 rng(0xCAFEBEE5u);
    for (int s = 0; s < nrand; s++) {
        std::vector<uint16_t> vals(N);
        for (int i = 0; i < N; i++) vals[i] = rand_fp16_unit(rng);
        check_frame(vals, "rand_" + std::to_string(s));
    }

    // ── Worst-case bound check ──
    // Spec: K*9 + (N-K)*10 + slack. Each accepted element costs ≤ 1 cycle
    // accept + ≤ ceil(log2(K)) cycles sift. For the implementation we use,
    // worst-case per element ≤ 1 + ceil(log2(K)) ≈ 10 for K=300.
    int log2K = 0;
    while ((1 << log2K) < K) log2K++;
    uint64_t bound = (uint64_t)N * (uint64_t)(1 + log2K) + 200;
    sim.check(worst_cycles_out <= bound,
              std::string(tag) + ": worst-case cycles within bound (got "
              + std::to_string(worst_cycles_out) + " <= "
              + std::to_string(bound) + ")");
}

// ───────────────────────── main entry points ─────────────────────
//
// The Verilator build is one binary per config (multi-test mode); the C++
// file is compiled with -DCFG_BIG or -DCFG_SMALL to pick which top.

#if defined(CFG_BIG)
#include "Vtopk_fp16_tb_big.h"
using DUT_T = Vtopk_fp16_tb_big;
static const int CFG_N     = 8400;
static const int CFG_K     = 300;
static const int CFG_IDXW  = 14;
static const char* CFG_TAG = "big";
#elif defined(CFG_SMALL)
#include "Vtopk_fp16_tb_small.h"
using DUT_T = Vtopk_fp16_tb_small;
static const int CFG_N     = 32;
static const int CFG_K     = 4;
static const int CFG_IDXW  = 5;
static const char* CFG_TAG = "small";
#else
#error "Define CFG_BIG or CFG_SMALL when compiling topk_fp16_test.cc"
#endif

int main(int argc, char** argv) {
    SimCtrl<DUT_T> sim(argc, argv);
    sim.max_time = 800000000ull;

    uint64_t worst = 0;
    run_tests<DUT_T>(sim, CFG_N, CFG_K, CFG_IDXW, CFG_TAG, worst);
    printf("\n%s: worst-frame cycles = %lu\n", CFG_TAG, (unsigned long)worst);

    return sim.finish();
}
