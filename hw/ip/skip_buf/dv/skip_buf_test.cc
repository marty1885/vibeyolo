// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// skip_buf — Verilator test. Drives DUT and SV REF in lockstep through
// a parameterized TB wrapper (skip_buf_tb_small for H=W=C=4 → Depth=64,
// skip_buf_tb_large for H=W=C=8 → Depth=512). For every cycle we assert
// `mismatch_o` is 0, and we compare the drained output against a C++
// `std::vector<int8_t>` shadow of the writes.

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#ifndef TB_CONFIG
#define TB_CONFIG 4
#endif

#if TB_CONFIG == 4
  #include "Vskip_buf_tb_small.h"
  using DUT = Vskip_buf_tb_small;
  static constexpr int H = 4, W = 4, C = 4;
#elif TB_CONFIG == 8
  #include "Vskip_buf_tb_large.h"
  using DUT = Vskip_buf_tb_large;
  static constexpr int H = 8, W = 8, C = 8;
#else
  #error "Unsupported TB_CONFIG"
#endif

static constexpr int DEPTH = H * W * C;

#include "sim_ctrl.h"

// Issue one cycle of stimulus; sample mismatch and any output beat.
struct CycleObs {
    bool w_fire;
    bool r_fire;
    int8_t rdata;
    bool mismatch;
};

static CycleObs do_cycle(SimCtrl<DUT>& sim,
                         bool want_w, int8_t wd,
                         bool want_r,
                         int& mismatch_cnt,
                         int cycles) {
    sim.dut->wvalid_i = want_w ? 1 : 0;
    sim.dut->wdata_i  = want_w ? static_cast<uint8_t>(wd) : 0;
    sim.dut->rready_i = want_r ? 1 : 0;
    sim.dut->eval();

    CycleObs obs{};
    obs.w_fire   = want_w && (sim.dut->wready_dut_o != 0);
    obs.r_fire   = want_r && (sim.dut->rvalid_dut_o != 0);
    obs.rdata    = static_cast<int8_t>(sim.dut->rdata_dut_o);
    obs.mismatch = sim.dut->mismatch_o != 0;

    if (obs.mismatch) {
        if (mismatch_cnt < 4) {
            printf("  mismatch @cy=%d: w_dut=%d w_ref=%d r_v_dut=%d r_v_ref=%d "
                   "full_d=%d full_r=%d empty_d=%d empty_r=%d d_dut=%d d_ref=%d\n",
                   cycles,
                   int(sim.dut->wready_dut_o), int(sim.dut->wready_ref_o),
                   int(sim.dut->rvalid_dut_o), int(sim.dut->rvalid_ref_o),
                   int(sim.dut->full_dut_o),   int(sim.dut->full_ref_o),
                   int(sim.dut->empty_dut_o),  int(sim.dut->empty_ref_o),
                   int(int8_t(sim.dut->rdata_dut_o)),
                   int(int8_t(sim.dut->rdata_ref_o)));
        }
        mismatch_cnt++;
    }
    sim.tick();
    return obs;
}

// Fill (write Depth entries) then drain (read Depth entries) with optional
// stalls. Returns drained vector.
static std::vector<int8_t> fill_then_drain(SimCtrl<DUT>& sim,
                                           const std::vector<int8_t>& in,
                                           std::mt19937& rng,
                                           int wgap_pct, int rgap_pct,
                                           int& mismatch_cnt,
                                           const std::string& tag) {
    std::uniform_int_distribution<int> d100(0, 99);
    std::vector<int8_t> got;
    got.reserve(in.size());

    // Phase 1: write all entries.
    size_t in_idx = 0;
    int cycles = 0;
    int max_cycles = DEPTH * 100 + 1000;
    while (in_idx < in.size()) {
        bool want_w = d100(rng) >= wgap_pct;
        bool want_r = false;
        auto obs = do_cycle(sim, want_w, want_w ? in[in_idx] : 0,
                            want_r, mismatch_cnt, cycles);
        if (obs.w_fire) in_idx++;
        cycles++;
        if (cycles > max_cycles) {
            sim.check(false, tag + ": fill timeout");
            return got;
        }
    }
    // Quiesce and verify full.
    auto obs0 = do_cycle(sim, false, 0, false, mismatch_cnt, cycles++);
    (void)obs0;
    sim.check(sim.dut->full_dut_o == 1, tag + ": full_o after fill");
    sim.check(sim.dut->empty_dut_o == 0, tag + ": !empty_o after fill");
    sim.check(sim.dut->wready_dut_o == 0, tag + ": !wready_o after fill");

    // Phase 2: drain.
    cycles = 0;
    while (got.size() < in.size()) {
        bool want_r = d100(rng) >= rgap_pct;
        bool want_w = false;
        auto obs = do_cycle(sim, want_w, 0, want_r, mismatch_cnt, cycles);
        if (obs.r_fire) got.push_back(obs.rdata);
        cycles++;
        if (cycles > max_cycles) {
            sim.check(false, tag + ": drain timeout");
            return got;
        }
    }
    return got;
}

static bool compare_streams(const std::vector<int8_t>& got,
                            const std::vector<int8_t>& exp,
                            SimCtrl<DUT>& sim, const std::string& tag) {
    if (got.size() != exp.size()) {
        sim.check(false, tag + ": size got=" + std::to_string(got.size()) +
                  " exp=" + std::to_string(exp.size()));
        return false;
    }
    int bad = 0;
    for (size_t i = 0; i < got.size(); i++) {
        if (got[i] != exp[i]) {
            if (bad < 8) {
                printf("  %s: diff @%zu got=%d exp=%d\n",
                       tag.c_str(), i, int(got[i]), int(exp[i]));
            }
            bad++;
        }
    }
    sim.check(bad == 0, tag + ": data match (bad=" + std::to_string(bad) + ")");
    return bad == 0;
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 400000000ull;

    printf("skip_buf test (H=%d W=%d C=%d Depth=%d)\n", H, W, C, DEPTH);

    sim.dut->wvalid_i = 0;
    sim.dut->wdata_i  = 0;
    sim.dut->rready_i = 0;
    sim.dut->clr_i    = 0;
    sim.reset();

    // ── Test 1: reset state ─────────────────────────────
    printf("test 1: reset/clr state\n");
    sim.dut->eval();
    sim.check(sim.dut->empty_dut_o == 1, "empty_o=1 after reset (DUT)");
    sim.check(sim.dut->empty_ref_o == 1, "empty_o=1 after reset (REF)");
    sim.check(sim.dut->full_dut_o  == 0, "full_o=0 after reset (DUT)");
    sim.check(sim.dut->full_ref_o  == 0, "full_o=0 after reset (REF)");
    sim.check(sim.dut->wready_dut_o == 1, "wready_o=1 after reset (DUT)");
    sim.check(sim.dut->rvalid_dut_o == 0, "rvalid_o=0 after reset (DUT)");
    sim.check(sim.dut->mismatch_o   == 0, "no mismatch after reset");

    // Try a read while empty/not-full: rvalid_o must stay 0.
    sim.dut->rready_i = 1;
    sim.dut->eval();
    sim.check(sim.dut->rvalid_dut_o == 0, "rvalid_o=0 while not full");
    sim.dut->rready_i = 0;

    // ── Test 2: directed fill+drain, no back-pressure ───
    printf("test 2: directed fill+drain\n");
    {
        std::vector<int8_t> in;
        in.reserve(DEPTH);
        for (int i = 0; i < DEPTH; i++) in.push_back(int8_t((i & 0xFF) - 64));
        std::mt19937 rng(1);
        int mm = 0;
        auto got = fill_then_drain(sim, in, rng, 0, 0, mm, "directed");
        sim.check(mm == 0, "directed: 0 mismatch");
        compare_streams(got, in, sim, "directed");
        // Post-drain: empty asserted? Note: empty_o is wptr==0 && !full,
        // so after drain it remains 0 until clr_i. Verify rvalid drops.
        sim.dut->eval();
        sim.check(sim.dut->rvalid_dut_o == 0, "rvalid_o=0 post-drain");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();
    sim.dut->eval();
    sim.check(sim.dut->empty_dut_o == 1, "empty_o=1 after clr post-drain");

    // ── Test 3: write-while-full ignored ────────────────
    printf("test 3: write-while-full ignored\n");
    {
        std::vector<int8_t> in;
        for (int i = 0; i < DEPTH; i++) in.push_back(int8_t(i));
        std::mt19937 rng(2);
        int mm = 0;
        // Fill but do not drain.
        size_t idx = 0;
        int cy = 0;
        while (idx < in.size()) {
            auto obs = do_cycle(sim, true, in[idx], false, mm, cy++);
            if (obs.w_fire) idx++;
        }
        do_cycle(sim, false, 0, false, mm, cy++);
        sim.check(sim.dut->full_dut_o == 1, "full asserted");
        sim.check(sim.dut->wready_dut_o == 0, "wready=0 when full");
        // Try to write again: must be ignored.
        for (int i = 0; i < 5; i++) {
            auto obs = do_cycle(sim, true, int8_t(0x7F), false, mm, cy++);
            sim.check(!obs.w_fire, "no write fire while full");
        }
        sim.check(mm == 0, "write-while-full: 0 mismatch");
        // Drain and confirm original data intact.
        std::vector<int8_t> got;
        while (got.size() < in.size()) {
            auto obs = do_cycle(sim, false, 0, true, mm, cy++);
            if (obs.r_fire) got.push_back(obs.rdata);
        }
        compare_streams(got, in, sim, "write-while-full drain");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 4: random back-pressure on read ────────────
    printf("test 4: read back-pressure\n");
    {
        std::vector<int8_t> in;
        for (int i = 0; i < DEPTH; i++)
            in.push_back(int8_t(((i * 37) % 251) - 125));
        std::mt19937 rng(0xBEEF);
        int mm = 0;
        auto got = fill_then_drain(sim, in, rng, 0, 50, mm, "rgap");
        sim.check(mm == 0, "rgap: 0 mismatch");
        compare_streams(got, in, sim, "rgap");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 5: slow producer ───────────────────────────
    printf("test 5: slow producer\n");
    {
        std::vector<int8_t> in;
        std::mt19937 rng_in(0xC0DE);
        std::uniform_int_distribution<int> d8(-128, 127);
        for (int i = 0; i < DEPTH; i++) in.push_back(int8_t(d8(rng_in)));
        std::mt19937 rng(0xC0DE);
        int mm = 0;
        auto got = fill_then_drain(sim, in, rng, 50, 0, mm, "wgap");
        sim.check(mm == 0, "wgap: 0 mismatch");
        compare_streams(got, in, sim, "wgap");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 6: mid-fill clr_i drops partial data ───────
    printf("test 6: mid-fill clr_i\n");
    {
        std::mt19937 rng(0xABCD);
        std::uniform_int_distribution<int> d8(-128, 127);
        int mm = 0;
        // Partial fill: half of Depth.
        for (int i = 0; i < DEPTH / 2; i++) {
            do_cycle(sim, true, int8_t(d8(rng)), false, mm, i);
        }
        // Quiesce inputs before clr.
        sim.dut->wvalid_i = 0;
        sim.dut->rready_i = 0;
        // Clr.
        sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();
        sim.dut->eval();
        sim.check(sim.dut->empty_dut_o == 1, "mid-fill clr: empty");
        sim.check(sim.dut->full_dut_o  == 0, "mid-fill clr: !full");
        sim.check(sim.dut->wready_dut_o == 1, "mid-fill clr: wready=1");
        sim.check(mm == 0, "mid-fill clr: 0 mismatch up to now");

        // Now do a fresh fill+drain to confirm.
        std::vector<int8_t> in;
        for (int i = 0; i < DEPTH; i++) in.push_back(int8_t(d8(rng)));
        auto got = fill_then_drain(sim, in, rng, 20, 20, mm, "post-clr");
        sim.check(mm == 0, "post-clr: 0 mismatch");
        compare_streams(got, in, sim, "post-clr");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 7: random stress (multiple fill+drain cycles) ──
    printf("test 7: random stress, 3 cycles\n");
    {
        std::mt19937 rng(0xDEADBEEFu);
        std::uniform_int_distribution<int> d8(-128, 127);
        for (int iter = 0; iter < 3; iter++) {
            std::vector<int8_t> in;
            for (int i = 0; i < DEPTH; i++) in.push_back(int8_t(d8(rng)));
            int mm = 0;
            auto got = fill_then_drain(sim, in, rng, 30, 30, mm, "stress");
            sim.check(mm == 0, "stress: 0 mismatch");
            compare_streams(got, in, sim,
                            "stress[" + std::to_string(iter) + "]");
            sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();
        }
    }

    return sim.finish();
}
