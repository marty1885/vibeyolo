// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// concat_mux — Verilator test.
//
// Drives DUT and SV REF in lockstep via a parameterized TB wrapper
// (concat_mux_tb_c2_3_p4 or concat_mux_tb_c8_16_p8 — selected via
// -DTB_CA -DTB_CB -DTB_PIXELS at compile time). Each cycle:
//
//   * `mismatch_o` must be 0
//   * the DUT's emitted output stream, drained on every output handshake,
//     matches an independent C++ shadow that builds the expected
//     channel-concat interleaved sequence.

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#ifndef TB_CA
#define TB_CA 2
#endif
#ifndef TB_CB
#define TB_CB 3
#endif
#ifndef TB_PIXELS
#define TB_PIXELS 4
#endif

#if (TB_CA == 2) && (TB_CB == 3) && (TB_PIXELS == 4)
  #include "Vconcat_mux_tb_c2_3_p4.h"
  using DUT = Vconcat_mux_tb_c2_3_p4;
#elif (TB_CA == 8) && (TB_CB == 16) && (TB_PIXELS == 8)
  #include "Vconcat_mux_tb_c8_16_p8.h"
  using DUT = Vconcat_mux_tb_c8_16_p8;
#else
  #error "Unsupported (TB_CA,TB_CB,TB_PIXELS) combination"
#endif

static constexpr int CA     = TB_CA;
static constexpr int CB     = TB_CB;
static constexpr int PIXELS = TB_PIXELS;

#include "sim_ctrl.h"

// Build the expected output stream for input streams A (PIXELS*CA samples,
// channel-by-channel for each pixel) and B (PIXELS*CB likewise). Output is
// per-pixel: all CA A samples then all CB B samples.
static std::vector<int8_t> expected_concat(const std::vector<int8_t>& a,
                                           const std::vector<int8_t>& b) {
    std::vector<int8_t> out;
    out.reserve(a.size() + b.size());
    for (int p = 0; p < PIXELS; p++) {
        for (int c = 0; c < CA; c++) out.push_back(a[p * CA + c]);
        for (int c = 0; c < CB; c++) out.push_back(b[p * CB + c]);
    }
    return out;
}

static int drive_stream(SimCtrl<DUT>& sim,
                        const std::vector<int8_t>& a,
                        const std::vector<int8_t>& b,
                        std::vector<int8_t>& got,
                        std::mt19937& rng,
                        int agap_pct,
                        int bgap_pct,
                        int rgap_pct,
                        int& mismatch_cnt) {
    std::uniform_int_distribution<int> d100(0, 99);

    size_t ai = 0, bi = 0;
    const size_t expected_out = a.size() + b.size();
    int cycles = 0;
    const int max_cycles = static_cast<int>(expected_out) * 50 + 1000;

    sim.dut->avalid_i = 0;
    sim.dut->bvalid_i = 0;
    sim.dut->rready_i = 0;
    sim.dut->adata_i  = 0;
    sim.dut->bdata_i  = 0;

    while (got.size() < expected_out) {
        bool want_a = (ai < a.size()) && (d100(rng) >= agap_pct);
        bool want_b = (bi < b.size()) && (d100(rng) >= bgap_pct);
        bool want_r = (d100(rng) >= rgap_pct);

        sim.dut->avalid_i = want_a ? 1 : 0;
        sim.dut->adata_i  = want_a ? static_cast<uint8_t>(a[ai]) : 0;
        sim.dut->bvalid_i = want_b ? 1 : 0;
        sim.dut->bdata_i  = want_b ? static_cast<uint8_t>(b[bi]) : 0;
        sim.dut->rready_i = want_r ? 1 : 0;

        sim.dut->eval();

        bool a_fire = want_a && (sim.dut->aready_dut_o != 0);
        bool b_fire = want_b && (sim.dut->bready_dut_o != 0);
        bool r_fire = want_r && (sim.dut->rvalid_dut_o != 0);

        if (r_fire) {
            got.push_back(static_cast<int8_t>(sim.dut->rdata_dut_o));
        }

        if (sim.dut->mismatch_o) {
            if (mismatch_cnt < 4) {
                printf("  mismatch @cy=%d: ar_dut=%d ar_ref=%d br_dut=%d "
                       "br_ref=%d rv_dut=%d rv_ref=%d d_dut=%d d_ref=%d\n",
                       cycles,
                       int(sim.dut->aready_dut_o), int(sim.dut->aready_ref_o),
                       int(sim.dut->bready_dut_o), int(sim.dut->bready_ref_o),
                       int(sim.dut->rvalid_dut_o), int(sim.dut->rvalid_ref_o),
                       int(int8_t(sim.dut->rdata_dut_o)),
                       int(int8_t(sim.dut->rdata_ref_o)));
            }
            mismatch_cnt++;
        }

        sim.tick();

        if (a_fire) ai++;
        if (b_fire) bi++;

        cycles++;
        if (cycles > max_cycles) {
            printf("  drive_stream: TIMEOUT after %d cycles (got %zu / %zu, "
                   "ai=%zu/%zu bi=%zu/%zu)\n",
                   cycles, got.size(), expected_out, ai, a.size(),
                   bi, b.size());
            sim.check(false, "drive_stream timeout");
            break;
        }
    }
    sim.dut->avalid_i = 0;
    sim.dut->bvalid_i = 0;
    sim.dut->rready_i = 0;
    sim.dut->adata_i  = 0;
    sim.dut->bdata_i  = 0;
    sim.dut->eval();
    return cycles;
}

static bool compare_streams(const std::vector<int8_t>& got,
                            const std::vector<int8_t>& exp,
                            SimCtrl<DUT>& sim,
                            const std::string& tag) {
    if (got.size() != exp.size()) {
        sim.check(false, tag + ": size mismatch got=" +
                  std::to_string(got.size()) + " exp=" +
                  std::to_string(exp.size()));
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
    sim.max_time = 200000000ull;

    printf("concat_mux test (Ca=%d Cb=%d Pixels=%d)\n", CA, CB, PIXELS);

    sim.dut->avalid_i = 0;
    sim.dut->bvalid_i = 0;
    sim.dut->adata_i  = 0;
    sim.dut->bdata_i  = 0;
    sim.dut->rready_i = 0;
    sim.dut->clr_i    = 0;
    sim.reset();

    // ── Test 1: reset state ────────────────────────────
    printf("test 1: reset state\n");
    sim.dut->eval();
    sim.check(sim.dut->rvalid_dut_o == 0, "rvalid_o=0 after reset (DUT)");
    sim.check(sim.dut->rvalid_ref_o == 0, "rvalid_o=0 after reset (REF)");
    sim.check(sim.dut->aready_dut_o == 0, "aready_o=0 with rready_i=0 (DUT)");
    sim.check(sim.dut->bready_dut_o == 0, "bready_o=0 after reset (DUT)");
    // With rready_i=1, A should be active first.
    sim.dut->rready_i = 1;
    sim.dut->eval();
    sim.check(sim.dut->aready_dut_o == 1, "aready_o=1 with rready_i=1 (DUT)");
    sim.check(sim.dut->bready_dut_o == 0, "bready_o=0 in phase A (DUT)");
    sim.check(sim.dut->mismatch_o   == 0, "no mismatch after reset");
    sim.dut->rready_i = 0;
    sim.dut->eval();

    // ── Test 2: directed Ca=2 Cb=3 Pixels=4 pattern ────
    // The directed pattern from the spec only makes sense when the
    // wrappers match; gate it by params and use generic data otherwise.
    printf("test 2: directed pattern, no back-pressure\n");
    {
        std::vector<int8_t> a, b;
        a.reserve(PIXELS * CA);
        b.reserve(PIXELS * CB);
        // A even sequence 2,4,6,8,...
        for (int i = 0; i < PIXELS * CA; i++) a.push_back(int8_t(2 * (i + 1)));
        // B odd sequence 1,3,5,...
        for (int i = 0; i < PIXELS * CB; i++) b.push_back(int8_t(2 * i + 1));
        auto exp = expected_concat(a, b);
        std::vector<int8_t> got;
        std::mt19937 rng(1);
        int mm = 0;
        drive_stream(sim, a, b, got, rng, 0, 0, 0, mm);
        sim.check(mm == 0, "directed: 0 mismatch_o cycles");
        compare_streams(got, exp, sim, "directed");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 3: random output back-pressure ────────────
    printf("test 3: random output back-pressure\n");
    {
        std::vector<int8_t> a, b;
        std::mt19937 rng(0xBEEF);
        std::uniform_int_distribution<int> d8(-128, 127);
        for (int i = 0; i < PIXELS * CA; i++) a.push_back(int8_t(d8(rng)));
        for (int i = 0; i < PIXELS * CB; i++) b.push_back(int8_t(d8(rng)));
        auto exp = expected_concat(a, b);
        std::vector<int8_t> got;
        int mm = 0;
        drive_stream(sim, a, b, got, rng, 0, 0, 50, mm);
        sim.check(mm == 0, "rgap: 0 mismatch_o cycles");
        compare_streams(got, exp, sim, "rgap");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 4: slow producer A ────────────────────────
    printf("test 4: slow producer on A\n");
    {
        std::vector<int8_t> a, b;
        std::mt19937 rng(0xC0DE);
        std::uniform_int_distribution<int> d8(-128, 127);
        for (int i = 0; i < PIXELS * CA; i++) a.push_back(int8_t(d8(rng)));
        for (int i = 0; i < PIXELS * CB; i++) b.push_back(int8_t(d8(rng)));
        auto exp = expected_concat(a, b);
        std::vector<int8_t> got;
        int mm = 0;
        drive_stream(sim, a, b, got, rng, 60, 0, 0, mm);
        sim.check(mm == 0, "agap: 0 mismatch_o cycles");
        compare_streams(got, exp, sim, "agap");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 5: slow producer B ────────────────────────
    printf("test 5: slow producer on B\n");
    {
        std::vector<int8_t> a, b;
        std::mt19937 rng(0xFACE);
        std::uniform_int_distribution<int> d8(-128, 127);
        for (int i = 0; i < PIXELS * CA; i++) a.push_back(int8_t(d8(rng)));
        for (int i = 0; i < PIXELS * CB; i++) b.push_back(int8_t(d8(rng)));
        auto exp = expected_concat(a, b);
        std::vector<int8_t> got;
        int mm = 0;
        drive_stream(sim, a, b, got, rng, 0, 60, 0, mm);
        sim.check(mm == 0, "bgap: 0 mismatch_o cycles");
        compare_streams(got, exp, sim, "bgap");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 6: full stress ────────────────────────────
    printf("test 6: random stress (all sides gapped)\n");
    {
        std::vector<int8_t> a, b;
        std::mt19937 rng(0xDEADBEEFu);
        std::uniform_int_distribution<int> d8(-128, 127);
        for (int i = 0; i < PIXELS * CA; i++) a.push_back(int8_t(d8(rng)));
        for (int i = 0; i < PIXELS * CB; i++) b.push_back(int8_t(d8(rng)));
        auto exp = expected_concat(a, b);
        std::vector<int8_t> got;
        int mm = 0;
        drive_stream(sim, a, b, got, rng, 30, 30, 30, mm);
        sim.check(mm == 0, "stress: 0 mismatch_o cycles");
        compare_streams(got, exp, sim, "stress");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 7: clr_i mid-stream ──────────────────────
    printf("test 7: clr_i mid-stream resets state\n");
    {
        // Push a couple A samples then clear.
        sim.dut->avalid_i = 1;
        sim.dut->adata_i  = 0x33;
        sim.dut->rready_i = 1;
        for (int i = 0; i < 2; i++) sim.tick();
        sim.dut->avalid_i = 0;
        sim.dut->rready_i = 0;

        sim.dut->clr_i = 1; sim.tick();
        sim.dut->clr_i = 0; sim.tick();
        sim.dut->eval();
        sim.check(sim.dut->rvalid_dut_o == 0, "post-clr: rvalid_o=0");
        sim.check(sim.dut->bready_dut_o == 0, "post-clr: bready_o=0");
        sim.check(sim.dut->mismatch_o   == 0, "post-clr: no mismatch");

        // Drive a fresh full frame.
        std::vector<int8_t> a, b;
        for (int i = 0; i < PIXELS * CA; i++) a.push_back(int8_t(i));
        for (int i = 0; i < PIXELS * CB; i++) b.push_back(int8_t(-i));
        auto exp = expected_concat(a, b);
        std::vector<int8_t> got;
        std::mt19937 rng(7);
        int mm = 0;
        drive_stream(sim, a, b, got, rng, 0, 0, 0, mm);
        sim.check(mm == 0, "post-clr: 0 mismatch");
        compare_streams(got, exp, sim, "post-clr");
    }

    return sim.finish();
}
