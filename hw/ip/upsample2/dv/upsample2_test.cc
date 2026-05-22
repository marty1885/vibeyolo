// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// upsample2 — Verilator test.
//
// Drives the DUT and the SV REF in lockstep via a parameterized TB wrapper
// (upsample2_tb_w4 for W=4 or upsample2_tb_w8 for W=8 — selected via
// -DTB_WIDTH=4|8 at compile time). For every cycle we assert:
//
//   * `mismatch_o` (sampled DUT vs REF) is 0
//   * the DUT's emitted output stream, drained on every output handshake,
//     matches an independent C++ nearest-neighbor-2x shadow of the entire
//     input sequence.
//
// The C++ shadow guards against the case where DUT and REF share the same
// bug.

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#ifndef TB_WIDTH
#define TB_WIDTH 4
#endif

#if TB_WIDTH == 4
  #include "Vupsample2_tb_w4.h"
  using DUT = Vupsample2_tb_w4;
  static constexpr int W = 4;
#elif TB_WIDTH == 8
  #include "Vupsample2_tb_w8.h"
  using DUT = Vupsample2_tb_w8;
  static constexpr int W = 8;
#else
  #error "Unsupported TB_WIDTH"
#endif

#include "sim_ctrl.h"

// Compute the expected nearest-neighbor 2x output stream for a sequence of
// `nrows` input rows of width W.  Each input pixel (r,c) is emitted into
// the four output positions (2r,2c), (2r,2c+1), (2r+1,2c), (2r+1,2c+1) in
// raster order across the (2*nrows) x (2*W) output image.
static std::vector<int8_t> nn2x_expected(const std::vector<int8_t>& in,
                                         int nrows) {
    std::vector<int8_t> out;
    out.reserve(in.size() * 4);
    for (int r = 0; r < nrows; r++) {
        for (int dup_row = 0; dup_row < 2; dup_row++) {
            for (int c = 0; c < W; c++) {
                int8_t p = in[r * W + c];
                out.push_back(p);
                out.push_back(p);
            }
        }
    }
    return out;
}

// Drive a sequence of input pixels into the DUT and drain its output
// stream into `got`, applying randomized wvalid / rready hold patterns
// (configured by `wgap_pct` and `rgap_pct` — probability of stalling).
// Returns total cycle count.  Asserts mismatch_o == 0 on every cycle.
static int drive_stream(SimCtrl<DUT>& sim,
                        const std::vector<int8_t>& in,
                        std::vector<int8_t>& got,
                        std::mt19937& rng,
                        int wgap_pct,
                        int rgap_pct,
                        int& mismatch_cnt) {
    std::uniform_int_distribution<int> d100(0, 99);

    size_t in_idx = 0;
    size_t expected_out = in.size() * 4;
    int cycles = 0;
    int max_cycles = static_cast<int>(expected_out) * 50 + 1000;

    // Pre-set rready_i / wvalid_i to 0 to start clean.
    sim.dut->wvalid_i = 0;
    sim.dut->rready_i = 0;
    sim.dut->wdata_i  = 0;

    while (got.size() < expected_out) {
        // Decide wvalid_i this cycle.
        bool want_w = (in_idx < in.size()) && (d100(rng) >= wgap_pct);
        bool want_r = (d100(rng) >= rgap_pct);

        sim.dut->wvalid_i = want_w ? 1 : 0;
        sim.dut->wdata_i  = want_w ? static_cast<uint8_t>(in[in_idx]) : 0;
        sim.dut->rready_i = want_r ? 1 : 0;

        // Evaluate combinational outputs at this set of inputs before tick.
        sim.dut->eval();

        bool w_fire = want_w && (sim.dut->wready_dut_o != 0);
        bool r_fire = want_r && (sim.dut->rvalid_dut_o != 0);

        if (r_fire) {
            got.push_back(static_cast<int8_t>(sim.dut->rdata_dut_o));
        }

        if (sim.dut->mismatch_o) {
            if (mismatch_cnt < 4) {
                printf("  mismatch @cy=%d: w_dut=%d w_ref=%d r_v_dut=%d "
                       "r_v_ref=%d d_dut=%d d_ref=%d\n",
                       cycles,
                       int(sim.dut->wready_dut_o), int(sim.dut->wready_ref_o),
                       int(sim.dut->rvalid_dut_o), int(sim.dut->rvalid_ref_o),
                       int(int8_t(sim.dut->rdata_dut_o)),
                       int(int8_t(sim.dut->rdata_ref_o)));
            }
            mismatch_cnt++;
        }

        sim.tick();

        if (w_fire) {
            in_idx++;
        }

        cycles++;
        if (cycles > max_cycles) {
            printf("  drive_stream: TIMEOUT after %d cycles (got %zu / %zu)\n",
                   cycles, got.size(), expected_out);
            sim.check(false, "drive_stream timeout");
            break;
        }
    }
    // Quiesce inputs.
    sim.dut->wvalid_i = 0;
    sim.dut->rready_i = 0;
    sim.dut->wdata_i  = 0;
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

    printf("upsample2 test (W=%d)\n", W);

    // ── Init ────────────────────────────────────────────
    sim.dut->wvalid_i = 0;
    sim.dut->wdata_i  = 0;
    sim.dut->rready_i = 0;
    sim.dut->clr_i    = 0;
    sim.reset();

    // ── Test 1: reset behavior ──────────────────────────
    printf("test 1: reset state\n");
    sim.dut->eval();
    sim.check(sim.dut->rvalid_dut_o == 0, "rvalid_o=0 after reset (DUT)");
    sim.check(sim.dut->rvalid_ref_o == 0, "rvalid_o=0 after reset (REF)");
    sim.check(sim.dut->wready_dut_o == 1, "wready_o=1 after reset (DUT)");
    sim.check(sim.dut->wready_ref_o == 1, "wready_o=1 after reset (REF)");
    sim.check(sim.dut->mismatch_o == 0, "no mismatch after reset");

    // ── Test 2: directed WxW pattern with no back-pressure ──
    printf("test 2: directed pattern, no back-pressure\n");
    {
        std::vector<int8_t> in;
        in.reserve(W * W);
        for (int r = 0; r < W; r++) {
            for (int c = 0; c < W; c++) {
                in.push_back(static_cast<int8_t>((r * W + c) - 64));
            }
        }
        auto exp = nn2x_expected(in, W);
        std::vector<int8_t> got;
        std::mt19937 rng(1);
        int mm = 0;
        drive_stream(sim, in, got, rng, /*wgap*/0, /*rgap*/0, mm);
        sim.check(mm == 0, "directed: 0 mismatch_o cycles");
        compare_streams(got, exp, sim, "directed");
    }

    // Re-clear between tests via clr_i pulse to drop any residual state.
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 3: random back-pressure on rready_i ────────
    printf("test 3: random output back-pressure\n");
    {
        std::vector<int8_t> in;
        for (int r = 0; r < W; r++)
            for (int c = 0; c < W; c++)
                in.push_back(static_cast<int8_t>(((r * 31 + c * 17) % 251) - 125));
        auto exp = nn2x_expected(in, W);
        std::vector<int8_t> got;
        std::mt19937 rng(0xBEEF);
        int mm = 0;
        drive_stream(sim, in, got, rng, /*wgap*/0, /*rgap*/40, mm);
        sim.check(mm == 0, "rgap: 0 mismatch_o cycles");
        compare_streams(got, exp, sim, "rgap");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 4: slow producer (random wvalid gaps) ──────
    printf("test 4: slow producer\n");
    {
        std::vector<int8_t> in;
        for (int r = 0; r < W; r++)
            for (int c = 0; c < W; c++)
                in.push_back(static_cast<int8_t>((c - r) * 3));
        auto exp = nn2x_expected(in, W);
        std::vector<int8_t> got;
        std::mt19937 rng(0xC0DE);
        int mm = 0;
        drive_stream(sim, in, got, rng, /*wgap*/40, /*rgap*/0, mm);
        sim.check(mm == 0, "wgap: 0 mismatch_o cycles");
        compare_streams(got, exp, sim, "wgap");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 5: full stress (random producer + consumer) ──
    printf("test 5: random stress (W=%d, 16 rows, random both sides)\n", W);
    {
        const int NROWS = 16;
        std::vector<int8_t> in;
        std::mt19937 rng(0xDEADBEEFu);
        std::uniform_int_distribution<int> d8(-128, 127);
        for (int i = 0; i < NROWS * W; i++) in.push_back(int8_t(d8(rng)));
        auto exp = nn2x_expected(in, NROWS);
        std::vector<int8_t> got;
        int mm = 0;
        drive_stream(sim, in, got, rng, /*wgap*/30, /*rgap*/30, mm);
        sim.check(mm == 0, "stress: 0 mismatch_o cycles");
        compare_streams(got, exp, sim, "stress");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 6: clr_i mid-stream ────────────────────────
    printf("test 6: clr_i mid-stream resets state\n");
    {
        // Push a partial input row, then clr_i, then a full new stream.
        sim.dut->wvalid_i = 1;
        sim.dut->rready_i = 1;
        sim.dut->wdata_i  = 0x55;
        for (int i = 0; i < 3; i++) sim.tick();
        sim.dut->wvalid_i = 0;
        sim.dut->rready_i = 0;

        // Clear.
        sim.dut->clr_i = 1;
        sim.tick();
        sim.dut->clr_i = 0;
        sim.tick();
        sim.dut->eval();
        sim.check(sim.dut->rvalid_dut_o == 0, "post-clr: rvalid_o=0");
        sim.check(sim.dut->wready_dut_o == 1, "post-clr: wready_o=1");
        sim.check(sim.dut->mismatch_o   == 0, "post-clr: no mismatch");

        // Now drive a fresh row to confirm we restart cleanly.
        std::vector<int8_t> in;
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < W; c++)
                in.push_back(int8_t(r * W + c));
        auto exp = nn2x_expected(in, 2);
        std::vector<int8_t> got;
        std::mt19937 rng(7);
        int mm = 0;
        drive_stream(sim, in, got, rng, 0, 0, mm);
        sim.check(mm == 0, "post-clr stream: 0 mismatch");
        compare_streams(got, exp, sim, "post-clr");
    }

    return sim.finish();
}
