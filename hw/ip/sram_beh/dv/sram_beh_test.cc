// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// sram_beh — Verilator test. Drives sram_beh_banked (DUT) and a flat magic-
// array REF in lockstep through sram_beh_tb, asserting mismatch_o == 0 every
// cycle. Two configs exercise read latency 1 and 2; both shapes force banking
// in BOTH dimensions (NW>1 and ND>1) so the depth-bank decode, width-slice
// stitching, and latency-aligned output mux are all covered.
//
// Reads are issued continuously (r_en=1) — the streaming use case — so the
// comparison covers banked read-before-write and latency alignment. Writes are
// random (addr, data, enable). LWidth here is small enough to fit a uint64_t
// per word for the C++ side; the TB compares full width in RTL.

#include <cstdint>
#include <cstdio>
#include <random>

#ifndef TB_CONFIG
#define TB_CONFIG 1
#endif

#if TB_CONFIG == 1
  #include "Vsram_beh_tb_lat1.h"
  using DUT = Vsram_beh_tb_lat1;
#elif TB_CONFIG == 2
  #include "Vsram_beh_tb_lat2.h"
  using DUT = Vsram_beh_tb_lat2;
#else
  #error "Unsupported TB_CONFIG"
#endif

static constexpr int LDEPTH = 96;   // matches sram_beh_tb default (ND=3 @ MaxDepth=32)

#include "sim_ctrl.h"

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    auto* d = sim.dut.get();

    std::mt19937 rng(0x5A3D);
    auto rnd = [&](uint64_t hi) { return rng() % hi; };

    sim.reset();
    d->r_en_i = 1;   // continuous read

    const int N = 50000;
    int writes = 0, mism = 0;
    for (int i = 0; i < N; i++) {
        d->w_en_i   = rnd(2);
        d->w_addr_i = rnd(LDEPTH);
        // 256-bit w_data is a packed array in Verilator; fill via the public
        // VlWide. Drive a deterministic-but-varied pattern per word.
        for (int k = 0; k < 8; k++)
            d->w_data_i[k] = (uint32_t)(rng());
        d->r_addr_i = rnd(LDEPTH);

        sim.tick();
        if (d->w_en_i) writes++;
        if (d->mismatch_o) mism++;
        sim.check(!d->mismatch_o, "DUT banked SRAM != flat REF");
    }

    printf("[sram_beh cfg=%d] %d cycles, %d writes, %d mismatches\n",
           TB_CONFIG, N, writes, mism);
    return sim.finish();
}
