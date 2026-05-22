// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// linebuf_kxk — Verilator test.
//
// Drives DUT and SV REF in lockstep via a parameterized TB wrapper
// (linebuf_kxk_tb_{k3w4,k5w8,k3w8c16,k3w8c64} selected by -DTB_CFG_*).
//
// Per cycle we assert mismatch_o == 0, and on each output handshake we
// compare the DUT's emitted patch against an independent C++ shadow that
// reconstructs the K x K x Channels patch from the buffered input frame
// with explicit zero-padding.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(TB_CFG_K3W4)
  #include "Vlinebuf_kxk_tb_k3w4.h"
  using DUT = Vlinebuf_kxk_tb_k3w4;
  static constexpr int K = 3;
  static constexpr int W = 4;
  static constexpr int H = 4;
  static constexpr int Channels = 1;
#elif defined(TB_CFG_K5W8)
  #include "Vlinebuf_kxk_tb_k5w8.h"
  using DUT = Vlinebuf_kxk_tb_k5w8;
  static constexpr int K = 5;
  static constexpr int W = 8;
  static constexpr int H = 8;
  static constexpr int Channels = 1;
#elif defined(TB_CFG_K3W8C16)
  #include "Vlinebuf_kxk_tb_k3w8c16.h"
  using DUT = Vlinebuf_kxk_tb_k3w8c16;
  static constexpr int K = 3;
  static constexpr int W = 8;
  static constexpr int H = 8;
  static constexpr int Channels = 16;
#elif defined(TB_CFG_K3W8C64)
  #include "Vlinebuf_kxk_tb_k3w8c64.h"
  using DUT = Vlinebuf_kxk_tb_k3w8c64;
  static constexpr int K = 3;
  static constexpr int W = 8;
  static constexpr int H = 8;
  static constexpr int Channels = 64;
#else
  #error "Define one of TB_CFG_K3W4, TB_CFG_K5W8, TB_CFG_K3W8C16, TB_CFG_K3W8C64"
#endif

static constexpr int N        = K * K;
static constexpr int P        = (K - 1) / 2;
static constexpr int PATCH_EL = K * K * Channels;  // bytes per output beat

#include "sim_ctrl.h"

// Pack `Channels` int8 lanes (LSB-first, lane 0 in bits [7:0]) into the
// DUT's wdata_i port. `bus` is the Verilator-emitted member: a small
// uint scalar for narrow Channels and a VlWide array for wider ones.
// In either case Verilator lays the bytes out little-endian-by-byte
// within the underlying storage, so a raw byte-copy gives the right
// packing.
template <typename T>
static void pack_wdata(T& bus, const int8_t* lanes) {
    uint8_t* p = reinterpret_cast<uint8_t*>(&bus);
    // Storage may be larger than Channels bytes (rounded to word size);
    // zero the whole bus then drop bytes in.
    std::memset(&bus, 0, sizeof(bus));
    for (int c = 0; c < Channels; c++) {
        p[c] = static_cast<uint8_t>(lanes[c]);
    }
}

// Read byte i of the flat packed patch port. The same little-endian
// byte layout applies for both narrow (uint{8,16,32,64}_t) and wide
// (VlWide<N>) Verilator port types.
template <typename T>
static int8_t patch_byte(const T& bus, int i) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&bus);
    return static_cast<int8_t>(p[i]);
}

// C++ shadow: build the expected K*K*Channels patch for output position
// (orow, ocol) from a flat H*W*Channels input vector (channel-minor:
// in[(r*W + c)*Channels + ch]), with zero-padding.
static void shadow_patch(const std::vector<int8_t>& in, int orow, int ocol,
                         int8_t out[PATCH_EL]) {
    for (int ky = 0; ky < K; ky++) {
        for (int kx = 0; kx < K; kx++) {
            int sr = orow + ky - P;
            int sc = ocol + kx - P;
            bool in_frame = (sr >= 0 && sr < H && sc >= 0 && sc < W);
            for (int c = 0; c < Channels; c++) {
                int8_t v = 0;
                if (in_frame) v = in[(sr * W + sc) * Channels + c];
                out[(ky * K + kx) * Channels + c] = v;
            }
        }
    }
}

struct DriveResult {
    int  mismatch_cnt = 0;
    int  bad_patches  = 0;
    int  patches_seen = 0;
    int  cycles       = 0;
};

// Drive a frame of H*W*Channels input pixels into the DUT and drain H*W
// output patches. Stimulus is channel-minor in `in`.
static DriveResult drive_frame(SimCtrl<DUT>& sim,
                               const std::vector<int8_t>& in,
                               std::mt19937& rng,
                               int wgap_pct,
                               int rgap_pct,
                               const std::string& tag) {
    DriveResult r;
    std::uniform_int_distribution<int> d100(0, 99);

    int in_idx       = 0;   // pixel position (0 .. H*W)
    int out_idx      = 0;
    const int total  = H * W;
    int max_cycles   = total * 80 + 2000;

    int8_t zero_lanes[Channels];
    std::memset(zero_lanes, 0, sizeof(zero_lanes));

    sim.dut->wvalid_i = 0;
    sim.dut->rready_i = 0;
    pack_wdata(sim.dut->wdata_i, zero_lanes);

    while (out_idx < total) {
        bool want_w = (in_idx < total) && (d100(rng) >= wgap_pct);
        bool want_r = (d100(rng) >= rgap_pct);

        sim.dut->wvalid_i = want_w ? 1 : 0;
        if (want_w) {
            pack_wdata(sim.dut->wdata_i, &in[in_idx * Channels]);
        } else {
            pack_wdata(sim.dut->wdata_i, zero_lanes);
        }
        sim.dut->rready_i = want_r ? 1 : 0;

        sim.dut->eval();

        bool w_fire = want_w && (sim.dut->wready_dut_o != 0);
        bool r_fire = want_r && (sim.dut->rvalid_dut_o != 0);

        if (r_fire) {
            int orow = out_idx / W;
            int ocol = out_idx % W;
            int8_t exp[PATCH_EL];
            shadow_patch(in, orow, ocol, exp);
            for (int i = 0; i < PATCH_EL; i++) {
                int8_t got = patch_byte(sim.dut->rdata_dut_o, i);
                if (got != exp[i]) {
                    if (r.bad_patches < 4) {
                        int el = i / Channels;
                        int ch = i % Channels;
                        printf("  %s: patch diff @out=(%d,%d) el=%d ch=%d "
                               "(ky=%d kx=%d) got=%d exp=%d\n",
                               tag.c_str(), orow, ocol, el, ch,
                               el / K, el % K, int(got), int(exp[i]));
                    }
                    r.bad_patches++;
                }
            }
            r.patches_seen++;
            out_idx++;
        }

        if (sim.dut->mismatch_o) {
            if (r.mismatch_cnt < 4) {
                printf("  %s: mismatch @cy=%d w_dut=%d w_ref=%d r_v_dut=%d "
                       "r_v_ref=%d\n",
                       tag.c_str(), r.cycles,
                       int(sim.dut->wready_dut_o), int(sim.dut->wready_ref_o),
                       int(sim.dut->rvalid_dut_o), int(sim.dut->rvalid_ref_o));
            }
            r.mismatch_cnt++;
        }

        sim.tick();

        if (w_fire) in_idx++;

        r.cycles++;
        if (r.cycles > max_cycles) {
            printf("  %s: TIMEOUT after %d cycles (in=%d out=%d / %d)\n",
                   tag.c_str(), r.cycles, in_idx, out_idx, total);
            sim.check(false, tag + " timeout");
            break;
        }
    }

    sim.dut->wvalid_i = 0;
    sim.dut->rready_i = 0;
    pack_wdata(sim.dut->wdata_i, zero_lanes);
    sim.dut->eval();
    return r;
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 400000000ull;

    printf("linebuf_kxk test (K=%d W=%d H=%d Channels=%d)\n",
           K, W, H, Channels);

    sim.dut->wvalid_i = 0;
    {
        int8_t zero_lanes[Channels];
        std::memset(zero_lanes, 0, sizeof(zero_lanes));
        pack_wdata(sim.dut->wdata_i, zero_lanes);
    }
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
    sim.check(sim.dut->mismatch_o   == 0, "no mismatch after reset");

    // ── Test 2: directed pattern, no back-pressure ──
    printf("test 2: directed pattern, no back-pressure\n");
    {
        std::vector<int8_t> in;
        in.reserve(H * W * Channels);
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                for (int ch = 0; ch < Channels; ch++)
                    in.push_back(static_cast<int8_t>(
                        ((r * 16 + c) + ch * 3) & 0x7F));

        std::mt19937 rng(1);
        auto res = drive_frame(sim, in, rng, 0, 0, "directed");
        sim.check(res.mismatch_cnt == 0, "directed: 0 mismatch_o cycles");
        sim.check(res.bad_patches  == 0, "directed: all patches match shadow");
        sim.check(res.patches_seen == H * W,
                  "directed: emitted H*W patches");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 3: back-pressure on rready_i ──
    printf("test 3: output back-pressure\n");
    {
        std::vector<int8_t> in;
        in.reserve(H * W * Channels);
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                for (int ch = 0; ch < Channels; ch++)
                    in.push_back(static_cast<int8_t>(
                        (((r * 31 + c * 17 + ch * 13) % 251) - 125)));
        std::mt19937 rng(0xBEEF);
        auto res = drive_frame(sim, in, rng, 0, 50, "rgap");
        sim.check(res.mismatch_cnt == 0, "rgap: 0 mismatch_o cycles");
        sim.check(res.bad_patches  == 0, "rgap: all patches match shadow");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 4: slow producer (wvalid gaps) ──
    printf("test 4: slow producer\n");
    {
        std::vector<int8_t> in;
        in.reserve(H * W * Channels);
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                for (int ch = 0; ch < Channels; ch++)
                    in.push_back(static_cast<int8_t>((c - r + ch) * 3));
        std::mt19937 rng(0xC0DE);
        auto res = drive_frame(sim, in, rng, 50, 0, "wgap");
        sim.check(res.mismatch_cnt == 0, "wgap: 0 mismatch_o cycles");
        sim.check(res.bad_patches  == 0, "wgap: all patches match shadow");
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 5: random stress (both sides gapped, several frames) ──
    printf("test 5: random stress (3 frames, both sides gapped)\n");
    {
        std::mt19937 rng(0xDEADBEEFu);
        std::uniform_int_distribution<int> d8(-128, 127);
        for (int frame = 0; frame < 3; frame++) {
            std::vector<int8_t> in;
            in.reserve(H * W * Channels);
            for (int i = 0; i < H * W * Channels; i++) in.push_back(int8_t(d8(rng)));
            auto res = drive_frame(sim, in, rng, 35, 35,
                                   "stress_f" + std::to_string(frame));
            sim.check(res.mismatch_cnt == 0,
                      "stress: 0 mismatch_o cycles (frame " +
                      std::to_string(frame) + ")");
            sim.check(res.bad_patches  == 0,
                      "stress: all patches match shadow (frame " +
                      std::to_string(frame) + ")");
            sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();
        }
    }
    sim.dut->clr_i = 1; sim.tick(); sim.dut->clr_i = 0; sim.tick();

    // ── Test 6: clr_i mid-stream ──
    printf("test 6: clr_i mid-stream resets state\n");
    {
        sim.dut->wvalid_i = 1;
        sim.dut->rready_i = 1;
        {
            int8_t lanes[Channels];
            for (int c = 0; c < Channels; c++) lanes[c] = int8_t(0x55 + c);
            pack_wdata(sim.dut->wdata_i, lanes);
        }
        for (int i = 0; i < 5; i++) sim.tick();
        sim.dut->wvalid_i = 0;
        sim.dut->rready_i = 0;

        sim.dut->clr_i = 1;
        sim.tick();
        sim.dut->clr_i = 0;
        sim.tick();
        sim.dut->eval();
        sim.check(sim.dut->rvalid_dut_o == 0, "post-clr: rvalid_o=0");
        sim.check(sim.dut->wready_dut_o == 1, "post-clr: wready_o=1");
        sim.check(sim.dut->mismatch_o   == 0, "post-clr: no mismatch");

        std::vector<int8_t> in;
        in.reserve(H * W * Channels);
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                for (int ch = 0; ch < Channels; ch++)
                    in.push_back(int8_t(r * W + c + ch));
        std::mt19937 rng(7);
        auto res = drive_frame(sim, in, rng, 0, 0, "post-clr");
        sim.check(res.mismatch_cnt == 0, "post-clr: 0 mismatch_o cycles");
        sim.check(res.bad_patches  == 0, "post-clr: all patches match shadow");
    }

    return sim.finish();
}
