// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// cs_frame_test — stream a real int8 frame through one conv_stage and dump the
// raster int8 output. Dimensions + file paths come from argv (the Verilated
// model is rebuilt per layer with matching -G params), so this driver is
// dimension-agnostic.
//
//   Vcs_frame_tb CIN COUT K STRIDE PAD H_IN W_IN <input_i8.hex> <out_i8.hex>

#include "Vcs_frame_tb.h"
#include "sim_ctrl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using DUT = Vcs_frame_tb;

static std::vector<int8_t> load_i8(const std::string& p) {
    std::ifstream f(p);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(2); }
    std::vector<int8_t> v; std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        v.push_back((int8_t)(uint8_t)(std::stoul(line, nullptr, 16) & 0xFF));
    }
    return v;
}

// Pack/unpack the flattened CIN*8 / COUT*8 ports (VlWide = uint32_t array).
template <typename T>
static void pack(T& dst, const int8_t* src, int n) {
    auto* p = reinterpret_cast<uint32_t*>(&dst);
    for (int i = 0; i < (n + 3) / 4; i++) p[i] = 0;
    for (int i = 0; i < n; i++)
        p[i / 4] |= ((uint32_t)(uint8_t)src[i]) << ((i % 4) * 8);
}
template <typename T>
static void unpack(const T& src, int8_t* dst, int n) {
    auto* p = reinterpret_cast<const uint32_t*>(&src);
    for (int i = 0; i < n; i++)
        dst[i] = (int8_t)((p[i / 4] >> ((i % 4) * 8)) & 0xFF);
}

int main(int argc, char** argv) {
    if (argc < 10) { fprintf(stderr, "need 9 args\n"); return 2; }
    const int CIN = atoi(argv[1]), COUT = atoi(argv[2]), K = atoi(argv[3]);
    const int STRIDE = atoi(argv[4]), PAD = atoi(argv[5]);
    const int H_IN = atoi(argv[6]), W_IN = atoi(argv[7]);
    const std::string in_path = argv[8], out_path = argv[9];
    const int H_OUT = (H_IN + 2 * PAD - K) / STRIDE + 1;
    const int W_OUT = (W_IN + 2 * PAD - K) / STRIDE + 1;
    const int N_IN = H_IN * W_IN, N_OUT = H_OUT * W_OUT;

    auto frame = load_i8(in_path);
    if ((int)frame.size() != N_IN * CIN) {
        fprintf(stderr, "frame size %zu != %d\n", frame.size(), N_IN * CIN);
        return 2;
    }

    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = (uint64_t)(N_IN + N_OUT) * 64ull * (COUT / 4 + 4) + 200000;
    sim.dut->oready_i = 1;
    sim.dut->ivalid_i = 0;
    sim.dut->start_i = 0;
    sim.reset();

    sim.dut->start_i = 1; sim.tick(); sim.dut->start_i = 0;

    std::vector<int8_t> out(N_OUT * COUT, 0);
    int in_idx = 0, out_idx = 0;
    std::vector<int8_t> pix(CIN), opix(COUT);

    uint64_t guard = 0, guard_max = sim.max_time;
    while (out_idx < N_OUT && guard++ < guard_max) {
        // present next input pixel (if any)
        if (in_idx < N_IN) {
            for (int c = 0; c < CIN; c++) pix[c] = frame[in_idx * CIN + c];
            pack(sim.dut->idata_flat_i, pix.data(), CIN);
            sim.dut->ivalid_i = 1;
        } else {
            sim.dut->ivalid_i = 0;
        }
        sim.dut->oready_i = 1;
        sim.dut->eval();
        // sample handshakes on this cycle
        bool acc_in  = sim.dut->ivalid_i && sim.dut->iready_o;
        bool acc_out = sim.dut->ovalid_o && sim.dut->oready_i;
        if (acc_out && out_idx < N_OUT) {
            unpack(sim.dut->odata_flat_o, opix.data(), COUT);
            for (int c = 0; c < COUT; c++) out[out_idx * COUT + c] = opix[c];
            out_idx++;
        }
        sim.tick();
        if (acc_in) in_idx++;
    }

    if (out_idx != N_OUT) {
        fprintf(stderr, "TIMEOUT: collected %d/%d output pixels (in %d/%d)\n",
                out_idx, N_OUT, in_idx, N_IN);
        return 1;
    }

    std::ofstream of(out_path);
    for (auto b : out) of << std::hex << ((unsigned)(uint8_t)b & 0xFF) << "\n";
    printf("OK: streamed %d in, collected %d out -> %s\n", in_idx, out_idx,
           out_path.c_str());
    return 0;
}
