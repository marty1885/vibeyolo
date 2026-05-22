// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_12_cv1_test — Micro-integration test for YOLO26n /model.6/cv1.
// Loads ORT-derived stimulus + reference (../extract.py), runs layer_12_cv1.sv
// cycle-by-cycle (32 phases per output pixel), compares the int8 outputs
// (dequantised via S_OUT_SILU) to the ORT fp32 reference.
//   PASS: cosine ≥ 0.998 (random tiles).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

#include "Vlayer_12_cv1_tb.h"
#include "sim_ctrl.h"

using DUT = Vlayer_12_cv1_tb;

// Geometry — must match extract.py
static constexpr int NCH_IN   = 128;
static constexpr int NCH_OUT  = 128;
static constexpr int K        = 1;
static constexpr int STRIDE   = 1;
static constexpr int N_LANE   = 4;             // P_CIN
static constexpr int N_PHASE  = NCH_IN / N_LANE;  // 32
static constexpr int ROI_H    = 8;
static constexpr int ROI_W    = 8;
static constexpr int OUT_H    = ROI_H;
static constexpr int OUT_W    = ROI_W;
static constexpr double S_OUT_SILU = 4.0 / 127.0;

static constexpr int DOT_LAT     = 3;   // 1 + clog2(4)
static constexpr int ACC_LAT     = 1;   // commit reg
static constexpr int REQUANT_LAT = 3;
static constexpr int SILU_LAT    = 1;
static constexpr int SILU_OUT_REG_LAT = 1;
// Latency from phase=(N_PHASE-1) valid_i to valid_o (in cycles):
static constexpr int TOTAL_LAT_FROM_LAST_PH =
    DOT_LAT + ACC_LAT + REQUANT_LAT + SILU_LAT + SILU_OUT_REG_LAT;

static std::string stim_dir() { return std::string("../stim"); }

static std::vector<uint32_t> load_hex(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "could not open %s\n", path.c_str()); std::exit(2); }
    std::vector<uint32_t> v;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        v.push_back((uint32_t)std::stoul(line, nullptr, 16));
    }
    return v;
}

static float u32_to_f32(uint32_t bits) {
    float f; std::memcpy(&f, &bits, 4); return f;
}

struct Sample {
    std::string name;
    std::vector<int8_t>   input_i8;
    std::vector<uint16_t> scale_fp16;
    std::vector<uint16_t> bias_fp16;
    std::vector<float>    ref_ort;
    std::vector<float>    ref_hw;
};

static Sample load_sample(const std::string& name) {
    Sample s; s.name = name;
    auto inp = load_hex(stim_dir() + "/" + name + ".input_i8.hex");
    s.input_i8.resize(inp.size());
    for (size_t i = 0; i < inp.size(); i++)
        s.input_i8[i] = (int8_t)(uint8_t)(inp[i] & 0xFF);
    for (auto v : load_hex(stim_dir() + "/" + name + ".scale_fp16.hex"))
        s.scale_fp16.push_back((uint16_t)v);
    for (auto v : load_hex(stim_dir() + "/" + name + ".bias_fp16.hex"))
        s.bias_fp16.push_back((uint16_t)v);
    for (auto v : load_hex(stim_dir() + "/" + name + ".ref_ort.f32.hex"))
        s.ref_ort.push_back(u32_to_f32(v));
    for (auto v : load_hex(stim_dir() + "/" + name + ".ref_hw.f32.hex"))
        s.ref_hw.push_back(u32_to_f32(v));
    return s;
}

template <typename T>
static void pack_bytes(T& dst, const int8_t* src, int nbytes) {
    auto* p = reinterpret_cast<uint32_t*>(&dst);
    int nwords = (nbytes + 3) / 4;
    for (int i = 0; i < nwords; i++) p[i] = 0;
    for (int i = 0; i < nbytes; i++) {
        uint32_t b = (uint8_t)src[i];
        p[i / 4] |= (b << ((i % 4) * 8));
    }
}

template <typename T>
static void pack_u16(T& dst, const uint16_t* src, int nshorts) {
    auto* p = reinterpret_cast<uint32_t*>(&dst);
    int nwords = (nshorts + 1) / 2;
    for (int i = 0; i < nwords; i++) p[i] = 0;
    for (int i = 0; i < nshorts; i++) {
        uint32_t v = (uint32_t)src[i];
        p[i / 2] |= (v << ((i % 2) * 16));
    }
}

template <typename T>
static void unpack_bytes(const T& src, int8_t* dst, int nbytes) {
    auto* p = reinterpret_cast<const uint32_t*>(&src);
    for (int i = 0; i < nbytes; i++) {
        dst[i] = (int8_t)((p[i / 4] >> ((i % 4) * 8)) & 0xFFu);
    }
}

// Build per-cycle phase window for (oy, ox, phase).
static void build_phase_window(const Sample& s, int oy, int ox, int phase,
                               int8_t out[N_LANE]) {
    for (int k = 0; k < N_LANE; k++) {
        int kc = phase * N_LANE + k;
        int in_off = (oy * ROI_W + ox) * NCH_IN + kc;
        out[k] = s.input_i8[in_off];
    }
}

// Build (NCH_OUT × N_LANE) weight slab for a given phase.
static void build_w_phase(const std::vector<int8_t>& w_all, int phase,
                          int8_t out[NCH_OUT * N_LANE]) {
    // w_all order: (c, phase, k)
    for (int c = 0; c < NCH_OUT; c++) {
        for (int k = 0; k < N_LANE; k++) {
            int idx = (c * N_PHASE + phase) * N_LANE + k;
            out[c * N_LANE + k] = w_all[idx];
        }
    }
}

struct Stats {
    double max_abs = 0.0, mae = 0.0, cos = 0.0, out_range = 0.0;
    int n = 0, worst_idx = -1;
    float worst_dut = 0.0f, worst_ref = 0.0f;
};

static Stats compute_stats(const std::vector<float>& dut,
                           const std::vector<float>& ref) {
    Stats st;
    st.n = (int)dut.size();
    double sum_abs = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    float mn = 1e30f, mx = -1e30f;
    for (int i = 0; i < st.n; i++) {
        double e = std::abs((double)dut[i] - (double)ref[i]);
        sum_abs += e;
        if (e > st.max_abs) {
            st.max_abs = e; st.worst_idx = i;
            st.worst_dut = dut[i]; st.worst_ref = ref[i];
        }
        dot += (double)dut[i] * (double)ref[i];
        na  += (double)dut[i] * (double)dut[i];
        nb  += (double)ref[i] * (double)ref[i];
        if (ref[i] < mn) mn = ref[i];
        if (ref[i] > mx) mx = ref[i];
    }
    st.mae = sum_abs / std::max(1, st.n);
    st.cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    st.out_range = (double)(mx - mn);
    return st;
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 2000000;

    auto w_raw = load_hex(stim_dir() + "/weights.i8.hex");
    if ((int)w_raw.size() != NCH_OUT * N_PHASE * N_LANE) {
        fprintf(stderr, "weights size mismatch: got %zu expected %d\n",
                w_raw.size(), NCH_OUT * N_PHASE * N_LANE);
        return 2;
    }
    std::vector<int8_t> w_bytes(NCH_OUT * N_PHASE * N_LANE);
    for (size_t i = 0; i < w_raw.size(); i++)
        w_bytes[i] = (int8_t)(uint8_t)(w_raw[i] & 0xFF);

    // Idle defaults
    sim.dut->valid_i = 0;
    sim.dut->phase_i = 0;
    std::vector<int8_t> zero_lanes(N_LANE, 0);
    pack_bytes(sim.dut->x_flat_i, zero_lanes.data(), N_LANE);
    std::vector<int8_t> zero_w(NCH_OUT * N_LANE, 0);
    pack_bytes(sim.dut->w_flat_i, zero_w.data(), NCH_OUT * N_LANE);
    std::vector<uint16_t> zero_u16(NCH_OUT, 0);
    pack_u16(sim.dut->scale_flat_i, zero_u16.data(), NCH_OUT);
    pack_u16(sim.dut->bias_flat_i,  zero_u16.data(), NCH_OUT);
    sim.reset();

    struct SampleEntry { std::string name; bool required; };
    const std::vector<SampleEntry> sample_names = {
        {"rand0",     true },
        {"rand1",     true },
        {"rand2",     true },
        {"rand3",     true },
        {"rand_low",  true },
        {"rand_high", true },
        {"half",      false},
        {"gradient",  false},
    };

    int n_pass = 0, n_fail = 0;
    double agg_cos = 0.0, agg_mae = 0.0, agg_max = 0.0;
    int cycle_count_per_frame = -1;

    // Pre-pack per-phase weight slabs (constant across the ROI).
    std::vector<std::vector<int8_t>> w_ph(N_PHASE,
        std::vector<int8_t>(NCH_OUT * N_LANE));
    for (int p = 0; p < N_PHASE; p++)
        build_w_phase(w_bytes, p, w_ph[p].data());

    for (const auto& se : sample_names) {
        const std::string& sname = se.name;
        Sample s = load_sample(sname);
        pack_u16(sim.dut->scale_flat_i, s.scale_fp16.data(), NCH_OUT);
        pack_u16(sim.dut->bias_flat_i,  s.bias_fp16.data(),  NCH_OUT);

        const int N_OUT = OUT_H * OUT_W;
        const int N_PHASES_TOT = N_OUT * N_PHASE;

        std::vector<int8_t> y_i8(N_OUT * NCH_OUT, 0);
        int produced = 0;
        std::vector<int> inflight;
        inflight.reserve(N_OUT + TOTAL_LAT_FROM_LAST_PH);

        int cycles_used = 0;

        for (int step = 0; step < N_PHASES_TOT + TOTAL_LAT_FROM_LAST_PH + 8; step++) {
            if (step < N_PHASES_TOT) {
                int pix = step / N_PHASE;
                int ph  = step % N_PHASE;
                int oy  = pix / OUT_W;
                int ox  = pix % OUT_W;
                int8_t win[N_LANE];
                build_phase_window(s, oy, ox, ph, win);
                pack_bytes(sim.dut->x_flat_i, win, N_LANE);
                pack_bytes(sim.dut->w_flat_i,
                           w_ph[ph].data(),
                           NCH_OUT * N_LANE);
                sim.dut->valid_i = 1;
                sim.dut->phase_i = (uint8_t)ph;
                if (ph == N_PHASE - 1) inflight.push_back(pix);
                cycles_used++;
            } else {
                sim.dut->valid_i = 0;
                sim.dut->phase_i = 0;
                pack_bytes(sim.dut->x_flat_i, zero_lanes.data(), N_LANE);
            }
            sim.tick();
            if (sim.dut->valid_o && produced < (int)inflight.size()) {
                int pix = inflight[produced];
                int8_t row[NCH_OUT];
                unpack_bytes(sim.dut->y_flat_o, row, NCH_OUT);
                for (int c = 0; c < NCH_OUT; c++) y_i8[pix * NCH_OUT + c] = row[c];
                produced++;
            }
        }

        if (cycle_count_per_frame < 0) cycle_count_per_frame = cycles_used;

        std::vector<float> dut_f(N_OUT * NCH_OUT);
        for (int i = 0; i < N_OUT * NCH_OUT; i++)
            dut_f[i] = (float)((double)y_i8[i] * S_OUT_SILU);

        Stats st_ort = compute_stats(dut_f, s.ref_ort);
        Stats st_hw  = compute_stats(dut_f, s.ref_hw);
        double cos_thresh = se.required ? 0.998 : 0.99;
        bool pass = (st_ort.cos >= cos_thresh);

        printf("\n──── sample %s%s ────\n", sname.c_str(), se.required ? "" : " (informational)");
        printf("  DUT vs ORT  : max_abs=%.4f  mae=%.4f  cos=%.6f  out_range=%.3f  cos_thresh=%.4f\n",
               st_ort.max_abs, st_ort.mae, st_ort.cos, st_ort.out_range, cos_thresh);
        printf("  DUT vs HW-ref: max_abs=%.4f mae=%.4f  cos=%.6f\n",
               st_hw.max_abs, st_hw.mae, st_hw.cos);
        if (st_ort.worst_idx >= 0) {
            int i = st_ort.worst_idx;
            int c = i % NCH_OUT, ox = (i / NCH_OUT) % OUT_W, oy = (i / NCH_OUT) / OUT_W;
            printf("  worst-px: (oy=%d ox=%d c=%d) dut=%.4f ort=%.4f hw=%.4f\n",
                   oy, ox, c, st_ort.worst_dut, st_ort.worst_ref, s.ref_hw[i]);
        }
        printf("  produced=%d expected=%d\n", produced, N_OUT);
        printf("  → %s\n", pass ? "PASS" : "FAIL");

        sim.check(pass, std::string("sample ") + sname + " pass");
        if (pass) n_pass++; else n_fail++;
        agg_cos += st_ort.cos;
        agg_mae += st_ort.mae;
        agg_max  = std::max(agg_max, st_ort.max_abs);
    }

    // Cycle / MAC accounting for full-frame
    const int FRAME_H = 40, FRAME_W = 40;
    const int FRAME_PIX = FRAME_H * FRAME_W;
    const int CYCLES_PER_PIX = N_PHASE;                  // 32
    const int FRAME_CYCLES = FRAME_PIX * CYCLES_PER_PIX; // 51200
    const int MACS_PER_FRAME = FRAME_PIX * NCH_OUT * NCH_IN; // 1600*128*128
    const int MAC_UNITS = NCH_OUT * N_LANE;              // 128 * 4 = 512

    printf("\n========================================\n");
    printf("Aggregate: %d/%zu samples passed\n", n_pass, sample_names.size());
    printf("  avg cos     = %.6f\n", agg_cos / sample_names.size());
    printf("  avg mae     = %.6f\n", agg_mae / sample_names.size());
    printf("  worst max   = %.6f\n", agg_max);
    printf("Cycle accounting (layer_12_cv1, 1x1 conv 128→128 @ 40x40):\n");
    printf("  MAC units            : %d  (= NCH_OUT(%d) * N_LANE(%d))\n",
           MAC_UNITS, NCH_OUT, N_LANE);
    printf("  cycles / output pix  : %d  (= N_PHASE = NCH_IN/N_LANE = %d/%d)\n",
           CYCLES_PER_PIX, NCH_IN, N_LANE);
    printf("  pipeline latency     : %d cycles\n", TOTAL_LAT_FROM_LAST_PH);
    printf("  ROI %dx%d (%d pix) drive cycles : %d\n",
           OUT_H, OUT_W, OUT_H*OUT_W, cycle_count_per_frame);
    printf("Full frame:\n");
    printf("  output pixels        : %d\n", FRAME_PIX);
    printf("  total MACs           : %d\n", MACS_PER_FRAME);
    printf("  frame cycles         : %d  (matches LAYER_12_CYCLES = 51200, target T_FRAME=100000)\n",
           FRAME_CYCLES);
    printf("========================================\n");

    return sim.finish();
}
