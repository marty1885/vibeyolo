// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_3_m0_cv1_test — Micro-integration test for YOLO26n /model.2/m.0/cv1.
// Loads ORT-derived stimulus + reference (../golden.py), runs the DUT
// cycle-by-cycle, compares int8 outputs (dequantised via S_OUT_SILU) to ORT.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

#include "Vlayer_3_m0_cv1_tb.h"
#include "sim_ctrl.h"

using DUT = Vlayer_3_m0_cv1_tb;

// Geometry — must match golden.py
static constexpr int NCH_IN   = 16;
static constexpr int NCH_OUT  = 8;
static constexpr int K        = 3;
static constexpr int STRIDE   = 1;
static constexpr int N_LANE   = K * K * NCH_IN;   // 144
static constexpr int ROI_H    = 8;
static constexpr int ROI_W    = 8;
static constexpr int PAD_H    = ROI_H + 2;        // 10
static constexpr int PAD_W    = ROI_W + 2;        // 10
static constexpr int OUT_H    = ROI_H;
static constexpr int OUT_W    = ROI_W;
static constexpr double S_OUT_SILU = 2.0 / 127.0;

static constexpr int DOT_LAT     = 9;   // 1 + clog2(144)
static constexpr int REQUANT_LAT = 3;
static constexpr int SILU_LAT    = 1;
static constexpr int TOTAL_LAT   = DOT_LAT + REQUANT_LAT + SILU_LAT;  // 13

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
    std::vector<int8_t>   input_i8;   // PAD_H*PAD_W*NCH_IN, order h,w,kc
    std::vector<uint16_t> scale_fp16; // NCH_OUT
    std::vector<uint16_t> bias_fp16;  // NCH_OUT
    std::vector<float>    ref_ort;    // OUT_H*OUT_W*NCH_OUT, order oy,ox,c
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

// Build the per-cycle window. Lane index = (kh*K + kw)*NCH_IN + kc.
static void build_window(const Sample& s, int oy, int ox, int8_t out[N_LANE]) {
    for (int kh = 0; kh < K; kh++) {
        for (int kw = 0; kw < K; kw++) {
            for (int kc = 0; kc < NCH_IN; kc++) {
                int h_idx = oy * STRIDE + kh;
                int w_idx = ox * STRIDE + kw;
                int in_off = (h_idx * PAD_W + w_idx) * NCH_IN + kc;
                int lane   = (kh * K + kw) * NCH_IN + kc;
                out[lane] = s.input_i8[in_off];
            }
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
    sim.max_time = 500000;

    auto w_raw = load_hex(stim_dir() + "/weights.i8.hex");
    if ((int)w_raw.size() != NCH_OUT * N_LANE) {
        fprintf(stderr, "weights size mismatch: got %zu expected %d\n",
                w_raw.size(), NCH_OUT * N_LANE);
        return 2;
    }
    std::vector<int8_t> w_bytes(NCH_OUT * N_LANE);
    for (int c = 0; c < NCH_OUT; c++)
        for (int k = 0; k < N_LANE; k++)
            w_bytes[c * N_LANE + k] = (int8_t)(uint8_t)(w_raw[c * N_LANE + k] & 0xFF);

    pack_bytes(sim.dut->w_flat_i, w_bytes.data(), NCH_OUT * N_LANE);
    sim.dut->valid_i = 0;
    std::vector<int8_t> zero_lanes(N_LANE, 0);
    pack_bytes(sim.dut->x_flat_i, zero_lanes.data(), N_LANE);
    std::vector<uint16_t> zero_u16(NCH_OUT, 0);
    pack_u16(sim.dut->scale_flat_i, zero_u16.data(), NCH_OUT);
    pack_u16(sim.dut->bias_flat_i,  zero_u16.data(), NCH_OUT);
    sim.reset();

    const std::vector<std::string> sample_names = {
        "rand0", "rand1", "half", "gradient", "rand_low", "rand_high"
    };

    int n_pass = 0, n_fail = 0;
    double agg_cos = 0.0, agg_mae = 0.0, agg_max = 0.0;

    for (const auto& sname : sample_names) {
        Sample s = load_sample(sname);
        pack_u16(sim.dut->scale_flat_i, s.scale_fp16.data(), NCH_OUT);
        pack_u16(sim.dut->bias_flat_i,  s.bias_fp16.data(),  NCH_OUT);

        std::vector<int8_t> y_i8(OUT_H * OUT_W * NCH_OUT, 0);
        int produced = 0;
        const int N_OUT = OUT_H * OUT_W;

        std::vector<int> inflight;
        inflight.reserve(N_OUT + TOTAL_LAT);

        for (int step = 0; step < N_OUT + TOTAL_LAT + 4; step++) {
            if (step < N_OUT) {
                int oy = step / OUT_W;
                int ox = step % OUT_W;
                int8_t win[N_LANE];
                build_window(s, oy, ox, win);
                pack_bytes(sim.dut->x_flat_i, win, N_LANE);
                sim.dut->valid_i = 1;
                inflight.push_back(step);
            } else {
                sim.dut->valid_i = 0;
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

        std::vector<float> dut_f(N_OUT * NCH_OUT);
        for (int i = 0; i < N_OUT * NCH_OUT; i++)
            dut_f[i] = (float)((double)y_i8[i] * S_OUT_SILU);

        Stats st_ort = compute_stats(dut_f, s.ref_ort);
        Stats st_hw  = compute_stats(dut_f, s.ref_hw);
        double pass_mae_thresh = 0.05 * st_ort.out_range;
        bool pass = (st_ort.cos > 0.998) && (st_ort.mae < pass_mae_thresh);

        printf("\n──── sample %s ────\n", sname.c_str());
        printf("  DUT vs ORT  : max_abs=%.4f  mae=%.4f  cos=%.6f  out_range=%.3f  thresh=%.4f\n",
               st_ort.max_abs, st_ort.mae, st_ort.cos, st_ort.out_range, pass_mae_thresh);
        printf("  DUT vs HW-ref: max_abs=%.4f mae=%.4f  cos=%.6f\n",
               st_hw.max_abs, st_hw.mae, st_hw.cos);
        if (st_ort.worst_idx >= 0) {
            int i = st_ort.worst_idx;
            int c = i % NCH_OUT, ox = (i / NCH_OUT) % OUT_W, oy = (i / NCH_OUT) / OUT_W;
            printf("  worst-px: (oy=%d ox=%d c=%d) dut=%.4f ort=%.4f hw=%.4f\n",
                   oy, ox, c, st_ort.worst_dut, st_ort.worst_ref, s.ref_hw[i]);
        }
        printf("  → %s\n", pass ? "PASS" : "FAIL");

        sim.check(pass, std::string("sample ") + sname + " pass");
        if (pass) n_pass++; else n_fail++;
        agg_cos += st_ort.cos;
        agg_mae += st_ort.mae;
        agg_max  = std::max(agg_max, st_ort.max_abs);
    }

    printf("\n========================================\n");
    printf("Aggregate: %d/%zu samples passed\n", n_pass, sample_names.size());
    printf("  avg cos     = %.6f\n", agg_cos / sample_names.size());
    printf("  avg mae     = %.6f\n", agg_mae / sample_names.size());
    printf("  worst max   = %.6f\n", agg_max);
    printf("Cycle accounting (layer_3 unit test, 8x8 ROI, stride 1):\n");
    printf("  ops/output-pixel : %d MAC + %d requant + %d silu (Cin=%d, K=%d)\n",
           N_LANE * NCH_OUT, NCH_OUT, NCH_OUT, NCH_IN, K);
    printf("  cycles/out-pixel : 1 (throughput)\n");
    printf("  pipeline latency : %d cycles (dotN=%d + requant=%d + silu=%d)\n",
           TOTAL_LAT, DOT_LAT, REQUANT_LAT, SILU_LAT);
    printf("  ROI (8x8 = %d pix)  total cycles : %d\n",
           OUT_H * OUT_W, OUT_H * OUT_W + TOTAL_LAT - 1);
    printf("Frame cycles at full scale (160x160 out = 25600 pix, P_PIX=1, P_COUT=8, P_CIN=48):\n");
    printf("  expected ~%d cycles (matches scale_pkg::LAYER_3_CYCLES=76800)\n",
           (NCH_IN * K * K / 48) * 160 * 160);
    printf("========================================\n");

    return sim.finish();
}
