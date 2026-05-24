// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_4_test — Integration test for tiled layer_4 (3x3 s1, 8->16, +residual).
// Single-tile case: N_COUT_TILE = N_CIN_TILE = 1, beats/pix = 1.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

#include "Vlayer_4_tb.h"
#include "sim_ctrl.h"

using DUT = Vlayer_4_tb;

static constexpr int NCH_IN     = 8;
static constexpr int NCH_OUT    = 16;
static constexpr int K          = 3;
static constexpr int STRIDE     = 1;
static constexpr int P_COUT     = 16;
static constexpr int P_CIN      = 8;
static constexpr int N_COUT_TILE = (NCH_OUT + P_COUT - 1) / P_COUT;       // 1
static constexpr int N_CIN_TILE  = (NCH_IN  + P_CIN  - 1) / P_CIN;        // 1
static constexpr int N_LANE_TILE = K * K * P_CIN;                          // 72
static constexpr int N_LANE_FULL = K * K * NCH_IN;                         // 72
static constexpr int ROI_H      = 8;
static constexpr int ROI_W      = 8;
static constexpr int PAD_H      = ROI_H + 2;
static constexpr int PAD_W      = ROI_W + 2;
static constexpr int OUT_H      = ROI_H / STRIDE;
static constexpr int OUT_W      = ROI_W / STRIDE;
static constexpr double S_OUT_SILU = 32.0 / 127.0;

// Pipeline latency: same as L1 + add_rq (1)
static constexpr int DOT_LAT     = 1 + 7;  // clog2(72)=7
static constexpr int ACC_LAT     = 1;
static constexpr int RQ_IN_LAT   = 1;
static constexpr int REQUANT_LAT = 7;
static constexpr int SILU_LAT    = 1;
static constexpr int ADD_LAT     = 12;       // add_rq depth (conservative)
static constexpr int TOTAL_LAT   = DOT_LAT + ACC_LAT + RQ_IN_LAT + REQUANT_LAT + SILU_LAT + ADD_LAT;

static std::string stim_dir() { return std::string("../stim"); }

static std::vector<uint32_t> load_hex(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "could not open %s\n", path.c_str()); std::exit(2); }
    std::vector<uint32_t> v; std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        v.push_back((uint32_t)std::stoul(line, nullptr, 16));
    }
    return v;
}

static float u32_to_f32(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }

struct Sample {
    std::string name;
    std::vector<int8_t>   input_i8;
    std::vector<int8_t>   resid_i8;
    std::vector<uint16_t> scale_fp16, bias_fp16;
    std::vector<uint16_t> r_scale_fp16, r_bias_fp16;
    std::vector<float>    ref_ort, ref_hw;
};

static Sample load_sample(const std::string& name) {
    Sample s; s.name = name;
    auto inp = load_hex(stim_dir() + "/" + name + ".input_i8.hex");
    s.input_i8.resize(inp.size());
    for (size_t i = 0; i < inp.size(); i++)
        s.input_i8[i] = (int8_t)(uint8_t)(inp[i] & 0xFF);
    auto rsd = load_hex(stim_dir() + "/" + name + ".resid_i8.hex");
    s.resid_i8.resize(rsd.size());
    for (size_t i = 0; i < rsd.size(); i++)
        s.resid_i8[i] = (int8_t)(uint8_t)(rsd[i] & 0xFF);
    for (auto v : load_hex(stim_dir() + "/" + name + ".scale_fp16.hex"))
        s.scale_fp16.push_back((uint16_t)v);
    for (auto v : load_hex(stim_dir() + "/" + name + ".bias_fp16.hex"))
        s.bias_fp16.push_back((uint16_t)v);
    for (auto v : load_hex(stim_dir() + "/" + name + ".r_scale_fp16.hex"))
        s.r_scale_fp16.push_back((uint16_t)v);
    for (auto v : load_hex(stim_dir() + "/" + name + ".r_bias_fp16.hex"))
        s.r_bias_fp16.push_back((uint16_t)v);
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

static void build_window_tile(const Sample& s, int oy, int ox, int /*cit*/,
                              int8_t out[N_LANE_TILE]) {
    // K=3, P_CIN=8, NCH_IN=8 -> cin_tile covers all channels.
    for (int kh = 0; kh < K; kh++) {
        for (int kw = 0; kw < K; kw++) {
            int h_idx = oy * STRIDE + kh;
            int w_idx = ox * STRIDE + kw;
            for (int kc_local = 0; kc_local < P_CIN; kc_local++) {
                int kc = kc_local;  // cit=0, kc_base=0
                int8_t v = 0;
                if (kc < NCH_IN) {
                    int in_off = (h_idx * PAD_W + w_idx) * NCH_IN + kc;
                    v = s.input_i8[in_off];
                }
                int lane = (kh * K + kw) * P_CIN + kc_local;
                out[lane] = v;
            }
        }
    }
}

static void build_weight_tile(const std::vector<int8_t>& w_full, int /*ct*/, int /*cit*/,
                              int8_t out[P_COUT * N_LANE_TILE]) {
    for (int co_local = 0; co_local < P_COUT; co_local++) {
        int co_global = co_local;
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                for (int kc_local = 0; kc_local < P_CIN; kc_local++) {
                    int kc = kc_local;
                    int8_t v = 0;
                    if (co_global < NCH_OUT && kc < NCH_IN) {
                        int full_lane = (kh * K + kw) * NCH_IN + kc;
                        v = w_full[co_global * N_LANE_FULL + full_lane];
                    }
                    int tile_lane = (kh * K + kw) * P_CIN + kc_local;
                    out[co_local * N_LANE_TILE + tile_lane] = v;
                }
            }
        }
    }
}

struct Stats {
    double max_abs = 0.0, mae = 0.0, cos = 0.0, out_range = 0.0;
    int n = 0;
};

static Stats compute_stats(const std::vector<float>& dut, const std::vector<float>& ref) {
    Stats st; st.n = (int)dut.size();
    double sum_abs = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    float mn = 1e30f, mx = -1e30f;
    for (int i = 0; i < st.n; i++) {
        double e = std::abs((double)dut[i] - (double)ref[i]);
        sum_abs += e;
        if (e > st.max_abs) st.max_abs = e;
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
    sim.max_time = 4000000;

    auto w_raw = load_hex(stim_dir() + "/weights.i8.hex");
    if ((int)w_raw.size() != NCH_OUT * N_LANE_FULL) {
        fprintf(stderr, "weights size mismatch\n"); return 2;
    }
    std::vector<int8_t> w_full(NCH_OUT * N_LANE_FULL);
    for (int i = 0; i < NCH_OUT * N_LANE_FULL; i++)
        w_full[i] = (int8_t)(uint8_t)(w_raw[i] & 0xFF);

    sim.dut->valid_i         = 0;
    sim.dut->first_cin_i     = 0;
    sim.dut->last_cin_i      = 0;
    sim.dut->cout_tile_idx_i = 0;
    std::vector<int8_t>   zero_lanes(N_LANE_TILE, 0);
    std::vector<int8_t>   zero_w(P_COUT * N_LANE_TILE, 0);
    std::vector<int8_t>   zero_r(P_COUT, 0);
    std::vector<uint16_t> zero_u16(P_COUT, 0);
    pack_bytes(sim.dut->x_flat_i, zero_lanes.data(), N_LANE_TILE);
    pack_bytes(sim.dut->w_flat_i, zero_w.data(), P_COUT * N_LANE_TILE);
    pack_bytes(sim.dut->r_flat_i, zero_r.data(), P_COUT);
    pack_u16(sim.dut->scale_flat_i,   zero_u16.data(), P_COUT);
    pack_u16(sim.dut->bias_flat_i,    zero_u16.data(), P_COUT);
    pack_u16(sim.dut->r_scale_flat_i, zero_u16.data(), P_COUT);
    pack_u16(sim.dut->r_bias_flat_i,  zero_u16.data(), P_COUT);
    sim.reset();

    const std::vector<std::string> sample_names = {
        "rand0", "rand1", "half", "gradient", "rand_low", "rand_high"
    };

    int n_pass = 0;
    double agg_cos = 0.0, agg_mae = 0.0, agg_max = 0.0;

    for (const auto& sname : sample_names) {
        Sample s = load_sample(sname);
        std::vector<int8_t> y_i8(OUT_H * OUT_W * NCH_OUT, 0);

        const int N_OUT = OUT_H * OUT_W;
        const int BEATS_PER_PIX = N_COUT_TILE * N_CIN_TILE;  // 1
        const int TOTAL_BEATS  = N_OUT * BEATS_PER_PIX;

        struct Commit { int pix; int ct; };
        std::vector<Commit> inflight;
        inflight.reserve(N_OUT * N_COUT_TILE + TOTAL_LAT);
        int produced = 0;

        for (int step = 0; step < TOTAL_BEATS + TOTAL_LAT + 8; step++) {
            if (step < TOTAL_BEATS) {
                int pix = step;
                int oy  = pix / OUT_W;
                int ox  = pix % OUT_W;

                int8_t win[N_LANE_TILE];
                build_window_tile(s, oy, ox, 0, win);
                pack_bytes(sim.dut->x_flat_i, win, N_LANE_TILE);

                int8_t wtile[P_COUT * N_LANE_TILE];
                build_weight_tile(w_full, 0, 0, wtile);
                pack_bytes(sim.dut->w_flat_i, wtile, P_COUT * N_LANE_TILE);

                // Conv requant scale/bias for the single cout-tile.
                uint16_t sbuf[P_COUT], bbuf[P_COUT];
                for (int c = 0; c < P_COUT; c++) {
                    sbuf[c] = (c < NCH_OUT) ? s.scale_fp16[c] : 0;
                    bbuf[c] = (c < NCH_OUT) ? s.bias_fp16 [c] : 0;
                }
                pack_u16(sim.dut->scale_flat_i, sbuf, P_COUT);
                pack_u16(sim.dut->bias_flat_i,  bbuf, P_COUT);

                // Residual per output pixel (16ch slice for this pix)
                int8_t   rbuf[P_COUT];
                uint16_t rsbuf[P_COUT], rbbuf[P_COUT];
                for (int c = 0; c < P_COUT; c++) {
                    rbuf[c]  = (c < NCH_OUT) ? s.resid_i8 [pix * NCH_OUT + c] : 0;
                    rsbuf[c] = (c < NCH_OUT) ? s.r_scale_fp16[c] : 0;
                    rbbuf[c] = (c < NCH_OUT) ? s.r_bias_fp16 [c] : 0;
                }
                pack_bytes(sim.dut->r_flat_i, rbuf, P_COUT);
                pack_u16(sim.dut->r_scale_flat_i, rsbuf, P_COUT);
                pack_u16(sim.dut->r_bias_flat_i,  rbbuf, P_COUT);

                sim.dut->valid_i         = 1;
                sim.dut->first_cin_i     = 1;
                sim.dut->last_cin_i      = 1;
                sim.dut->cout_tile_idx_i = 0;

                inflight.push_back({pix, 0});
            } else {
                sim.dut->valid_i     = 0;
                sim.dut->first_cin_i = 0;
                sim.dut->last_cin_i  = 0;
            }

            sim.tick();

            if (sim.dut->valid_o && produced < (int)inflight.size()) {
                int pix = inflight[produced].pix;
                int8_t row[P_COUT];
                unpack_bytes(sim.dut->y_flat_o, row, P_COUT);
                for (int c = 0; c < P_COUT; c++) {
                    if (c < NCH_OUT)
                        y_i8[pix * NCH_OUT + c] = row[c];
                }
                produced++;
            }
        }

        if (produced != N_OUT * N_COUT_TILE) {
            printf("  WARN: produced=%d expected=%d\n", produced, N_OUT * N_COUT_TILE);
        }

        // Output is post-add int8 scaled by S_OUT_SILU (conv_layer inv_out_scale = 1/S_OUT_SILU).
        std::vector<float> dut_f(N_OUT * NCH_OUT);
        for (int i = 0; i < N_OUT * NCH_OUT; i++)
            dut_f[i] = (float)((double)y_i8[i] * S_OUT_SILU);

        Stats st_ort = compute_stats(dut_f, s.ref_ort);
        Stats st_hw  = compute_stats(dut_f, s.ref_hw);
        double pass_mae_thresh = 0.05 * st_ort.out_range;
        bool pass = (st_ort.cos > 0.998) && (st_ort.mae < pass_mae_thresh);

        printf("\n--- sample %s ---\n", sname.c_str());
        printf("  DUT vs ORT  : max=%.4f mae=%.4f cos=%.6f range=%.3f thresh=%.4f\n",
               st_ort.max_abs, st_ort.mae, st_ort.cos, st_ort.out_range, pass_mae_thresh);
        printf("  DUT vs HW   : max=%.4f mae=%.4f cos=%.6f\n",
               st_hw.max_abs, st_hw.mae, st_hw.cos);
        printf("  => %s\n", pass ? "PASS" : "FAIL");

        sim.check(pass, std::string("sample ") + sname + " pass");
        if (pass) n_pass++;
        agg_cos += st_ort.cos; agg_mae += st_ort.mae;
        agg_max = std::max(agg_max, st_ort.max_abs);
    }

    printf("\n==== layer_4: %d/%zu pass, avg cos=%.6f ====\n",
           n_pass, sample_names.size(), agg_cos / sample_names.size());

    return sim.finish();
}
