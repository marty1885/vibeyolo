// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_11_test — Integration test for the tiled YOLO26n /model.5/conv layer.
//
// Layer params (scale_pkg_dv):
//   COUT=128, CIN=128, K=3, P_COUT=16, P_CIN=8
//   N_COUT_TILE=8, N_CIN_TILE=16
// Beats per output pixel = N_COUT_TILE * N_CIN_TILE = 128.
// For an 8x8 output ROI: 8*8*128 = 8192 driver beats.
//
// Driver protocol (cin innermost, cout outermost):
//   for oy, ox:
//     for ct = 0..N_COUT_TILE-1:
//       for cit = 0..N_CIN_TILE-1:
//         drive (x_tile[oy,ox,cit], w_tile[ct,cit], scale[ct], bias[ct],
//                first=(cit==0), last=(cit==N_CIN_TILE-1), cout_tile=ct)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

#include "Vlayer_11_tb.h"
#include "sim_ctrl.h"

using DUT = Vlayer_11_tb;

// Geometry — must match extract.py + layer_11.sv parameters
static constexpr int NCH_IN     = 128;
static constexpr int NCH_OUT    = 128;
static constexpr int K          = 3;
static constexpr int STRIDE     = 2;
static constexpr int P_COUT     = 16;
static constexpr int P_CIN      = 8;
static constexpr int N_COUT_TILE = NCH_OUT / P_COUT;       // 8
static constexpr int N_CIN_TILE  = NCH_IN  / P_CIN;        // 16
static constexpr int N_LANE_TILE = K * K * P_CIN;          // 72
static constexpr int N_LANE_FULL = K * K * NCH_IN;         // 1152 (extract.py layout)
static constexpr int ROI_H      = 16;
static constexpr int ROI_W      = 16;
static constexpr int PAD_H      = ROI_H + 2;               // 18
static constexpr int PAD_W      = ROI_W + 2;               // 18
static constexpr int OUT_H      = ROI_H / STRIDE;          // 8
static constexpr int OUT_W      = ROI_W / STRIDE;          // 8
static constexpr double S_OUT_SILU = 4.0 / 127.0;

// Pipeline latency for tracking inflight commits.
static constexpr int DOT_LAT     = 8;   // 1 + clog2(72)
static constexpr int ACC_LAT     = 1;   // accumulator stage
static constexpr int RQ_IN_LAT   = 1;   // commit_q -> rq_valid_in
static constexpr int REQUANT_LAT = 4;
static constexpr int SILU_LAT    = 1;
static constexpr int TOTAL_LAT   = DOT_LAT + ACC_LAT + RQ_IN_LAT + REQUANT_LAT + SILU_LAT;

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
    std::vector<int8_t>   input_i8;   // PAD_H*PAD_W*NCH_IN
    std::vector<uint16_t> scale_fp16; // NCH_OUT
    std::vector<uint16_t> bias_fp16;  // NCH_OUT
    std::vector<float>    ref_ort;    // OUT_H*OUT_W*NCH_OUT
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

// Build the K*K*P_CIN i8 tile for output pixel (oy,ox) and cin_tile cit.
// Lane index = (kh*K + kw)*P_CIN + kc_local.
static void build_window_tile(const Sample& s, int oy, int ox, int cit,
                              int8_t out[N_LANE_TILE]) {
    int kc_base = cit * P_CIN;
    for (int kh = 0; kh < K; kh++) {
        for (int kw = 0; kw < K; kw++) {
            int h_idx = oy * STRIDE + kh;
            int w_idx = ox * STRIDE + kw;
            for (int kc_local = 0; kc_local < P_CIN; kc_local++) {
                int kc = kc_base + kc_local;
                int in_off = (h_idx * PAD_W + w_idx) * NCH_IN + kc;
                int lane   = (kh * K + kw) * P_CIN + kc_local;
                out[lane] = s.input_i8[in_off];
            }
        }
    }
}

// Build the P_COUT * (K*K*P_CIN) i8 weight tile for (cout_tile=ct, cin_tile=cit).
// w_full layout is (NCH_OUT, N_LANE_FULL) with lane = (kh*K+kw)*NCH_IN + kc.
// Output layout: per output channel (co_local), lane = (kh*K+kw)*P_CIN + kc_local.
static void build_weight_tile(const std::vector<int8_t>& w_full, int ct, int cit,
                              int8_t out[P_COUT * N_LANE_TILE]) {
    int kc_base = cit * P_CIN;
    int co_base = ct  * P_COUT;
    for (int co_local = 0; co_local < P_COUT; co_local++) {
        int co_global = co_base + co_local;
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                for (int kc_local = 0; kc_local < P_CIN; kc_local++) {
                    int kc        = kc_base + kc_local;
                    int full_lane = (kh * K + kw) * NCH_IN + kc;
                    int tile_lane = (kh * K + kw) * P_CIN  + kc_local;
                    out[co_local * N_LANE_TILE + tile_lane] =
                        w_full[co_global * N_LANE_FULL + full_lane];
                }
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
    sim.max_time = 4000000;   // ~ROI 64px × 128 beats × 6 latency margin

    auto w_raw = load_hex(stim_dir() + "/weights.i8.hex");
    if ((int)w_raw.size() != NCH_OUT * N_LANE_FULL) {
        fprintf(stderr, "weights size mismatch: got %zu expected %d\n",
                w_raw.size(), NCH_OUT * N_LANE_FULL);
        return 2;
    }
    std::vector<int8_t> w_full(NCH_OUT * N_LANE_FULL);
    for (int i = 0; i < NCH_OUT * N_LANE_FULL; i++)
        w_full[i] = (int8_t)(uint8_t)(w_raw[i] & 0xFF);

    // Initial state
    sim.dut->valid_i         = 0;
    sim.dut->first_cin_i     = 0;
    sim.dut->last_cin_i      = 0;
    sim.dut->cout_tile_idx_i = 0;
    std::vector<int8_t> zero_lanes(N_LANE_TILE, 0);
    pack_bytes(sim.dut->x_flat_i, zero_lanes.data(), N_LANE_TILE);
    std::vector<int8_t> zero_w(P_COUT * N_LANE_TILE, 0);
    pack_bytes(sim.dut->w_flat_i, zero_w.data(), P_COUT * N_LANE_TILE);
    std::vector<uint16_t> zero_u16(P_COUT, 0);
    pack_u16(sim.dut->scale_flat_i, zero_u16.data(), P_COUT);
    pack_u16(sim.dut->bias_flat_i,  zero_u16.data(), P_COUT);
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
        const int BEATS_PER_PIX = N_COUT_TILE * N_CIN_TILE;
        const int TOTAL_BEATS  = N_OUT * BEATS_PER_PIX;

        // We track inflight commits — each (pixel, cout_tile) launches one
        // requant commit at the last_cin beat. The DUT emits them in order.
        struct Commit { int pix; int ct; };
        std::vector<Commit> inflight;
        inflight.reserve(N_OUT * N_COUT_TILE + TOTAL_LAT);
        int produced = 0;

        for (int step = 0; step < TOTAL_BEATS + TOTAL_LAT + 8; step++) {
            if (step < TOTAL_BEATS) {
                int pix = step / BEATS_PER_PIX;
                int rem = step % BEATS_PER_PIX;
                int ct  = rem / N_CIN_TILE;
                int cit = rem % N_CIN_TILE;
                int oy  = pix / OUT_W;
                int ox  = pix % OUT_W;

                int8_t win[N_LANE_TILE];
                build_window_tile(s, oy, ox, cit, win);
                pack_bytes(sim.dut->x_flat_i, win, N_LANE_TILE);

                int8_t wtile[P_COUT * N_LANE_TILE];
                build_weight_tile(w_full, ct, cit, wtile);
                pack_bytes(sim.dut->w_flat_i, wtile, P_COUT * N_LANE_TILE);

                // Per-cout-tile scale/bias slice
                uint16_t sbuf[P_COUT], bbuf[P_COUT];
                int co_base = ct * P_COUT;
                for (int c = 0; c < P_COUT; c++) {
                    sbuf[c] = s.scale_fp16[co_base + c];
                    bbuf[c] = s.bias_fp16 [co_base + c];
                }
                pack_u16(sim.dut->scale_flat_i, sbuf, P_COUT);
                pack_u16(sim.dut->bias_flat_i,  bbuf, P_COUT);

                sim.dut->valid_i         = 1;
                sim.dut->first_cin_i     = (cit == 0);
                sim.dut->last_cin_i      = (cit == N_CIN_TILE - 1);
                sim.dut->cout_tile_idx_i = (uint8_t)ct;

                if (cit == N_CIN_TILE - 1) {
                    inflight.push_back({pix, ct});
                }
            } else {
                sim.dut->valid_i     = 0;
                sim.dut->first_cin_i = 0;
                sim.dut->last_cin_i  = 0;
            }

            sim.tick();

            if (sim.dut->valid_o && produced < (int)inflight.size()) {
                int pix = inflight[produced].pix;
                int ct  = inflight[produced].ct;
                int got_ct = sim.dut->cout_tile_idx_o;
                if (got_ct != ct) {
                    fprintf(stderr,
                        "cout_tile_idx mismatch at produced=%d: expected %d got %d\n",
                        produced, ct, got_ct);
                }
                int8_t row[P_COUT];
                unpack_bytes(sim.dut->y_flat_o, row, P_COUT);
                int co_base = ct * P_COUT;
                for (int c = 0; c < P_COUT; c++)
                    y_i8[pix * NCH_OUT + co_base + c] = row[c];
                produced++;
            }
        }

        if (produced != N_OUT * N_COUT_TILE) {
            printf("  WARN: produced=%d expected=%d\n",
                   produced, N_OUT * N_COUT_TILE);
        }

        std::vector<float> dut_f(N_OUT * NCH_OUT);
        for (int i = 0; i < N_OUT * NCH_OUT; i++)
            dut_f[i] = (float)((double)y_i8[i] * S_OUT_SILU);

        Stats st_ort = compute_stats(dut_f, s.ref_ort);
        Stats st_hw  = compute_stats(dut_f, s.ref_hw);
        double pass_mae_thresh = 0.05 * st_ort.out_range;
        bool pass = (st_ort.cos > 0.998) && (st_ort.mae < pass_mae_thresh);

        printf("\n--- sample %s ---\n", sname.c_str());
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
        printf("  => %s\n", pass ? "PASS" : "FAIL");

        sim.check(pass, std::string("sample ") + sname + " pass");
        if (pass) n_pass++;
        agg_cos += st_ort.cos;
        agg_mae += st_ort.mae;
        agg_max  = std::max(agg_max, st_ort.max_abs);
    }

    printf("\n========================================\n");
    printf("Aggregate: %d/%zu samples passed\n", n_pass, sample_names.size());
    printf("  avg cos     = %.6f\n", agg_cos / sample_names.size());
    printf("  avg mae     = %.6f\n", agg_mae / sample_names.size());
    printf("  worst max   = %.6f\n", agg_max);
    printf("Tiled layer_11 (P_COUT=%d, P_CIN=%d, K=%d):\n",
           P_COUT, P_CIN, K);
    printf("  beats/pixel = N_COUT_TILE * N_CIN_TILE = %d * %d = %d\n",
           N_COUT_TILE, N_CIN_TILE, N_COUT_TILE * N_CIN_TILE);
    printf("  pipeline latency = %d (dot=%d acc=%d rq_in=%d rq=%d silu=%d)\n",
           TOTAL_LAT, DOT_LAT, ACC_LAT, RQ_IN_LAT, REQUANT_LAT, SILU_LAT);
    printf("  ROI %dx%d => %d driver beats\n",
           OUT_H, OUT_W, OUT_H * OUT_W * N_COUT_TILE * N_CIN_TILE);
    printf("  full 40x40 frame => %d cycles (vs scale_pkg::LAYER_11_CYCLES)\n",
           40 * 40 * N_COUT_TILE * N_CIN_TILE);
    printf("========================================\n");

    return sim.finish();
}
