// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// conv_layer_test — random-stimulus test for the conv_layer IP.
//
// One config is baked in per binary via -DCFG_* macros. Each binary runs
// N_SEEDS random seeds, compares DUT vs REF (must be bit-exact) and DUT
// vs a C++ floating-point shadow (cosine ≥ 0.999, no tolerance for the
// shadow itself — both sides use the same math).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>

#include "Vconv_layer_tb.h"
#include "sim_ctrl.h"

// ─── Per-config geometry (set by Makefile -DCFG_*) ────────────────
#ifndef CFG_CIN
#define CFG_CIN 32
#endif
#ifndef CFG_COUT
#define CFG_COUT 32
#endif
#ifndef CFG_K
#define CFG_K 1
#endif
#ifndef CFG_STRIDE
#define CFG_STRIDE 1
#endif
#ifndef CFG_P_COUT
#define CFG_P_COUT 16
#endif
#ifndef CFG_P_CIN
#define CFG_P_CIN 8
#endif
#ifndef CFG_RESIDUAL
#define CFG_RESIDUAL 0
#endif
#ifndef CFG_SILU
#define CFG_SILU 1
#endif
#ifndef CFG_NAME
#define CFG_NAME "default"
#endif

using DUT = Vconv_layer_tb;

static constexpr int CIN         = CFG_CIN;
static constexpr int COUT        = CFG_COUT;
static constexpr int K           = CFG_K;
static constexpr int STRIDE      = CFG_STRIDE;
static constexpr int P_COUT      = CFG_P_COUT;
static constexpr int P_CIN       = CFG_P_CIN;
static constexpr int RESIDUAL    = CFG_RESIDUAL;
static constexpr int SILU_EN     = CFG_SILU;
static constexpr int N_COUT_TILE = (COUT + P_COUT - 1) / P_COUT;
static constexpr int N_CIN_TILE  = (CIN  + P_CIN  - 1) / P_CIN;
static constexpr int N_LANE_TILE = K * K * P_CIN;
static constexpr int OUT_H       = 8;
static constexpr int OUT_W       = 8;
static constexpr int IN_H        = OUT_H * STRIDE + (K - 1);
static constexpr int IN_W        = OUT_W * STRIDE + (K - 1);
static constexpr int N_OUT       = OUT_H * OUT_W;
static constexpr int N_SEEDS     = 4;

static constexpr double S_OUT_PRE  = 4.0 / 127.0;
static constexpr double S_OUT_SILU = 4.0 / 127.0;

// Pipeline latency: 1 + clog2(N_LANE_TILE) + 1(acc) + 1(rq_in) + 7(requant)
// + 1 if SILU + 12 if RESIDUAL.  (requant/add_rq grew after i32_to_fp16 and
// fp16_fma were pipelined for timing.)
static int clog2i(int v) { int r = 0; while ((1 << r) < v) r++; return r; }

// ─── fp16 conversion (RNE, finite normal/subnormal path) ──────────
static uint16_t f32_to_fp16(float fv) {
    uint32_t f; std::memcpy(&f, &fv, 4);
    uint32_t sign = (f >> 31) & 0x1u;
    int32_t  exp  = ((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFu;
    if (((f >> 23) & 0xFFu) == 0xFFu) {
        // inf/nan
        return (uint16_t)((sign << 15) | (0x1Fu << 10) | (mant ? 0x200u : 0u));
    }
    if (exp >= 31) return (uint16_t)((sign << 15) | (0x1Fu << 10));
    if (exp <= 0) {
        // subnormal / zero — clamp to 0 for our purposes (we never use tiny)
        return (uint16_t)(sign << 15);
    }
    // RNE round of the 13 LSB
    uint32_t mant10 = mant >> 13;
    uint32_t rem    = mant & 0x1FFFu;
    uint32_t half   = 0x1000u;
    if (rem > half || (rem == half && (mant10 & 1))) {
        mant10 += 1;
        if (mant10 == 0x400u) { mant10 = 0; exp += 1; if (exp >= 31) return (uint16_t)((sign << 15) | (0x1Fu << 10)); }
    }
    return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | mant10);
}
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            int e = -1;
            while ((mant & 0x400u) == 0) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            f = (sign << 31) | (uint32_t)((127 + (-14 + e)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | (uint32_t)((exp - 15 + 127) << 23) | (mant << 13);
    }
    float fv; std::memcpy(&fv, &f, 4); return fv;
}

// ─── Packing helpers (i8 / u16 → flat words) ─────────────────────
static void pack_bytes(uint8_t* dst, int nbytes_dst_words,
                       const int8_t* src, int nbytes) {
    for (int i = 0; i < nbytes_dst_words * 4; i++) dst[i] = 0;
    for (int i = 0; i < nbytes; i++) dst[i] = (uint8_t)src[i];
}
static void pack_u16(uint8_t* dst, int nwords, const uint16_t* src, int nshorts) {
    for (int i = 0; i < nwords * 4; i++) dst[i] = 0;
    for (int i = 0; i < nshorts; i++) {
        dst[i*2 + 0] = (uint8_t)(src[i]      & 0xFF);
        dst[i*2 + 1] = (uint8_t)((src[i]>>8) & 0xFF);
    }
}
static void unpack_bytes(const uint8_t* src, int8_t* dst, int nbytes) {
    for (int i = 0; i < nbytes; i++) dst[i] = (int8_t)src[i];
}

// ─── Wide-port read/write helpers ─────────────────────────────────
// VlWide objects are arrays of uint32_t for sizes > 64 bits.
template <typename T>
static uint8_t* wide_bytes(T& dst) { return reinterpret_cast<uint8_t*>(&dst); }
template <typename T>
static int wide_words(T& dst) { return (int)(sizeof(dst) / 4); }

// ─── Drive one tile-beat ──────────────────────────────────────────
static void drive_beat(DUT* dut,
                       const int8_t* xtile, int xn,
                       const int8_t* wtile, int wn,
                       const uint16_t* sbuf, const uint16_t* bbuf,
                       const int8_t* rbuf, const uint16_t* rsbuf, const uint16_t* rbias,
                       bool first, bool last, int ct) {
    pack_bytes(wide_bytes(dut->x_flat_i),       wide_words(dut->x_flat_i),       xtile, xn);
    pack_bytes(wide_bytes(dut->w_flat_i),       wide_words(dut->w_flat_i),       wtile, wn);
    pack_u16  (wide_bytes(dut->scale_flat_i),   wide_words(dut->scale_flat_i),   sbuf,  P_COUT);
    pack_u16  (wide_bytes(dut->bias_flat_i),    wide_words(dut->bias_flat_i),    bbuf,  P_COUT);
    pack_bytes(wide_bytes(dut->r_flat_i),       wide_words(dut->r_flat_i),       rbuf,  P_COUT);
    pack_u16  (wide_bytes(dut->r_scale_flat_i), wide_words(dut->r_scale_flat_i), rsbuf, P_COUT);
    pack_u16  (wide_bytes(dut->r_bias_flat_i),  wide_words(dut->r_bias_flat_i),  rbias, P_COUT);
    dut->valid_i         = 1;
    dut->first_cin_i     = first ? 1 : 0;
    dut->last_cin_i      = last  ? 1 : 0;
    dut->cout_tile_idx_i = (uint8_t)ct;
}
static void idle_beat(DUT* dut) {
    dut->valid_i     = 0;
    dut->first_cin_i = 0;
    dut->last_cin_i  = 0;
}

// ─── C++ shadow ───────────────────────────────────────────────────
// Computes the same i8 output the DUT should produce, in float.
static float silu_q(int8_t cp, float in_scale, float out_scale) {
    float x = (float)cp * in_scale;
    float s = x / (1.0f + std::exp(-x));
    float y = s / out_scale;
    if (y >  127.0f) y =  127.0f;
    if (y < -128.0f) y = -128.0f;
    return std::round(y);  // RNE on .5 ties is close enough vs LUT for cosine
}

static std::vector<int8_t>
shadow_conv(const std::vector<int8_t>& in,
            const std::vector<int8_t>& wt,
            const std::vector<uint16_t>& scale,
            const std::vector<uint16_t>& bias,
            const std::vector<int8_t>& rdat,
            const std::vector<uint16_t>& rscale,
            const std::vector<uint16_t>& rbias) {
    std::vector<int8_t> out(N_OUT * COUT, 0);
    float fp_s_out     = fp16_to_f32(f32_to_fp16(S_OUT_SILU));
    float fp_inv_s_out = fp16_to_f32(f32_to_fp16(1.0f / (float)S_OUT_SILU));

    for (int oy = 0; oy < OUT_H; oy++) {
      for (int ox = 0; ox < OUT_W; ox++) {
        for (int co = 0; co < COUT; co++) {
          int64_t acc = 0;
          for (int kh = 0; kh < K; kh++)
          for (int kw = 0; kw < K; kw++)
          for (int ci = 0; ci < CIN; ci++) {
            int ih = oy * STRIDE + kh;
            int iw = ox * STRIDE + kw;
            int8_t x_v = in[(ih * IN_W + iw) * CIN + ci];
            int8_t w_v = wt[((co * K + kh) * K + kw) * CIN + ci];
            acc += (int64_t)x_v * (int64_t)w_v;
          }
          float facc = (float)acc;
          float sc = fp16_to_f32(scale[co]);
          float bs = fp16_to_f32(bias[co]);
          float yf = facc * sc + bs;
          int yi = (int)std::lrintf(yf);
          if (yi >  127) yi =  127;
          if (yi < -128) yi = -128;
          int8_t y_rq = (int8_t)yi;
          int8_t y_sil;
          if (SILU_EN) {
              y_sil = (int8_t)silu_q(y_rq, (float)S_OUT_PRE, (float)S_OUT_SILU);
          } else {
              y_sil = y_rq;
          }
          int8_t y_final = y_sil;
          if (RESIDUAL) {
            float a = (float)y_sil * fp_s_out;
            float b = (float)rdat[(oy*OUT_W+ox)*COUT + co] * fp16_to_f32(rscale[co]);
            float sum = a + b;
            float yo = sum * fp_inv_s_out + fp16_to_f32(rbias[co]);
            int yoi = (int)std::lrintf(yo);
            if (yoi >  127) yoi =  127;
            if (yoi < -128) yoi = -128;
            y_final = (int8_t)yoi;
          }
          out[(oy*OUT_W+ox)*COUT + co] = y_final;
        }
      }
    }
    return out;
}

// ─── Tile builders (same convention as layer_11) ──────────────────
static void build_window_tile(const std::vector<int8_t>& in,
                              int oy, int ox, int cit, int8_t* o) {
    int kc_base = cit * P_CIN;
    for (int kh = 0; kh < K; kh++)
      for (int kw = 0; kw < K; kw++)
        for (int kc_local = 0; kc_local < P_CIN; kc_local++) {
            int ih = oy * STRIDE + kh;
            int iw = ox * STRIDE + kw;
            int kc = kc_base + kc_local;
            int in_off = (ih * IN_W + iw) * CIN + (kc < CIN ? kc : 0);
            int lane   = (kh * K + kw) * P_CIN + kc_local;
            o[lane] = (kc < CIN) ? in[in_off] : 0;
        }
}

static void build_weight_tile(const std::vector<int8_t>& w_full,
                              int ct, int cit, int8_t* o) {
    int kc_base = cit * P_CIN;
    int co_base = ct  * P_COUT;
    for (int co_local = 0; co_local < P_COUT; co_local++) {
      int co_global = co_base + co_local;
      bool co_oob = co_global >= COUT;
      for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++)
          for (int kc_local = 0; kc_local < P_CIN; kc_local++) {
              int kc = kc_base + kc_local;
              int tile_lane = (kh * K + kw) * P_CIN + kc_local;
              int8_t v = 0;
              if (!co_oob && kc < CIN) {
                  v = w_full[((co_global * K + kh) * K + kw) * CIN + kc];
              }
              o[co_local * N_LANE_TILE + tile_lane] = v;
          }
    }
}

// ─── Stats ────────────────────────────────────────────────────────
struct Stats { double max_abs = 0, mae = 0, cos = 0; int n = 0; };
static Stats compare(const std::vector<float>& a, const std::vector<float>& b) {
    Stats s; s.n = (int)a.size();
    double sa = 0, dot = 0, na = 0, nb = 0;
    for (int i = 0; i < s.n; i++) {
        double e = std::abs(a[i] - b[i]);
        sa += e; if (e > s.max_abs) s.max_abs = e;
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    s.mae = sa / std::max(1, s.n);
    s.cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    return s;
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);

    int DOT_LAT = 1 + clog2i(N_LANE_TILE);
    // requant=7, add_rq=12 after i32_to_fp16/fp16_fma were pipelined.
    int TOTAL_LAT = DOT_LAT + 1 + 1 + 7 + (SILU_EN ? 1 : 0) + (RESIDUAL ? 12 : 0);

    int BEATS_PER_PIX = N_COUT_TILE * N_CIN_TILE;
    int TOTAL_BEATS   = N_OUT * BEATS_PER_PIX;
    sim.max_time = (uint64_t)(TOTAL_BEATS + TOTAL_LAT + 64) * 4 * N_SEEDS + 200000;

    printf("=== conv_layer config: %s ===\n", CFG_NAME);
    printf("  CIN=%d COUT=%d K=%d STRIDE=%d P_COUT=%d P_CIN=%d RESIDUAL=%d SILU=%d\n",
           CIN, COUT, K, STRIDE, P_COUT, P_CIN, RESIDUAL, SILU_EN);
    printf("  N_COUT_TILE=%d N_CIN_TILE=%d beats/pix=%d ROI %dx%d total=%d beats lat=%d\n",
           N_COUT_TILE, N_CIN_TILE, BEATS_PER_PIX, OUT_H, OUT_W,
           TOTAL_BEATS, TOTAL_LAT);

    // initial zero
    sim.dut->valid_i = 0;
    sim.dut->first_cin_i = 0;
    sim.dut->last_cin_i  = 0;
    sim.dut->cout_tile_idx_i = 0;
    std::memset(wide_bytes(sim.dut->x_flat_i),     0, wide_words(sim.dut->x_flat_i)*4);
    std::memset(wide_bytes(sim.dut->w_flat_i),     0, wide_words(sim.dut->w_flat_i)*4);
    std::memset(wide_bytes(sim.dut->scale_flat_i), 0, wide_words(sim.dut->scale_flat_i)*4);
    std::memset(wide_bytes(sim.dut->bias_flat_i),  0, wide_words(sim.dut->bias_flat_i)*4);
    std::memset(wide_bytes(sim.dut->r_flat_i),     0, wide_words(sim.dut->r_flat_i)*4);
    std::memset(wide_bytes(sim.dut->r_scale_flat_i),0,wide_words(sim.dut->r_scale_flat_i)*4);
    std::memset(wide_bytes(sim.dut->r_bias_flat_i),0, wide_words(sim.dut->r_bias_flat_i)*4);
    sim.reset();

    int n_pass = 0;
    for (int seed = 0; seed < N_SEEDS; seed++) {
        std::mt19937 rng(0x515 + seed * 17 + 31 * CFG_K);
        std::uniform_int_distribution<int> i8(-32, 32);   // keep accumulator modest
        std::uniform_int_distribution<int> w8(-16, 16);
        std::uniform_int_distribution<int> r8(-32, 32);

        std::vector<int8_t> in(IN_H * IN_W * CIN);
        for (auto& v : in) v = (int8_t)i8(rng);
        std::vector<int8_t> wt(COUT * K * K * CIN);
        for (auto& v : wt) v = (int8_t)w8(rng);

        // Per-channel scale/bias. Pick small scale so accumulator * scale ~ ±64.
        // With acc up to ~ K*K*CIN*1024 we pick scale ~ 1/(K*K*CIN*16).
        float ss = 1.0f / (float)(K*K*CIN * 8);
        std::vector<uint16_t> scale(COUT), bias(COUT);
        std::uniform_real_distribution<float> sj(0.5f, 1.5f);
        std::uniform_real_distribution<float> bj(-1.5f, 1.5f);
        for (int c = 0; c < COUT; c++) {
            scale[c] = f32_to_fp16(ss * sj(rng));
            bias [c] = f32_to_fp16(bj(rng));
        }

        std::vector<int8_t> rdat(N_OUT * COUT, 0);
        std::vector<uint16_t> rscale(COUT, f32_to_fp16(0.0f)), rbias(COUT, f32_to_fp16(0.0f));
        if (RESIDUAL) {
            for (auto& v : rdat) v = (int8_t)r8(rng);
            for (int c = 0; c < COUT; c++) {
                rscale[c] = f32_to_fp16((float)S_OUT_SILU * sj(rng));
                rbias [c] = f32_to_fp16(bj(rng) * 0.2f);
            }
        }

        // Reference output via shadow
        auto shadow = shadow_conv(in, wt, scale, bias, rdat, rscale, rbias);

        struct Commit { int pix; int ct; };
        std::vector<Commit> inflight;
        inflight.reserve(N_OUT * N_COUT_TILE + TOTAL_LAT);
        int produced_dut = 0, produced_ref = 0;
        std::vector<int8_t> dut_out(N_OUT * COUT, 0);
        std::vector<int8_t> ref_out(N_OUT * COUT, 0);

        for (int step = 0; step < TOTAL_BEATS + TOTAL_LAT + 32; step++) {
            if (step < TOTAL_BEATS) {
                int pix = step / BEATS_PER_PIX;
                int rem = step % BEATS_PER_PIX;
                int ct  = rem / N_CIN_TILE;
                int cit = rem % N_CIN_TILE;
                int oy  = pix / OUT_W;
                int ox  = pix % OUT_W;

                int8_t xt[N_LANE_TILE];
                int8_t wt2[P_COUT * N_LANE_TILE];
                build_window_tile(in, oy, ox, cit, xt);
                build_weight_tile(wt, ct, cit, wt2);

                uint16_t sbuf[P_COUT], bbuf[P_COUT];
                uint16_t rsbuf[P_COUT], rbbuf[P_COUT];
                int8_t   rbuf[P_COUT];
                int co_base = ct * P_COUT;
                for (int c = 0; c < P_COUT; c++) {
                    int co = co_base + c;
                    if (co < COUT) {
                        sbuf[c]  = scale[co];
                        bbuf[c]  = bias [co];
                        rsbuf[c] = rscale[co];
                        rbbuf[c] = rbias [co];
                        rbuf[c]  = rdat[(oy*OUT_W+ox)*COUT + co];
                    } else {
                        sbuf[c]=bbuf[c]=rsbuf[c]=rbbuf[c]=0;
                        rbuf[c]=0;
                    }
                }
                bool first = (cit == 0);
                bool last  = (cit == N_CIN_TILE - 1);
                drive_beat(sim.dut.get(), xt, N_LANE_TILE, wt2, P_COUT*N_LANE_TILE,
                           sbuf, bbuf, rbuf, rsbuf, rbbuf, first, last, ct);
                if (last) inflight.push_back({pix, ct});
            } else {
                idle_beat(sim.dut.get());
            }
            sim.tick();

            if (sim.dut->dut_valid_o && produced_dut < (int)inflight.size()) {
                int pix = inflight[produced_dut].pix;
                int ct  = inflight[produced_dut].ct;
                int8_t row[P_COUT];
                unpack_bytes(wide_bytes(sim.dut->dut_y_flat_o), row, P_COUT);
                int co_base = ct * P_COUT;
                for (int c = 0; c < P_COUT; c++) {
                    int co = co_base + c;
                    if (co < COUT) dut_out[pix * COUT + co] = row[c];
                }
                produced_dut++;
            }
            if (sim.dut->ref_valid_o && produced_ref < (int)inflight.size()) {
                int pix = inflight[produced_ref].pix;
                int ct  = inflight[produced_ref].ct;
                int8_t row[P_COUT];
                unpack_bytes(wide_bytes(sim.dut->ref_y_flat_o), row, P_COUT);
                int co_base = ct * P_COUT;
                for (int c = 0; c < P_COUT; c++) {
                    int co = co_base + c;
                    if (co < COUT) ref_out[pix * COUT + co] = row[c];
                }
                produced_ref++;
            }
        }

        // Build float vectors for comparison.
        std::vector<float> dut_f(N_OUT * COUT), ref_f(N_OUT * COUT), sh_f(N_OUT * COUT);
        double scale_out = (RESIDUAL || SILU_EN) ? S_OUT_SILU : S_OUT_PRE;
        for (int i = 0; i < N_OUT * COUT; i++) {
            dut_f[i] = (float)((double)dut_out[i] * scale_out);
            ref_f[i] = (float)((double)ref_out[i] * scale_out);
            sh_f [i] = (float)((double)shadow [i] * scale_out);
        }
        // DUT vs REF must be bit-exact (they use the same sub-IPs and same math).
        int n_diff = 0;
        for (int i = 0; i < N_OUT * COUT; i++) if (dut_out[i] != ref_out[i]) n_diff++;

        Stats d_v_r = compare(dut_f, ref_f);
        Stats d_v_s = compare(dut_f, sh_f);

        printf("\n--- seed %d ---\n", seed);
        printf("  produced dut=%d ref=%d expected=%d\n", produced_dut, produced_ref,
               (int)inflight.size());
        printf("  DUT vs REF   : n_diff=%d max=%.4f mae=%.4f cos=%.6f\n",
               n_diff, d_v_r.max_abs, d_v_r.mae, d_v_r.cos);
        printf("  DUT vs shadow: max=%.4f mae=%.4f cos=%.6f\n",
               d_v_s.max_abs, d_v_s.mae, d_v_s.cos);

        bool pass = (n_diff == 0) && (d_v_s.cos > 0.999);
        printf("  => %s\n", pass ? "PASS" : "FAIL");
        sim.check(pass, std::string("config ") + CFG_NAME + " seed " + std::to_string(seed));
        if (pass) n_pass++;
    }

    printf("\n=== %s: %d/%d seeds passed ===\n", CFG_NAME, n_pass, N_SEEDS);
    return sim.finish();
}
