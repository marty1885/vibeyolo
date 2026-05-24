// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// conv_stage DV — bit-exact vs the validated conv_layer compute core.
//
// DUT (conv_stage): random zero-padded frame streamed in raster; outputs
//   collected from the ovalid/odata stream.
// REF (conv_layer): the SAME weights/scale/bias driven through the proven
//   software tile schedule (pixel-major, outer cout-tile, inner cin-tile) over
//   a zero-padded copy of the frame.
// Both share the conv_layer RTL, so the outputs must match bit-for-bit; any
// mismatch is a conv_stage sequencing/plumbing bug.

#include "Vconv_stage_tb.h"
#include "sim_ctrl.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <random>

// ── Config (overridable by -D from the Makefile) ──
#ifndef CFG_CIN
#define CFG_CIN 8
#endif
#ifndef CFG_COUT
#define CFG_COUT 8
#endif
#ifndef CFG_K
#define CFG_K 3
#endif
#ifndef CFG_STRIDE
#define CFG_STRIDE 1
#endif
#ifndef CFG_PAD
#define CFG_PAD 1
#endif
#ifndef CFG_H_IN
#define CFG_H_IN 6
#endif
#ifndef CFG_W_IN
#define CFG_W_IN 6
#endif
#ifndef CFG_P_COUT
#define CFG_P_COUT 4
#endif
#ifndef CFG_P_CIN
#define CFG_P_CIN 4
#endif
#ifndef CFG_SILU
#define CFG_SILU 1
#endif

static constexpr int CIN    = CFG_CIN;
static constexpr int COUT   = CFG_COUT;
static constexpr int K      = CFG_K;
static constexpr int STRIDE = CFG_STRIDE;
static constexpr int PAD    = CFG_PAD;
static constexpr int H_IN   = CFG_H_IN;
static constexpr int W_IN   = CFG_W_IN;
static constexpr int P_COUT = CFG_P_COUT;
static constexpr int P_CIN  = CFG_P_CIN;

static constexpr int N_LANE_FULL = K*K*CIN;
static constexpr int N_LANE_TILE = K*K*P_CIN;
static constexpr int N_CIN_TILE  = (CIN + P_CIN - 1) / P_CIN;
static constexpr int N_COUT_TILE = (COUT + P_COUT - 1) / P_COUT;
static constexpr int H_OUT = (H_IN + 2*PAD - K)/STRIDE + 1;
static constexpr int W_OUT = (W_IN + 2*PAD - K)/STRIDE + 1;
static constexpr int N_OUT = H_OUT * W_OUT;
static constexpr int PAD_H = H_IN + 2*PAD;
static constexpr int PAD_W = W_IN + 2*PAD;

// ── fp16 from float (RNE) for scale/bias generation ──
static uint16_t f32_to_fp16(float fv) {
    uint32_t f; std::memcpy(&f, &fv, 4);
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t  exp  = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFu;
    if (((f >> 23) & 0xFF) == 0xFF)
        return (uint16_t)(sign | (0x1Fu << 10) | (mant ? 0x200u : 0u));
    if (exp >= 31) return (uint16_t)(sign | (0x1Fu << 10));
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t m = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (m & 1))) m++;
        return (uint16_t)(sign | m);
    }
    uint32_t mant10 = mant >> 13;
    uint32_t rem = mant & 0x1FFFu, half = 0x1000u;
    if (rem > half || (rem == half && (mant10 & 1))) {
        mant10++;
        if (mant10 == 0x400u) { mant10 = 0; exp++; if (exp >= 31) return (uint16_t)(sign | (0x1Fu<<10)); }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant10);
}

template <typename T>
static void pack_bytes(T& dst, const int8_t* src, int nbytes) {
    auto* p = reinterpret_cast<uint32_t*>(&dst);
    int nwords = (nbytes + 3) / 4;
    for (int i = 0; i < nwords; i++) p[i] = 0;
    for (int i = 0; i < nbytes; i++)
        p[i/4] |= ((uint32_t)(uint8_t)src[i]) << ((i % 4) * 8);
}
template <typename T>
static void pack_u16(T& dst, const uint16_t* src, int n) {
    auto* p = reinterpret_cast<uint32_t*>(&dst);
    int nwords = (n + 1) / 2;
    for (int i = 0; i < nwords; i++) p[i] = 0;
    for (int i = 0; i < n; i++) p[i/2] |= ((uint32_t)src[i]) << ((i % 2) * 16);
}
template <typename T>
static void unpack_bytes(const T& src, int8_t* dst, int nbytes) {
    auto* p = reinterpret_cast<const uint32_t*>(&src);
    for (int i = 0; i < nbytes; i++)
        dst[i] = (int8_t)((p[i/4] >> ((i % 4) * 8)) & 0xFFu);
}

// build K*K*P_CIN window tile for output (oy,ox), cin-tile cit, from padded buf
static void build_window_tile(const std::vector<int8_t>& pad, int oy, int ox,
                              int cit, int8_t out[N_LANE_TILE]) {
    int kc_base = cit * P_CIN;
    for (int kh = 0; kh < K; kh++)
      for (int kw = 0; kw < K; kw++) {
        int h = oy*STRIDE + kh, w = ox*STRIDE + kw;
        for (int p = 0; p < P_CIN; p++) {
            int kc = kc_base + p;
            int lane = (kh*K + kw)*P_CIN + p;
            out[lane] = (kc < CIN) ? pad[(h*PAD_W + w)*CIN + kc] : 0;
        }
      }
}
// build P_COUT*(K*K*P_CIN) weight tile for (ct,cit) from w_full[oc][full_lane]
static void build_weight_tile(const std::vector<int8_t>& wf, int ct, int cit,
                              int8_t out[P_COUT*N_LANE_TILE]) {
    int kc_base = cit*P_CIN, co_base = ct*P_COUT;
    for (int co = 0; co < P_COUT; co++) {
        int ocg = co_base + co;
        for (int kh = 0; kh < K; kh++)
          for (int kw = 0; kw < K; kw++)
            for (int p = 0; p < P_CIN; p++) {
                int kc = kc_base + p;
                int tl = (kh*K + kw)*P_CIN + p;
                int8_t v = 0;
                if (ocg < COUT && kc < CIN)
                    v = wf[ocg*N_LANE_FULL + (kh*K + kw)*CIN + kc];
                out[co*N_LANE_TILE + tl] = v;
            }
    }
}

int main(int argc, char** argv) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> d8(-100, 100);

    // ── weights / scale / bias ──
    std::vector<int8_t> w_full(COUT * N_LANE_FULL);
    for (auto& v : w_full) v = (int8_t)d8(rng);
    std::vector<uint16_t> scale(COUT), bias(COUT);
    std::uniform_real_distribution<float> ds(0.005f, 0.05f), db(-0.5f, 0.5f);
    for (int c = 0; c < COUT; c++) { scale[c] = f32_to_fp16(ds(rng)); bias[c] = f32_to_fp16(db(rng)); }

    // write ROM hex for conv_stage $readmemh (relative to run dir)
    FILE* fw = fopen("cs_w.hex", "w");
    for (int i = 0; i < COUT*N_LANE_FULL; i++) fprintf(fw, "%02x\n", (uint8_t)w_full[i]);
    fclose(fw);
    FILE* fs = fopen("cs_s.hex", "w");
    for (int c = 0; c < COUT; c++) fprintf(fs, "%04x\n", scale[c]);
    fclose(fs);
    FILE* fb = fopen("cs_b.hex", "w");
    for (int c = 0; c < COUT; c++) fprintf(fb, "%04x\n", bias[c]);
    fclose(fb);

    // ── random frame (unpadded) + padded copy for the ref ──
    std::vector<int8_t> frame(H_IN * W_IN * CIN);
    for (auto& v : frame) v = (int8_t)d8(rng);
    std::vector<int8_t> pad(PAD_H * PAD_W * CIN, 0);
    for (int r = 0; r < H_IN; r++)
      for (int c = 0; c < W_IN; c++)
        for (int ch = 0; ch < CIN; ch++)
          pad[((r+PAD)*PAD_W + (c+PAD))*CIN + ch] = frame[(r*W_IN + c)*CIN + ch];

    SimCtrl<Vconv_stage_tb> sim(argc, argv);
    sim.max_time = 4000000;
    auto* d = sim.dut.get();
    d->start_i = 0; d->ivalid_i = 0; d->oready_i = 1;
    d->ref_valid_i = 0; d->ref_first_i = 0; d->ref_last_i = 0; d->ref_ct_i = 0;
    sim.reset();

    // ════════ Phase 1: stream frame through conv_stage (DUT) ════════
    d->start_i = 1; sim.tick(); d->start_i = 0;

    std::vector<int8_t> dut_out(N_OUT * COUT, 0);
    int in_idx = 0, out_idx = 0;
    int guard = N_OUT * N_COUT_TILE * N_CIN_TILE * 4 + H_IN*W_IN*8 + 2000;
    for (int step = 0; step < guard && out_idx < N_OUT; step++) {
        // drive next input pixel
        if (in_idx < H_IN*W_IN) {
            int8_t px[CIN];
            for (int ch = 0; ch < CIN; ch++) px[ch] = frame[in_idx*CIN + ch];
            pack_bytes(d->idata_flat_i, px, CIN);
            d->ivalid_i = 1;
        } else {
            d->ivalid_i = 0;
        }
        d->oready_i = 1;
        // sample iready/ovalid (combinational, valid pre-edge)
        d->eval();
        bool accept = d->ivalid_i && d->iready_o;
        bool ovld   = d->ovalid_o;
        int8_t row[COUT];
        if (ovld) unpack_bytes(d->odata_flat_o, row, COUT);
        sim.tick();
        if (accept) in_idx++;
        if (ovld && out_idx < N_OUT) {
            for (int c = 0; c < COUT; c++) dut_out[out_idx*COUT + c] = row[c];
            out_idx++;
        }
    }
    sim.check(out_idx == N_OUT, "DUT produced N_OUT pixels (got " +
              std::to_string(out_idx) + "/" + std::to_string(N_OUT) + ")");
    d->ivalid_i = 0;

    // ════════ Phase 2: drive ref conv_layer with the proven schedule ════════
    std::vector<int8_t> ref_out(N_OUT * COUT, 0);
    struct IF { int pix, ct; };
    std::vector<IF> inflight;
    int TOTAL_BEATS = N_OUT * N_COUT_TILE * N_CIN_TILE;
    int produced = 0;
    int REF_LAT = 1 + 32 + 16; // generous drain margin
    for (int step = 0; step < TOTAL_BEATS + REF_LAT + 64; step++) {
        if (step < TOTAL_BEATS) {
            int pix = step / (N_COUT_TILE*N_CIN_TILE);
            int rem = step % (N_COUT_TILE*N_CIN_TILE);
            int ct  = rem / N_CIN_TILE;
            int cit = rem % N_CIN_TILE;
            int oy = pix / W_OUT, ox = pix % W_OUT;
            int8_t win[N_LANE_TILE];  build_window_tile(pad, oy, ox, cit, win);
            pack_bytes(d->ref_x_flat_i, win, N_LANE_TILE);
            int8_t wt[P_COUT*N_LANE_TILE]; build_weight_tile(w_full, ct, cit, wt);
            pack_bytes(d->ref_w_flat_i, wt, P_COUT*N_LANE_TILE);
            uint16_t sb[P_COUT], bb[P_COUT];
            for (int c = 0; c < P_COUT; c++) {
                int oc = ct*P_COUT + c;
                sb[c] = (oc < COUT) ? scale[oc] : 0;
                bb[c] = (oc < COUT) ? bias[oc]  : 0;
            }
            pack_u16(d->ref_scale_flat_i, sb, P_COUT);
            pack_u16(d->ref_bias_flat_i,  bb, P_COUT);
            d->ref_valid_i = 1;
            d->ref_first_i = (cit == 0);
            d->ref_last_i  = (cit == N_CIN_TILE - 1);
            d->ref_ct_i    = ct;
            if (cit == N_CIN_TILE - 1) inflight.push_back({pix, ct});
        } else {
            d->ref_valid_i = 0; d->ref_first_i = 0; d->ref_last_i = 0;
        }
        d->eval();
        bool rvld = d->ref_valid_o;
        int  rct  = d->ref_ct_o;
        int8_t row[P_COUT];
        if (rvld) unpack_bytes(d->ref_y_flat_o, row, P_COUT);
        sim.tick();
        if (rvld && produced < (int)inflight.size()) {
            int pix = inflight[produced].pix, ct = inflight[produced].ct;
            sim.check(rct == ct, "ref cout_tile order");
            for (int c = 0; c < P_COUT; c++) {
                int oc = ct*P_COUT + c;
                if (oc < COUT) ref_out[pix*COUT + oc] = row[c];
            }
            produced++;
        }
    }
    sim.check(produced == N_OUT*N_COUT_TILE, "ref produced all commits");

    // ════════ Compare bit-exact ════════
    int mism = 0;
    for (int i = 0; i < N_OUT*COUT; i++)
        if (dut_out[i] != ref_out[i]) {
            if (mism < 8)
                printf("  mismatch pix=%d ch=%d dut=%d ref=%d\n",
                       i/COUT, i%COUT, dut_out[i], ref_out[i]);
            mism++;
        }
    sim.check(mism == 0, "conv_stage == conv_layer bit-exact (" +
              std::to_string(mism) + " mismatches of " +
              std::to_string(N_OUT*COUT) + ")");

    printf("cfg CIN=%d COUT=%d K=%d S=%d HxW=%dx%d P_COUT=%d P_CIN=%d -> out %dx%d\n",
           CIN, COUT, K, STRIDE, H_IN, W_IN, P_COUT, P_CIN, H_OUT, W_OUT);
    return sim.finish();
}
