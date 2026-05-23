// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// attn_test.cc — drives the attention block with stim/<sample>.{qkv,pe}_i8.hex,
// captures the DUT's 128ch×20×20 int8 ATTN_OUT, and compares to:
//   1. Software int8 golden (stim/<sample>.golden_attn_i8.hex): the fp16-ish
//      reference; cos check is the gating criterion (DUT is fp16, ref fp64).
//   2. Float ORT reference (stim/<sample>.ref_attn_f32.hex): cos ≥ 0.99.
// ATTN_OUT = attention(qkv) + pe, dequantised at S_AOUT (from manifest.json).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include "Vattn_tb.h"
#include "sim_ctrl.h"

static constexpr int H        = 20;
static constexpr int W        = 20;
static constexpr int C_QKV    = 256;
static constexpr int C_FE     = 128;

static const char* STIM_DIR =
    "/home/marty/Documents/aif/vibeyolo/integ/attn_model10/stim";


static std::vector<int8_t> load_i8_hex(const std::string& path, size_t expected) {
    std::ifstream in(path);
    if (!in) { fprintf(stderr, "could not open %s\n", path.c_str()); std::exit(2); }
    std::vector<int8_t> v; v.reserve(expected);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        unsigned val;
        std::stringstream ss; ss << std::hex << line; ss >> val;
        v.push_back(static_cast<int8_t>(val & 0xFF));
    }
    if (expected && v.size() != expected) {
        fprintf(stderr, "size mismatch in %s: got %zu, expected %zu\n",
                path.c_str(), v.size(), expected);
        std::exit(2);
    }
    return v;
}

static std::vector<float> load_f32_hex(const std::string& path, size_t expected) {
    std::ifstream in(path);
    if (!in) { fprintf(stderr, "could not open %s\n", path.c_str()); std::exit(2); }
    std::vector<float> v; v.reserve(expected);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        uint32_t bits;
        std::stringstream ss; ss << std::hex << line; ss >> bits;
        float f; std::memcpy(&f, &bits, 4);
        v.push_back(f);
    }
    if (expected && v.size() != expected) {
        fprintf(stderr, "size mismatch in %s: got %zu, expected %zu\n",
                path.c_str(), v.size(), expected);
        std::exit(2);
    }
    return v;
}

template <typename DataT>
static void pack_lanes(DataT* bus, const int8_t* lanes, int n) {
    uint8_t* p = reinterpret_cast<uint8_t*>(bus);
    std::memset(bus, 0, sizeof(DataT));
    for (int c = 0; c < n; c++) p[c] = static_cast<uint8_t>(lanes[c]);
}

template <typename DataT>
static void unpack_lanes(const DataT* bus, int8_t* lanes, int n) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(bus);
    for (int c = 0; c < n; c++) lanes[c] = static_cast<int8_t>(p[c]);
}

// Drive one frame: feed qkv (256ch) then pe (128ch) raster, collect
// attn_out (128ch). Returns the captured int8 output frame.
static std::vector<int8_t> drive(SimCtrl<Vattn_tb>& sim,
                                 const std::vector<int8_t>& qkv_in,
                                 const std::vector<int8_t>& pe_in) {
    const size_t N = size_t(H) * W;
    int8_t zero_qkv[C_QKV] = {0}; int8_t zero_fe[C_FE] = {0};

    sim.dut->start_i = sim.dut->qkv_valid_i = sim.dut->pe_valid_i = 0;
    sim.dut->oready_i = 0;
    pack_lanes(&sim.dut->qkv_data_i, zero_qkv, C_QKV);
    pack_lanes(&sim.dut->pe_data_i,  zero_fe,  C_FE);
    sim.tick();
    sim.dut->start_i = 1; sim.tick(); sim.dut->start_i = 0;

    size_t qi = 0, pi = 0, oi = 0;
    std::vector<int8_t> dut_out(N * C_FE, 0); int8_t out_buf[C_FE];
    const int max_cycles = 50'000'000; int cycles = 0;
    while (oi < N) {
        sim.dut->qkv_valid_i = qi < N;
        pack_lanes(&sim.dut->qkv_data_i, qi < N ? &qkv_in[qi*C_QKV] : zero_qkv, C_QKV);
        sim.dut->pe_valid_i = pi < N;
        pack_lanes(&sim.dut->pe_data_i, pi < N ? &pe_in[pi*C_FE] : zero_fe, C_FE);
        sim.dut->oready_i = 1;
        sim.dut->eval();
        bool qf = sim.dut->qkv_valid_i && sim.dut->qkv_ready_o;
        bool pf = sim.dut->pe_valid_i  && sim.dut->pe_ready_o;
        bool of = sim.dut->ovalid_o    && sim.dut->oready_i;
        if (of) { unpack_lanes(&sim.dut->odata_o, out_buf, C_FE);
                  std::memcpy(&dut_out[oi*C_FE], out_buf, C_FE); oi++; }
        sim.tick();
        if (qf) qi++; if (pf) pi++;
        if (++cycles > max_cycles) { fprintf(stderr, "attn TIMEOUT o=%zu\n", oi); break; }
    }
    return dut_out;
}

static double load_real_from_manifest(const std::string& key) {
    std::ifstream in(std::string(STIM_DIR) + "/manifest.json");
    if (!in) { fprintf(stderr, "missing manifest.json\n"); std::exit(2); }
    std::string txt((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::string k = "\"" + key + "\"";
    auto pos = txt.find(k);
    if (pos == std::string::npos) { fprintf(stderr, "no key %s\n", key.c_str()); std::exit(2); }
    auto colon = txt.find(':', pos);
    auto comma = txt.find_first_of(",}\n", colon + 1);
    return std::stod(txt.substr(colon + 1, comma - colon - 1));
}

// ── chain mode ──
// Driven by tools/e2e/chain.py to run the REAL attn IP inside the end-to-end
// chip chain on a live image. Reads qkv + pe int8 (pix-major, channel-fastest)
// from explicit paths, drives one frame, dumps the 128ch int8 ATTN_OUT.
// Scales are baked at elaboration (attn_scales_pkg) — the chip's fixed config —
// so chain.py re-quantizes qkv/pe at those scales (read from manifest).
//   env: CHAIN_QKV, CHAIN_PE (int8 pix-major hex), CHAIN_OUT (int8 hex)
static int chain_mode(int argc, char** argv) {
    SimCtrl<Vattn_tb> sim(argc, argv);
    sim.max_time = 8'000'000'000ull;
    const size_t N = size_t(H) * W;
    sim.dut->start_i = sim.dut->qkv_valid_i = sim.dut->pe_valid_i = sim.dut->oready_i = 0;
    sim.reset();
    auto qkv_in = load_i8_hex(getenv("CHAIN_QKV"), N * C_QKV);
    auto pe_in  = load_i8_hex(getenv("CHAIN_PE"),  N * C_FE);
    auto out = drive(sim, qkv_in, pe_in);
    FILE* f = fopen(getenv("CHAIN_OUT"), "w");
    for (size_t i = 0; i < out.size(); i++) fprintf(f, "%02x\n", int(out[i]) & 0xFF);
    fclose(f);
    fprintf(stderr, "chain attn: %zu px -> %s\n", N, getenv("CHAIN_OUT"));
    return 0;
}

int main(int argc, char** argv) {
    if (getenv("CHAIN_OUT")) return chain_mode(argc, argv);
    SimCtrl<Vattn_tb> sim(argc, argv);
    sim.max_time = 8'000'000'000ull;
    printf("attn test (H=%d W=%d C_QKV=%d C_FE=%d)\n", H, W, C_QKV, C_FE);

    sim.dut->start_i = sim.dut->qkv_valid_i = sim.dut->pe_valid_i = sim.dut->oready_i = 0;
    sim.reset();

    double s_aout = load_real_from_manifest("S_AOUT");
    printf("  S_AOUT = %.6f\n", s_aout);

    const std::vector<std::string> samples = {"rand0","rand1","rand2","half","gradient"};
    const size_t N = size_t(H) * W;
    int passed = 0;

    auto cos_vs = [&](const std::vector<int8_t>& dut, const float* ref) {
        long double dot=0, na=0, nb=0;
        for (size_t i=0;i<N*C_FE;i++){ double a=double(dut[i])*s_aout, b=double(ref[i]);
            dot+=a*b; na+=a*a; nb+=b*b; }
        return double(dot/(std::sqrt(double(na))*std::sqrt(double(nb))+1e-30L));
    };

    for (auto& nm : samples) {
        auto qkv_in = load_i8_hex (std::string(STIM_DIR)+"/"+nm+".qkv_i8.hex", N*C_QKV);
        auto pe_in  = load_i8_hex (std::string(STIM_DIR)+"/"+nm+".pe_i8.hex",  N*C_FE);
        auto golden = load_i8_hex (std::string(STIM_DIR)+"/"+nm+".golden_attn_i8.hex", N*C_FE);
        auto ref    = load_f32_hex(std::string(STIM_DIR)+"/"+nm+".ref_attn_f32.hex",  N*C_FE);

        auto dut = drive(sim, qkv_in, pe_in);

        int bad = 0;
        for (size_t i=0;i<dut.size();i++) if (dut[i]!=golden[i]) {
            if (bad<4){ size_t pix=i/C_FE,ch=i%C_FE;
                printf("  %s: i8 diff @ (h=%zu w=%zu ch=%zu) got=%d exp=%d\n",
                       nm.c_str(), pix/W, pix%W, ch, int(dut[i]), int(golden[i])); }
            bad++;
        }
        std::vector<float> golden_f(N*C_FE);
        for (size_t i=0;i<golden_f.size();i++) golden_f[i]=float(golden[i])*float(s_aout);
        double cos_sw  = cos_vs(dut, golden_f.data());
        double cos_ort = cos_vs(dut, ref.data());
        printf("  [%s] bit_exact_bad=%d  cos(DUT vs SW)=%.6f  cos(DUT vs ORT)=%.6f\n",
               nm.c_str(), bad, cos_sw, cos_ort);
        const bool rnd = (nm.rfind("rand",0)==0);
        sim.check(cos_ort >= (rnd ? 0.99 : 0.95), nm + ": cos(DUT vs ORT) ok");
        sim.check(cos_sw  >= 0.99, nm + ": cos(DUT vs SW int8 golden) >= 0.99");
        if (cos_ort >= (rnd ? 0.99 : 0.95)) passed++;
    }
    printf("samples passed: %d / %zu\n", passed, samples.size());
    return sim.finish();
}
