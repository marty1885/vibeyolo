// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// upsample_concat_test.cc — drives the model.14/15 upsample_concat block
// with stim/<sample>.a_i8.hex + stim/<sample>.b_i8.hex, captures the DUT
// output stream, and compares to:
//   1. Software int8 golden in stim/<sample>.golden_i8.hex (bit-exact).
//   2. Float ORT reference in stim/<sample>.ref_ort.f32.hex over the
//      full 80x80 output (cosine >= 0.998 target on random tiles).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include "Vupsample_concat_tb.h"
#include "sim_ctrl.h"

static constexpr int H_A = 40;
static constexpr int W_A = 40;
static constexpr int C_A = 128;
static constexpr int H_B = 80;
static constexpr int W_B = 80;
static constexpr int C_B = 128;
static constexpr int H_O = H_B;
static constexpr int W_O = W_B;
static constexpr int C_O = C_A + C_B;

static const char* STIM_DIR =
    "/home/marty/Documents/aif/vibeyolo/integ/upsample_model14/stim";

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
    if (v.size() != expected) {
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
    if (v.size() != expected) {
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

struct SampleResult {
    int    bit_exact_bad = 0;
    double cos_sw        = 0.0;
    double cos_ort       = 0.0;
};

static SampleResult drive_sample(SimCtrl<Vupsample_concat_tb>& sim,
                                 const std::string& name,
                                 double s_out) {
    SampleResult r{};
    auto a_in  = load_i8_hex (std::string(STIM_DIR) + "/" + name + ".a_i8.hex",
                              size_t(H_A) * W_A * C_A);
    auto b_in  = load_i8_hex (std::string(STIM_DIR) + "/" + name + ".b_i8.hex",
                              size_t(H_B) * W_B * C_B);
    auto golden = load_i8_hex (std::string(STIM_DIR) + "/" + name + ".golden_i8.hex",
                               size_t(H_O) * W_O * C_O);
    auto ref_ort = load_f32_hex(std::string(STIM_DIR) + "/" + name + ".ref_ort.f32.hex",
                                size_t(H_O) * W_O * C_O);

    sim.dut->start_i  = 0;
    sim.dut->avalid_i = 0;
    sim.dut->bvalid_i = 0;
    sim.dut->oready_i = 0;
    int8_t zero_a[C_A]; std::memset(zero_a, 0, sizeof(zero_a));
    int8_t zero_b[C_B]; std::memset(zero_b, 0, sizeof(zero_b));
    pack_lanes(&sim.dut->adata_i, zero_a, C_A);
    pack_lanes(&sim.dut->bdata_i, zero_b, C_B);
    sim.tick();

    sim.dut->start_i = 1;
    sim.tick();
    sim.dut->start_i = 0;

    const int total_a = H_A * W_A;
    const int total_b = H_B * W_B;
    const int total_o = H_O * W_O;

    int a_idx = 0, b_idx = 0, o_idx = 0;
    std::vector<int8_t> dut_out(size_t(total_o) * C_O, 0);

    const int max_cycles = 5'000'000;
    int cycles = 0;
    int8_t out_buf[C_O];

    while (o_idx < total_o) {
        if (a_idx < total_a) {
            sim.dut->avalid_i = 1;
            pack_lanes(&sim.dut->adata_i, &a_in[a_idx * C_A], C_A);
        } else {
            sim.dut->avalid_i = 0;
            pack_lanes(&sim.dut->adata_i, zero_a, C_A);
        }
        if (b_idx < total_b) {
            sim.dut->bvalid_i = 1;
            pack_lanes(&sim.dut->bdata_i, &b_in[b_idx * C_B], C_B);
        } else {
            sim.dut->bvalid_i = 0;
            pack_lanes(&sim.dut->bdata_i, zero_b, C_B);
        }
        sim.dut->oready_i = 1;

        sim.dut->eval();

        bool a_fire = sim.dut->avalid_i && sim.dut->aready_o;
        bool b_fire = sim.dut->bvalid_i && sim.dut->bready_o;
        bool o_fire = sim.dut->ovalid_o && sim.dut->oready_i;

        if (o_fire) {
            unpack_lanes(&sim.dut->odata_o, out_buf, C_O);
            std::memcpy(&dut_out[size_t(o_idx) * C_O], out_buf, C_O);
            o_idx++;
        }

        sim.tick();
        if (a_fire) a_idx++;
        if (b_fire) b_idx++;

        cycles++;
        if (cycles > max_cycles) {
            printf("  %s: TIMEOUT a=%d/%d b=%d/%d o=%d/%d\n",
                   name.c_str(), a_idx, total_a, b_idx, total_b,
                   o_idx, total_o);
            sim.check(false, name + " timeout");
            return r;
        }
    }

    sim.dut->avalid_i = 0;
    sim.dut->bvalid_i = 0;
    sim.dut->oready_i = 0;
    pack_lanes(&sim.dut->adata_i, zero_a, C_A);
    pack_lanes(&sim.dut->bdata_i, zero_b, C_B);
    sim.dut->eval();

    int bad_a = 0, bad_b = 0;
    for (size_t i = 0; i < dut_out.size(); i++) {
        if (dut_out[i] != golden[i]) {
            size_t pix = i / C_O;
            size_t ch  = i % C_O;
            if (ch < (size_t)C_A) bad_a++; else bad_b++;
            if (r.bit_exact_bad < 8) {
                int hh = int(pix / W_O);
                int ww = int(pix % W_O);
                printf("  %s: bit-exact diff @ (h=%d w=%d ch=%zu) got=%d exp=%d\n",
                       name.c_str(), hh, ww, ch,
                       int(dut_out[i]), int(golden[i]));
            }
            r.bit_exact_bad++;
        }
    }
    printf("  %s:   by stream a=%d b=%d\n", name.c_str(), bad_a, bad_b);

    auto cos_vs_f32 = [&](const float* ref) {
        long double dot = 0, na = 0, nb = 0;
        for (int h = 0; h < H_O; h++) {
            for (int w = 0; w < W_O; w++) {
                for (int c = 0; c < C_O; c++) {
                    int dut_i8 = dut_out[(size_t(h) * W_O + w) * C_O + c];
                    double aa = double(dut_i8) * s_out;
                    double bb = ref[(size_t(h) * W_O + w) * C_O + c];
                    dot += aa * bb;
                    na  += aa * aa;
                    nb  += bb * bb;
                }
            }
        }
        return double(dot / (std::sqrt(double(na)) * std::sqrt(double(nb)) + 1e-30L));
    };

    {
        std::vector<float> golden_f(size_t(H_O) * W_O * C_O);
        for (size_t i = 0; i < golden_f.size(); i++) {
            golden_f[i] = float(golden[i]) * float(s_out);
        }
        r.cos_sw = cos_vs_f32(golden_f.data());
    }
    r.cos_ort = cos_vs_f32(ref_ort.data());

    return r;
}

static double load_s_out_for(const std::string& name) {
    std::ifstream in(std::string(STIM_DIR) + "/manifest.json");
    if (!in) { fprintf(stderr, "missing manifest.json\n"); std::exit(2); }
    std::string txt((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    std::string key = "\"name\": \"" + name + "\"";
    auto npos = txt.find(key);
    if (npos == std::string::npos) {
        fprintf(stderr, "no entry for sample %s in manifest.json\n", name.c_str());
        std::exit(2);
    }
    auto spos = txt.find("\"s_out\"", npos);
    if (spos == std::string::npos) {
        fprintf(stderr, "no s_out after sample %s\n", name.c_str());
        std::exit(2);
    }
    auto colon = txt.find(':', spos);
    auto comma = txt.find_first_of(",}\n", colon + 1);
    return std::stod(txt.substr(colon + 1, comma - colon - 1));
}

// Chain mode (see model.11 copy): external A/B int8 frames -> real
// upsample_concat IP -> C_O int8 frame for tools/e2e/chain.py.
static int chain_mode(SimCtrl<Vupsample_concat_tb>& sim, const char* ap,
                      const char* bp, const char* op) {
    auto a_in = load_i8_hex(ap, size_t(H_A) * W_A * C_A);
    auto b_in = load_i8_hex(bp, size_t(H_B) * W_B * C_B);
    sim.reset();
    sim.dut->start_i = 0; sim.dut->avalid_i = 0; sim.dut->bvalid_i = 0;
    sim.dut->oready_i = 0;
    int8_t za[C_A]; std::memset(za, 0, sizeof(za));
    int8_t zb[C_B]; std::memset(zb, 0, sizeof(zb));
    pack_lanes(&sim.dut->adata_i, za, C_A); pack_lanes(&sim.dut->bdata_i, zb, C_B);
    sim.tick();
    sim.dut->start_i = 1; sim.tick(); sim.dut->start_i = 0;
    const int ta = H_A * W_A, tb = H_B * W_B, to = H_O * W_O;
    std::vector<int8_t> out(size_t(to) * C_O, 0);
    int ai = 0, bi = 0, oi = 0; int8_t buf[C_O];
    while (oi < to) {
        if (ai < ta) { sim.dut->avalid_i = 1; pack_lanes(&sim.dut->adata_i, &a_in[ai * C_A], C_A); }
        else { sim.dut->avalid_i = 0; pack_lanes(&sim.dut->adata_i, za, C_A); }
        if (bi < tb) { sim.dut->bvalid_i = 1; pack_lanes(&sim.dut->bdata_i, &b_in[bi * C_B], C_B); }
        else { sim.dut->bvalid_i = 0; pack_lanes(&sim.dut->bdata_i, zb, C_B); }
        sim.dut->oready_i = 1; sim.dut->eval();
        bool af = sim.dut->avalid_i && sim.dut->aready_o;
        bool bf = sim.dut->bvalid_i && sim.dut->bready_o;
        bool of = sim.dut->ovalid_o && sim.dut->oready_i;
        if (of) { unpack_lanes(&sim.dut->odata_o, buf, C_O);
            std::memcpy(&out[size_t(oi) * C_O], buf, C_O); oi++; }
        sim.tick(); if (af) ai++; if (bf) bi++;
    }
    FILE* f = fopen(op, "w");
    for (auto b : out) fprintf(f, "%02x\n", (unsigned)(uint8_t)b);
    fclose(f);
    printf("upsample chain: a=%d b=%d o=%d -> %s\n", ai, bi, oi, op);
    return 0;
}

int main(int argc, char** argv) {
    SimCtrl<Vupsample_concat_tb> sim(argc, argv);
    sim.max_time = 4'000'000'000ull;
    if (const char* ca = getenv("CHAIN_IN_A"))
        return chain_mode(sim, ca, getenv("CHAIN_IN_B"), getenv("CHAIN_OUT"));

    printf("upsample_concat (model.14/15) test (A=%dx%dx%d  B=%dx%dx%d  O=%dx%dx%d)\n",
           C_A, H_A, W_A, C_B, H_B, W_B, C_O, H_O, W_O);

    sim.dut->start_i  = 0;
    sim.dut->avalid_i = 0;
    sim.dut->bvalid_i = 0;
    sim.dut->oready_i = 0;
    sim.reset();

    const std::vector<std::string> samples = {
        "rand0", "rand1", "rand2", "half", "gradient",
    };

    int passed = 0;
    for (auto& nm : samples) {
        double s_out = load_s_out_for(nm);
        printf("  --- sample %s  S_OUT=%.6f ---\n", nm.c_str(), s_out);
        auto r = drive_sample(sim, nm, s_out);
        printf("  [%s] bit_exact_bad=%d  cos(DUT vs SW)=%.6f  cos(DUT vs ORT)=%.6f\n",
               nm.c_str(), r.bit_exact_bad, r.cos_sw, r.cos_ort);
        sim.check(r.bit_exact_bad == 0,
                  nm + ": DUT bit-exact match against SW int8 golden");
        const bool is_random_tile = (nm.rfind("rand", 0) == 0);
        if (is_random_tile) {
            sim.check(r.cos_ort >= 0.998,
                      nm + ": cos(DUT vs ORT float) >= 0.998");
        } else {
            sim.check(r.cos_ort >= 0.990,
                      nm + ": cos(DUT vs ORT float) >= 0.990 (non-random tile)");
        }
        if (r.bit_exact_bad == 0 &&
            r.cos_ort >= (is_random_tile ? 0.998 : 0.990)) passed++;
    }
    printf("samples passed: %d / %zu\n", passed, samples.size());

    return sim.finish();
}
