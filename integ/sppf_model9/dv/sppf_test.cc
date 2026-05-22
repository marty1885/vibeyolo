// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// sppf_test.cc — drives the SPPF block with stim/<sample>.input_i8.hex,
// captures the DUT output stream, and compares to:
//   1. The software int8 golden in stim/<sample>.golden_i8.hex (bit-exact).
//   2. The float ORT reference in stim/<sample>.ref_ort.f32.hex on the
//      inner ROI (cosine >= 0.998 target).
//
// The sample list is fixed (matches extract.py); manifest.json carries
// metadata if needed but we just use hard-coded shape constants here.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include "Vsppf_tb.h"
#include "sim_ctrl.h"

static constexpr int H = 20;
static constexpr int W = 20;
static constexpr int C = 128;
static constexpr int K = 5;
static constexpr int ROI_LO = 6;
static constexpr int ROI_HI = 14;
static constexpr int ROI_H  = ROI_HI - ROI_LO;
static constexpr int ROI_W  = ROI_HI - ROI_LO;
static constexpr int OUT_C  = 4 * C;

static const char* STIM_DIR =
    "/home/marty/Documents/aif/vibeyolo/integ/sppf_model9/stim";

static std::vector<int8_t> load_i8_hex(const std::string& path, size_t expected) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "could not open %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<int8_t> v;
    v.reserve(expected);
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
    if (!in) {
        fprintf(stderr, "could not open %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<float> v;
    v.reserve(expected);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        uint32_t bits;
        std::stringstream ss; ss << std::hex << line; ss >> bits;
        float f;
        std::memcpy(&f, &bits, 4);
        v.push_back(f);
    }
    if (v.size() != expected) {
        fprintf(stderr, "size mismatch in %s: got %zu, expected %zu\n",
                path.c_str(), v.size(), expected);
        std::exit(2);
    }
    return v;
}

// Pack C int8 lanes into the DUT's wide idata_i bus. Verilator lays bytes
// out little-endian-by-byte; matches linebuf test pattern.
static void pack_in(Vsppf_tb* dut, const int8_t* lanes) {
    uint8_t* p = reinterpret_cast<uint8_t*>(&dut->idata_i);
    std::memset(&dut->idata_i, 0, sizeof(dut->idata_i));
    for (int c = 0; c < C; c++) {
        p[c] = static_cast<uint8_t>(lanes[c]);
    }
}

// Read OUT_C int8 lanes from the DUT's odata_o bus.
static void unpack_out(const Vsppf_tb* dut, int8_t* lanes) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&dut->odata_o);
    for (int c = 0; c < OUT_C; c++) {
        lanes[c] = static_cast<int8_t>(p[c]);
    }
}

struct SampleResult {
    int bit_exact_bad = 0;
    double cos_sw   = 0.0;
    double cos_ort  = 0.0;
    double s_in     = 0.0;
};

static SampleResult drive_sample(SimCtrl<Vsppf_tb>& sim,
                                 const std::string& name,
                                 double s_in) {
    SampleResult r{}; r.s_in = s_in;
    auto input  = load_i8_hex(std::string(STIM_DIR) + "/" + name + ".input_i8.hex",
                              size_t(H) * W * C);
    auto golden = load_i8_hex(std::string(STIM_DIR) + "/" + name + ".golden_i8.hex",
                              size_t(H) * W * OUT_C);
    auto ref_ort = load_f32_hex(std::string(STIM_DIR) + "/" + name + ".ref_ort.f32.hex",
                                size_t(ROI_H) * ROI_W * OUT_C);

    // Reset DUT inputs.
    sim.dut->start_i  = 0;
    sim.dut->ivalid_i = 0;
    sim.dut->oready_i = 0;
    int8_t zero[C]; std::memset(zero, 0, sizeof(zero));
    pack_in(sim.dut.get(), zero);
    sim.tick();

    // Pulse start.
    sim.dut->start_i = 1;
    sim.tick();
    sim.dut->start_i = 0;

    // Drive H*W input pixels, then drain H*W output pixels.
    int in_idx  = 0;
    int out_idx = 0;
    const int total = H * W;
    std::vector<int8_t> dut_out(size_t(total) * OUT_C, 0);

    const int max_cycles = 2000000;
    int cycles = 0;
    int8_t out_buf[OUT_C];
    int dbg_iready_seen = 0, dbg_ovalid_seen = 0;

    while (out_idx < total) {
        // Drive input handshake.
        if (in_idx < total) {
            sim.dut->ivalid_i = 1;
            pack_in(sim.dut.get(), &input[in_idx * C]);
        } else {
            sim.dut->ivalid_i = 0;
            pack_in(sim.dut.get(), zero);
        }
        sim.dut->oready_i = 1;

        sim.dut->eval();
        if (sim.dut->iready_o) dbg_iready_seen++;
        if (sim.dut->ovalid_o) dbg_ovalid_seen++;

        bool i_fire = sim.dut->ivalid_i && sim.dut->iready_o;
        bool o_fire = sim.dut->ovalid_o && sim.dut->oready_i;

        if (o_fire) {
            unpack_out(sim.dut.get(), out_buf);
            std::memcpy(&dut_out[size_t(out_idx) * OUT_C], out_buf, OUT_C);
            out_idx++;
        }

        sim.tick();
        if (i_fire) in_idx++;

        cycles++;
        if (cycles > max_cycles) {
            printf("  %s: TIMEOUT in=%d out=%d / %d (iready_seen=%d ovalid_seen=%d)\n",
                   name.c_str(), in_idx, out_idx, total,
                   dbg_iready_seen, dbg_ovalid_seen);
            sim.check(false, name + " timeout");
            return r;
        }
    }

    sim.dut->ivalid_i = 0;
    sim.dut->oready_i = 0;
    pack_in(sim.dut.get(), zero);
    sim.dut->eval();

    // Bit-exact compare vs SW golden over the entire frame.
    int bad_inner = 0, bad_cv1 = 0, bad_mp1 = 0, bad_mp2 = 0, bad_mp3 = 0;
    for (size_t i = 0; i < dut_out.size(); i++) {
        if (dut_out[i] != golden[i]) {
            size_t pix = i / OUT_C;
            size_t ch  = i % OUT_C;
            int hh = int(pix / W);
            int ww = int(pix % W);
            bool inner = (hh >= ROI_LO && hh < ROI_HI && ww >= ROI_LO && ww < ROI_HI);
            if (inner) bad_inner++;
            if      (ch <   C) bad_cv1++;
            else if (ch < 2*C) bad_mp1++;
            else if (ch < 3*C) bad_mp2++;
            else               bad_mp3++;
            if (r.bit_exact_bad < 8) {
                printf("  %s: bit-exact diff @ (h=%d w=%d ch=%zu) got=%d exp=%d\n",
                       name.c_str(), hh, ww, ch,
                       int(dut_out[i]), int(golden[i]));
            }
            r.bit_exact_bad++;
        }
    }
    printf("  %s:   by stream cv1=%d mp1=%d mp2=%d mp3=%d  inner_ROI_bad=%d\n",
           name.c_str(), bad_cv1, bad_mp1, bad_mp2, bad_mp3, bad_inner);

    // Cosine vs SW golden on the inner ROI (sanity: should be 1.0).
    // Cosine vs ORT float reference on the inner ROI (the real metric).
    auto cos_inner_vs_f32 = [&](const float* ref) {
        long double dot = 0, na = 0, nb = 0;
        for (int h = ROI_LO; h < ROI_HI; h++) {
            for (int w = ROI_LO; w < ROI_HI; w++) {
                int rel_h = h - ROI_LO;
                int rel_w = w - ROI_LO;
                for (int c = 0; c < OUT_C; c++) {
                    int dut_i8 = dut_out[(size_t(h) * W + w) * OUT_C + c];
                    double a = double(dut_i8) * s_in;
                    double b = ref[(size_t(rel_h) * ROI_W + rel_w) * OUT_C + c];
                    dot += a * b;
                    na  += a * a;
                    nb  += b * b;
                }
            }
        }
        return double(dot / (std::sqrt(double(na)) * std::sqrt(double(nb)) + 1e-30L));
    };

    // SW golden cosine (in float dequantized form) — should match bit-exact.
    {
        std::vector<float> golden_f(size_t(ROI_H) * ROI_W * OUT_C);
        for (int h = ROI_LO; h < ROI_HI; h++) {
            for (int w = ROI_LO; w < ROI_HI; w++) {
                int rel_h = h - ROI_LO;
                int rel_w = w - ROI_LO;
                for (int c = 0; c < OUT_C; c++) {
                    golden_f[(size_t(rel_h) * ROI_W + rel_w) * OUT_C + c] =
                        float(golden[(size_t(h) * W + w) * OUT_C + c]) * float(s_in);
                }
            }
        }
        r.cos_sw = cos_inner_vs_f32(golden_f.data());
    }
    r.cos_ort = cos_inner_vs_f32(ref_ort.data());

    return r;
}

// Read S_IN from manifest.json (tiny scan, just look for "s_in").
static double load_s_in() {
    std::ifstream in(std::string(STIM_DIR) + "/manifest.json");
    if (!in) { fprintf(stderr, "missing manifest.json\n"); std::exit(2); }
    std::string txt((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    auto pos = txt.find("\"s_in\"");
    if (pos == std::string::npos) { fprintf(stderr, "no s_in key\n"); std::exit(2); }
    auto colon = txt.find(':', pos);
    auto comma = txt.find_first_of(",}\n", colon + 1);
    return std::stod(txt.substr(colon + 1, comma - colon - 1));
}

int main(int argc, char** argv) {
    SimCtrl<Vsppf_tb> sim(argc, argv);
    sim.max_time = 2'000'000'000ull;

    printf("sppf test (H=%d W=%d C=%d K=%d ROI=%dx%d)\n",
           H, W, C, K, ROI_H, ROI_W);

    double s_in = load_s_in();
    printf("S_IN = %.6f\n", s_in);

    sim.dut->start_i  = 0;
    sim.dut->ivalid_i = 0;
    sim.dut->oready_i = 0;
    sim.reset();

    const std::vector<std::string> samples = {
        "rand0", "rand1", "rand2", "half", "gradient",
    };

    int passed = 0;
    for (auto& nm : samples) {
        auto r = drive_sample(sim, nm, s_in);
        printf("  [%s] bit_exact_bad=%d  cos(DUT vs SW)=%.6f  cos(DUT vs ORT)=%.6f\n",
               nm.c_str(), r.bit_exact_bad, r.cos_sw, r.cos_ort);
        sim.check(r.bit_exact_bad == 0,
                  nm + ": DUT bit-exact match against SW int8 golden");
        sim.check(r.cos_ort >= 0.998,
                  nm + ": cos(DUT vs ORT float) >= 0.998");
        if (r.bit_exact_bad == 0 && r.cos_ort >= 0.998) passed++;
    }
    printf("samples passed: %d / %zu\n", passed, samples.size());

    return sim.finish();
}
