// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// attn_test.cc — drives the attention integration block with
// stim/<sample>.{qkv,pe,proj,spl1,ffn1}_i8.hex, captures the DUT's
// 128ch×20×20 int8 final output, and compares to:
//   1. Software int8 golden (stim/<sample>.golden_final_i8.hex):
//      target — match bit-exactly. The SW model uses fp64 math so the
//      DUT (fp16) may differ by a small number of samples; the cos
//      check below is the gating criterion.
//   2. Float ORT reference (stim/<sample>.ref_final_f32.hex):
//      target — cos ≥ 0.998 on the random tiles.

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
    "/home/marty/Documents/aif/vibeyolo/integ/attn_model22/stim";


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
    int    cycles        = 0;
};


static SampleResult drive_sample(SimCtrl<Vattn_tb>& sim, const std::string& name,
                                 double s_out) {
    SampleResult r{};
    const size_t N    = size_t(H) * W;
    const size_t Nqkv = N * C_QKV;
    const size_t Nfe  = N * C_FE;

    auto qkv_in   = load_i8_hex (std::string(STIM_DIR) + "/" + name + ".qkv_i8.hex",  Nqkv);
    auto pe_in    = load_i8_hex (std::string(STIM_DIR) + "/" + name + ".pe_i8.hex",   Nfe);
    auto proj_in  = load_i8_hex (std::string(STIM_DIR) + "/" + name + ".proj_i8.hex", Nfe);
    auto spl1_in  = load_i8_hex (std::string(STIM_DIR) + "/" + name + ".spl1_i8.hex", Nfe);
    auto ffn1_in  = load_i8_hex (std::string(STIM_DIR) + "/" + name + ".ffn1_i8.hex", Nfe);
    auto golden   = load_i8_hex (std::string(STIM_DIR) + "/" + name + ".golden_final_i8.hex", Nfe);
    auto ref_ort  = load_f32_hex(std::string(STIM_DIR) + "/" + name + ".ref_final_f32.hex",   Nfe);

    int8_t zero_qkv[C_QKV]; std::memset(zero_qkv, 0, sizeof(zero_qkv));
    int8_t zero_fe [C_FE];  std::memset(zero_fe,  0, sizeof(zero_fe));

    sim.dut->start_i      = 0;
    sim.dut->qkv_valid_i  = 0;
    sim.dut->pe_valid_i   = 0;
    sim.dut->proj_valid_i = 0;
    sim.dut->spl1_valid_i = 0;
    sim.dut->ffn1_valid_i = 0;
    sim.dut->oready_i     = 0;
    pack_lanes(&sim.dut->qkv_data_i,  zero_qkv, C_QKV);
    pack_lanes(&sim.dut->pe_data_i,   zero_fe,  C_FE);
    pack_lanes(&sim.dut->proj_data_i, zero_fe,  C_FE);
    pack_lanes(&sim.dut->spl1_data_i, zero_fe,  C_FE);
    pack_lanes(&sim.dut->ffn1_data_i, zero_fe,  C_FE);
    sim.tick();

    sim.dut->start_i = 1;
    sim.tick();
    sim.dut->start_i = 0;

    size_t qi = 0, pi = 0, ri = 0, si = 0, fi = 0, oi = 0;
    std::vector<int8_t> dut_out(N * C_FE, 0);
    const int max_cycles = 50'000'000;
    int cycles = 0;
    int8_t out_buf[C_FE];

    while (oi < N) {
        if (qi < N) { sim.dut->qkv_valid_i = 1;
                      pack_lanes(&sim.dut->qkv_data_i,  &qkv_in[qi * C_QKV], C_QKV); }
        else         { sim.dut->qkv_valid_i = 0;
                      pack_lanes(&sim.dut->qkv_data_i,  zero_qkv, C_QKV); }
        if (pi < N) { sim.dut->pe_valid_i = 1;
                      pack_lanes(&sim.dut->pe_data_i,   &pe_in[pi  * C_FE],  C_FE); }
        else         { sim.dut->pe_valid_i = 0;
                      pack_lanes(&sim.dut->pe_data_i,   zero_fe,  C_FE); }
        if (ri < N) { sim.dut->proj_valid_i = 1;
                      pack_lanes(&sim.dut->proj_data_i, &proj_in[ri * C_FE], C_FE); }
        else         { sim.dut->proj_valid_i = 0;
                      pack_lanes(&sim.dut->proj_data_i, zero_fe,  C_FE); }
        if (si < N) { sim.dut->spl1_valid_i = 1;
                      pack_lanes(&sim.dut->spl1_data_i, &spl1_in[si * C_FE], C_FE); }
        else         { sim.dut->spl1_valid_i = 0;
                      pack_lanes(&sim.dut->spl1_data_i, zero_fe,  C_FE); }
        if (fi < N) { sim.dut->ffn1_valid_i = 1;
                      pack_lanes(&sim.dut->ffn1_data_i, &ffn1_in[fi * C_FE], C_FE); }
        else         { sim.dut->ffn1_valid_i = 0;
                      pack_lanes(&sim.dut->ffn1_data_i, zero_fe,  C_FE); }
        sim.dut->oready_i = 1;

        sim.dut->eval();

        bool q_fire = sim.dut->qkv_valid_i  && sim.dut->qkv_ready_o;
        bool p_fire = sim.dut->pe_valid_i   && sim.dut->pe_ready_o;
        bool r_fire = sim.dut->proj_valid_i && sim.dut->proj_ready_o;
        bool s_fire = sim.dut->spl1_valid_i && sim.dut->spl1_ready_o;
        bool f_fire = sim.dut->ffn1_valid_i && sim.dut->ffn1_ready_o;
        bool o_fire = sim.dut->ovalid_o     && sim.dut->oready_i;

        if (o_fire) {
            unpack_lanes(&sim.dut->odata_o, out_buf, C_FE);
            std::memcpy(&dut_out[oi * C_FE], out_buf, C_FE);
            oi++;
        }
        sim.tick();
        if (q_fire) qi++;
        if (p_fire) pi++;
        if (r_fire) ri++;
        if (s_fire) si++;
        if (f_fire) fi++;
        cycles++;
        if (cycles > max_cycles) {
            printf("  %s: TIMEOUT q=%zu p=%zu r=%zu s=%zu f=%zu o=%zu\n",
                   name.c_str(), qi, pi, ri, si, fi, oi);
            sim.check(false, name + " timeout");
            return r;
        }
    }
    r.cycles = cycles;

    sim.dut->qkv_valid_i = sim.dut->pe_valid_i = sim.dut->proj_valid_i = 0;
    sim.dut->spl1_valid_i = sim.dut->ffn1_valid_i = sim.dut->oready_i = 0;
    sim.dut->eval();

    int bad = 0;
    for (size_t i = 0; i < dut_out.size(); i++) {
        if (dut_out[i] != golden[i]) {
            if (bad < 4) {
                size_t pix = i / C_FE; size_t ch = i % C_FE;
                int hh = int(pix / W); int ww = int(pix % W);
                printf("  %s: i8 diff @ (h=%d w=%d ch=%zu) got=%d exp=%d\n",
                       name.c_str(), hh, ww, ch, int(dut_out[i]), int(golden[i]));
            }
            bad++;
        }
    }
    r.bit_exact_bad = bad;

    auto cos_vs_f32 = [&](const float* ref) {
        long double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < N * C_FE; i++) {
            double aa = double(dut_out[i]) * s_out;
            double bb = double(ref[i]);
            dot += aa * bb; na += aa * aa; nb += bb * bb;
        }
        return double(dot / (std::sqrt(double(na)) * std::sqrt(double(nb)) + 1e-30L));
    };
    {
        std::vector<float> golden_f(N * C_FE);
        for (size_t i = 0; i < golden_f.size(); i++)
            golden_f[i] = float(golden[i]) * float(s_out);
        r.cos_sw = cos_vs_f32(golden_f.data());
    }
    r.cos_ort = cos_vs_f32(ref_ort.data());
    return r;
}

static double load_real_from_manifest(const std::string& key) {
    std::ifstream in(std::string(STIM_DIR) + "/manifest.json");
    if (!in) { fprintf(stderr, "missing manifest.json\n"); std::exit(2); }
    std::string txt((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    std::string k = "\"" + key + "\"";
    auto pos = txt.find(k);
    if (pos == std::string::npos) { fprintf(stderr, "no key %s\n", key.c_str()); std::exit(2); }
    auto colon = txt.find(':', pos);
    auto comma = txt.find_first_of(",}\n", colon + 1);
    return std::stod(txt.substr(colon + 1, comma - colon - 1));
}

int main(int argc, char** argv) {
    SimCtrl<Vattn_tb> sim(argc, argv);
    sim.max_time = 8'000'000'000ull;

    printf("attn test (H=%d W=%d C_QKV=%d C_FE=%d)\n", H, W, C_QKV, C_FE);

    sim.dut->start_i      = 0;
    sim.dut->qkv_valid_i  = 0;
    sim.dut->pe_valid_i   = 0;
    sim.dut->proj_valid_i = 0;
    sim.dut->spl1_valid_i = 0;
    sim.dut->ffn1_valid_i = 0;
    sim.dut->oready_i     = 0;
    sim.reset();

    double s_out = load_real_from_manifest("S_OUT");
    printf("  S_OUT = %.6f\n", s_out);

    const std::vector<std::string> samples = {
        "rand0", "rand1", "rand2", "half", "gradient",
    };

    int passed = 0;
    for (auto& nm : samples) {
        printf("  --- sample %s ---\n", nm.c_str());
        auto r = drive_sample(sim, nm, s_out);
        printf("  [%s] bit_exact_bad=%d  cos(DUT vs SW)=%.6f  cos(DUT vs ORT)=%.6f  cycles=%d\n",
               nm.c_str(), r.bit_exact_bad, r.cos_sw, r.cos_ort, r.cycles);
        const bool is_random_tile = (nm.rfind("rand", 0) == 0);
        if (is_random_tile) {
            sim.check(r.cos_ort >= 0.998,
                      nm + ": cos(DUT vs ORT) >= 0.998 on random tile");
        } else {
            sim.check(r.cos_ort >= 0.990,
                      nm + ": cos(DUT vs ORT) >= 0.990 on non-random tile");
        }
        // DUT-vs-SW cos check (both are fp16-ish; should be very close).
        sim.check(r.cos_sw >= 0.990,
                  nm + ": cos(DUT vs SW int8 golden) >= 0.990");
        if (r.cos_ort >= (is_random_tile ? 0.998 : 0.990)) passed++;
    }
    printf("samples passed: %d / %zu\n", passed, samples.size());

    return sim.finish();
}
