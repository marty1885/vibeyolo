// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// weight_rom — Verilator test.
//
// Drives DUT (weight_rom) and behavioral REF (weight_rom_ref) with
// identical stimulus via weight_rom_tb. Each cycle we (a) assert
// mismatch_o == 0 between DUT and REF and (b) cross-check the DUT
// against an independent C++ shadow built from the same deterministic
// pattern the Python generator used. This guards against the case
// where DUT and REF share the same $readmemh bug.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "Vweight_rom_tb.h"
#include "sim_ctrl.h"

#ifndef WROM_KH
#define WROM_KH 3
#endif
#ifndef WROM_KW
#define WROM_KW 3
#endif
#ifndef WROM_IC
#define WROM_IC 4
#endif
#ifndef WROM_OC
#define WROM_OC 8
#endif

static constexpr int KH = WROM_KH;
static constexpr int KW = WROM_KW;
static constexpr int IC = WROM_IC;
static constexpr int OC = WROM_OC;
static constexpr int ROW_LEN = KH * KW * IC;

using DUT = Vweight_rom_tb;

// ── fp16 encode (matches Python's struct 'e' format) ────
//
// IEEE-754 binary16 round-to-nearest-even encoder, written by hand so
// the test doesn't depend on the __fp16 extension (which is ARM-only
// on most GCC builds). Bit-identical to Python's struct '<e'.
static uint16_t encode_fp16(float x) {
    uint32_t f;
    std::memcpy(&f, &x, sizeof(f));
    uint32_t sign = (f >> 31) & 0x1u;
    int32_t  exp  = static_cast<int32_t>((f >> 23) & 0xFFu) - 127;
    uint32_t mant = f & 0x7FFFFFu;

    uint16_t h_sign = static_cast<uint16_t>(sign << 15);

    // NaN / Inf
    if (exp == 128) {
        if (mant != 0) return static_cast<uint16_t>(h_sign | 0x7E00u);  // qNaN
        return static_cast<uint16_t>(h_sign | 0x7C00u);                 // Inf
    }

    // Normal fp16 range: -14 <= exp <= 15.
    if (exp > 15) {
        return static_cast<uint16_t>(h_sign | 0x7C00u);  // overflow -> Inf
    }

    if (exp >= -14) {
        // Normal: shift 23-bit mantissa down to 10, round to nearest even.
        uint32_t m10 = mant >> 13;
        uint32_t rem = mant & 0x1FFFu;
        uint32_t half = 0x1000u;
        if (rem > half || (rem == half && (m10 & 1u))) {
            m10++;
            if (m10 == 0x400u) {  // mantissa overflow -> bump exponent
                m10 = 0;
                exp++;
                if (exp > 15) return static_cast<uint16_t>(h_sign | 0x7C00u);
            }
        }
        uint16_t h_exp = static_cast<uint16_t>((exp + 15) << 10);
        return static_cast<uint16_t>(h_sign | h_exp | (m10 & 0x3FFu));
    }

    // Subnormal or zero.
    if (exp < -24) return h_sign;  // underflow -> signed zero

    // Subnormal: include implicit leading 1, shift down.
    uint32_t m_with_implicit = mant | 0x800000u;
    int shift = -14 - exp + 13;  // shift right by this many to get 10-bit
    uint32_t m10 = m_with_implicit >> shift;
    uint32_t rem = m_with_implicit & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (m10 & 1u))) m10++;
    return static_cast<uint16_t>(h_sign | (m10 & 0x3FFu));
}

static uint8_t expected_byte(int oc, int b) {
    return static_cast<uint8_t>((oc * ROW_LEN + b) & 0xFF);
}
static uint16_t expected_scale(int oc) {
    return encode_fp16(1.0f / static_cast<float>(oc + 1));
}
static uint16_t expected_bias(int oc) {
    return encode_fp16(static_cast<float>(oc) * 0.5f);
}

static void apply(SimCtrl<DUT>& s, bool req, int oc_addr, int byte_idx) {
    s.dut->req_i      = req ? 1 : 0;
    s.dut->oc_addr_i  = static_cast<uint32_t>(oc_addr);
    s.dut->byte_idx_i = static_cast<uint32_t>(byte_idx);
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 20000000ull;

    sim.dut->req_i      = 0;
    sim.dut->oc_addr_i  = 0;
    sim.dut->byte_idx_i = 0;
    sim.reset();

    // ── Test 1: outputs zero after reset, hold semantics ──
    printf("test 1: reset + hold semantics\n");
    sim.check(sim.dut->scale_dut_o == 0, "dut scale==0 after reset");
    sim.check(sim.dut->bias_dut_o  == 0, "dut bias==0 after reset");
    sim.check(sim.dut->mismatch_o  == 0, "no mismatch after reset");

    // Idle with req=0 — outputs must hold (still zero here).
    for (int i = 0; i < 4; i++) {
        apply(sim, false, 3, 5);
        sim.tick();
        sim.check(sim.dut->scale_dut_o == 0, "hold: scale unchanged");
        sim.check(sim.dut->bias_dut_o  == 0, "hold: bias unchanged");
        sim.check(sim.dut->mismatch_o  == 0, "hold: no mismatch");
    }

    // ── Test 2: walk all OC addresses, verify every byte ──
    printf("test 2: sweep all OC addresses, all bytes\n");
    for (int oc = 0; oc < OC; oc++) {
        // Cycle T: issue request for this oc, byte_idx=0.
        apply(sim, true, oc, 0);
        sim.tick();  // T+1: row registered.

        // Now sweep all bytes combinationally (byte_idx is a slice select).
        for (int b = 0; b < ROW_LEN; b++) {
            sim.dut->byte_idx_i = static_cast<uint32_t>(b);
            // Deassert req so outputs hold while we walk byte_idx.
            sim.dut->req_i = 0;
            sim.dut->eval();
            uint8_t got_dut = sim.dut->w_byte_dut_o;
            uint8_t got_ref = sim.dut->w_byte_ref_o;
            uint8_t exp     = expected_byte(oc, b);
            if (got_dut != exp || got_ref != exp) {
                sim.check(false,
                          "sweep oc=" + std::to_string(oc) +
                          " b=" + std::to_string(b) +
                          " exp=" + std::to_string(int(exp)) +
                          " dut=" + std::to_string(int(got_dut)) +
                          " ref=" + std::to_string(int(got_ref)));
                break;
            }
        }

        uint16_t exp_s = expected_scale(oc);
        uint16_t exp_b = expected_bias(oc);
        sim.check(sim.dut->scale_dut_o == exp_s,
                  "oc=" + std::to_string(oc) + " scale matches");
        sim.check(sim.dut->bias_dut_o  == exp_b,
                  "oc=" + std::to_string(oc) + " bias matches");
        sim.check(sim.dut->scale_ref_o == exp_s,
                  "oc=" + std::to_string(oc) + " ref scale matches");
        sim.check(sim.dut->bias_ref_o  == exp_b,
                  "oc=" + std::to_string(oc) + " ref bias matches");
        sim.check(sim.dut->mismatch_o  == 0,
                  "oc=" + std::to_string(oc) + " no DUT/REF mismatch");
    }

    // ── Test 3: 1-cycle latency timing ─────────────────────
    printf("test 3: 1-cycle read latency\n");
    // Force outputs to a known non-target state first.
    apply(sim, true, 0, 0);
    sim.tick();
    apply(sim, true, OC - 1, 0);  // request last OC
    // Before tick: outputs still reflect oc=0.
    sim.dut->eval();
    sim.check(sim.dut->scale_dut_o == expected_scale(0),
              "pre-tick: scale still oc=0");
    sim.tick();  // now outputs reflect OC-1
    sim.check(sim.dut->scale_dut_o == expected_scale(OC - 1),
              "post-tick: scale oc=OC-1");
    sim.check(sim.dut->mismatch_o == 0, "latency: no mismatch");

    // ── Test 4: hold across many idle cycles ───────────────
    printf("test 4: hold while req=0\n");
    apply(sim, true, 2, 0);
    sim.tick();
    uint16_t held_s = sim.dut->scale_dut_o;
    uint16_t held_b = sim.dut->bias_dut_o;
    for (int i = 0; i < 10; i++) {
        // Drive a different address with req=0 — must not affect outputs.
        apply(sim, false, (i * 3) % OC, (i * 5) % ROW_LEN);
        sim.tick();
        sim.check(sim.dut->scale_dut_o == held_s, "hold: scale stable");
        sim.check(sim.dut->bias_dut_o  == held_b, "hold: bias stable");
        sim.check(sim.dut->mismatch_o  == 0,      "hold: no mismatch");
    }

    // ── Test 5: random address sequence, 1000+ accesses ────
    printf("test 5: random address stress (2000 accesses)\n");
    std::mt19937 rng(0xBADC0FFEu);
    std::uniform_int_distribution<int> docc(0, OC - 1);
    std::uniform_int_distribution<int> dbyte(0, ROW_LEN - 1);
    std::uniform_int_distribution<int> dreq(0, 9);

    int last_oc = -1;     // last oc whose row is currently registered
    int n_checked = 0;
    int n_mismatch = 0;
    for (int i = 0; i < 2000; i++) {
        int oc = docc(rng);
        int by = dbyte(rng);
        bool req = (dreq(rng) < 8);  // 80% request, 20% hold
        apply(sim, req, oc, by);
        sim.tick();

        if (req) last_oc = oc;

        // Pick a random byte to probe (combinational slice).
        int probe_b = dbyte(rng);
        sim.dut->byte_idx_i = static_cast<uint32_t>(probe_b);
        sim.dut->req_i      = 0;
        sim.dut->eval();

        if (sim.dut->mismatch_o) n_mismatch++;

        if (last_oc >= 0) {
            uint8_t exp_byte = expected_byte(last_oc, probe_b);
            uint16_t exp_s   = expected_scale(last_oc);
            uint16_t exp_b   = expected_bias(last_oc);
            if (sim.dut->w_byte_dut_o != exp_byte ||
                sim.dut->scale_dut_o  != exp_s    ||
                sim.dut->bias_dut_o   != exp_b) {
                sim.check(false,
                          "rand@" + std::to_string(i) +
                          " oc=" + std::to_string(last_oc) +
                          " b=" + std::to_string(probe_b) +
                          " exp_byte=" + std::to_string(int(exp_byte)) +
                          " got_byte=" + std::to_string(int(sim.dut->w_byte_dut_o)));
                break;
            }
            n_checked++;
        }
    }
    sim.check(n_mismatch == 0,
              "rand: 0 DUT/REF mismatches (was " +
              std::to_string(n_mismatch) + ")");
    sim.check(n_checked >= 1500,
              "rand: at least 1500 shadow-checked accesses (got " +
              std::to_string(n_checked) + ")");

    return sim.finish();
}
