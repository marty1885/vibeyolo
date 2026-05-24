// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// dequant_n — Verilator test. Drives DUT (dequant_n) and behavioral REF in
// lockstep and cross-checks both against a C++ double shadow. dequant is
// exact (fp16(int8) is exact), so all three agree BIT-FOR-BIT (0 ULP).
// N via -DDQ_N=<n>.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "Vdequant_n_tb.h"
#include "sim_ctrl.h"

#ifndef DQ_N
#error "DQ_N must be defined (e.g. -DDQ_N=80)"
#endif
static constexpr int kN = DQ_N;

using DUT = Vdequant_n_tb;

static double fp16_to_double(uint16_t x) {
    int s=(x>>15)&1,e=(x>>10)&0x1F,f=x&0x3FF; double v;
    if(e==0x1F) v=(f==0)?1e300:std::nan(""); else if(e==0) v=std::ldexp((double)f,-24);
    else v=std::ldexp(1.0+(double)f/1024.0,e-15); return s?-v:v;
}
static uint16_t double_to_fp16(double v){
    if(std::isnan(v))return 0x7E00; int s=std::signbit(v)?1:0; double av=std::fabs(v);
    if(std::isinf(av)||av>=65520.0)return (uint16_t)((s<<15)|0x7C00);
    if(av==0.0)return (uint16_t)(s<<15); int e; double m=std::frexp(av,&e); int biased=(e-1)+15;
    if(biased>=31)return (uint16_t)((s<<15)|0x7C00);
    if(biased<=0){double sc=av*(double)(1<<24),fl=std::floor(sc);double fr=sc-fl;long mi=(long)fl;
        if(fr>0.5)mi++;else if(fr==0.5&&(mi&1))mi++; if(mi>=1024)return (uint16_t)((s<<15)|(1<<10));
        return (uint16_t)((s<<15)|(mi&0x3FF));}
    double md=(m*2.0-1.0)*1024.0,fl=std::floor(md);double fr=md-fl;long mi=(long)fl;
    if(fr>0.5)mi++;else if(fr==0.5&&(mi&1))mi++;
    if(mi>=1024){biased++;mi=0;if(biased>=31)return (uint16_t)((s<<15)|0x7C00);}
    return (uint16_t)((s<<15)|((biased&0x1F)<<10)|(mi&0x3FF));
}

static void set_x(DUT* dut, const std::vector<int8_t>& v) {
    std::memset(&dut->x_i, 0, sizeof(dut->x_i));
    uint8_t* raw = reinterpret_cast<uint8_t*>(&dut->x_i);
    for (int i = 0; i < kN; i++) raw[i] = (uint8_t)v[i];
}
static uint16_t y_dut(DUT* dut, int i) {
    return reinterpret_cast<uint16_t*>(&dut->y_dut_o)[i];
}

int main(int argc, char** argv) {
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 200000000ull;
    printf("==== dequant_n N=%d ====\n", kN);

    sim.dut->en_i = 0; set_x(sim.dut.get(), std::vector<int8_t>(kN,0)); sim.dut->scale_i = 0;
    sim.reset();
    sim.check(sim.dut->valid_dut_o == 0, "valid low after reset");

    std::mt19937 rng(0xDE9A47u + kN);
    std::uniform_int_distribution<int> d8(-128,127);
    std::uniform_int_distribution<int> dprob(0,99);

    // a few representative cls scales (S_cls per scale) + randoms
    std::vector<uint16_t> scales = {
        double_to_fp16(0.2415), double_to_fp16(0.3203), double_to_fp16(0.5624),
        double_to_fp16(1.0), double_to_fp16(0.01)
    };

    struct InFlight { bool v; std::vector<int8_t> x; uint16_t s; };
    std::vector<InFlight> q;
    int mism = 0, shadow_fail = 0, checks_seen = 0;

    auto on_edge = [&](void){
        if (sim.dut->valid_dut_o) {
            // pop oldest valid expectation
            size_t idx=0; bool found=false;
            for(; idx<q.size(); ++idx){ if(q[idx].v){found=true;break;} }
            if(!found){ shadow_fail++; return; }
            InFlight e = q[idx]; q.erase(q.begin()+idx);
            double sc = fp16_to_double(e.s);
            for (int i=0;i<kN;i++){
                uint16_t exp = double_to_fp16((double)e.x[i]*sc);
                if (y_dut(sim.dut.get(), i) != exp) shadow_fail++;
            }
            checks_seen++;
        }
        if (sim.dut->mismatch_o) mism++;
    };

    auto drive = [&](const std::vector<int8_t>& x, uint16_t s, bool en){
        set_x(sim.dut.get(), x); sim.dut->scale_i = s; sim.dut->en_i = en?1:0;
        sim.tick();
    };

    // Latency = I2F_LAT(2) + FMA_LAT(3) = 5 cycles; drain >= latency to
    // flush all in-flight vectors after the last drive.
    const int LATENCY = 5;
    const int NIT = 20000;
    for (int i = 0; i < NIT + LATENCY + 2; i++) {
        on_edge();
        if (i < NIT) {
            std::vector<int8_t> x(kN);
            for (int j=0;j<kN;j++) x[j]=(int8_t)d8(rng);
            uint16_t s = (dprob(rng)<50) ? scales[rng()%scales.size()]
                                         : double_to_fp16(0.001 + (rng()%2000)/1000.0);
            bool en = dprob(rng) < 85;
            drive(x, s, en);
            q.push_back({en, x, s});
        } else {
            drive(std::vector<int8_t>(kN,0), 0, false);
        }
    }

    printf("  outputs checked=%d\n", checks_seen);
    sim.check(mism == 0, "0 dut-vs-ref mismatches (was " + std::to_string(mism) + ")");
    sim.check(shadow_fail == 0, "0 dut-vs-shadow mismatches (was " + std::to_string(shadow_fail) + ")");
    sim.check(checks_seen > 15000, "saw enough valid outputs");

    return sim.finish();
}
