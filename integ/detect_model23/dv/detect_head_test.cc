// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// detect_head — full-block Verilator test for the YOLO26n /model.23 head.
//
// Drives the six int8 conv-tail streams (8400 anchors, canonical order) from
// the extractor's stim and validates the gathered detections against:
//   * the int8 REFERENCE goldens (bit-exact gate): selected anchor SET ==
//     ref TopK set; gathered logits bit-exact; gathered boxes within fp16
//     ULP tolerance (box_affine fuses/folds differently than the numpy ref).
//   * (informational) ORT, only at the stable stages — not gated, per the
//     tie-driven selection-churn caveat on object-free inputs.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "Vdetect_head_tb.h"
#include "sim_ctrl.h"

using DUT = Vdetect_head_tb;

static const int N_ANCHOR = 8400, N_CLS = 80, K = 300;
static const int SCNT[3] = {6400, 1600, 400};
static const char* SAMPLES[] = {"rand0","rand1","rand2","low","half","grad"};
static const int NSAMP = 6;
// Derive the stim dir from this source file's own path (__FILE__, which
// Verilator passes as an absolute path): .../detect_model23/dv/<this>.cc ->
// .../detect_model23/stim/. Portable across checkouts/machines and
// independent of the working directory — no hardcoded path.
static std::string stim_dir() {
    std::string f = __FILE__;
    auto p = f.rfind("/dv/");
    if (p == std::string::npos) return "../stim/";  // fallback: run from dv/
    return f.substr(0, p) + "/stim/";
}
static const std::string STIM = stim_dir();

// ── fp16 helpers ──
static double f2d(uint16_t x){int s=(x>>15)&1,e=(x>>10)&0x1F,f=x&0x3FF;double v;
  if(e==0x1F)v=(f==0)?1e300:std::nan("");else if(e==0)v=std::ldexp((double)f,-24);
  else v=std::ldexp(1.0+(double)f/1024.0,e-15);return s?-v:v;}
static uint16_t d2f(double v){if(std::isnan(v))return 0x7E00;int s=std::signbit(v)?1:0;double av=std::fabs(v);
  if(std::isinf(av)||av>=65520.0)return (uint16_t)((s<<15)|0x7C00);if(av==0.0)return (uint16_t)(s<<15);
  int e;double m=std::frexp(av,&e);int b=(e-1)+15;if(b>=31)return (uint16_t)((s<<15)|0x7C00);
  if(b<=0){double sc=av*(double)(1<<24),fl=std::floor(sc);double fr=sc-fl;long mi=(long)fl;
    if(fr>0.5)mi++;else if(fr==0.5&&(mi&1))mi++;if(mi>=1024)return (uint16_t)((s<<15)|(1<<10));
    return (uint16_t)((s<<15)|(mi&0x3FF));}
  double md=(m*2.0-1.0)*1024.0,fl=std::floor(md);double fr=md-fl;long mi=(long)fl;
  if(fr>0.5)mi++;else if(fr==0.5&&(mi&1))mi++;if(mi>=1024){b++;mi=0;if(b>=31)return (uint16_t)((s<<15)|0x7C00);}
  return (uint16_t)((s<<15)|((b&0x1F)<<10)|(mi&0x3FF));}
static int32_t f16ord(uint16_t x){return (x&0x8000)?-(int32_t)(x&0x7FFF):(int32_t)x;}
static int32_t f16ulp(uint16_t a,uint16_t b){int32_t d=f16ord(a)-f16ord(b);return d<0?-d:d;}
static double f16step(double v){double av=std::fabs(v);if(av==0)return std::ldexp(1.0,-24);
  int e;std::frexp(av,&e);int u=e-1;if(u<-14)u=-14;return std::ldexp(1.0,u-10);}

// Independent double-precision box decode for a gathered anchor, returning
// the four fp16 box words and the pre-cancellation pixel magnitude pmax
// (the fp16 rounding-error floor scales with pmax, not the cancelled result).
struct BoxShadow { uint16_t box[4]; double pmax; };
static BoxShadow box_shadow(const int8_t* ltrb, double S, int col, int row, int stride){
  double dl=ltrb[0]*S, dt=ltrb[1]*S, dr=ltrb[2]*S, db=ltrb[3]*S;
  double ax=col+0.5, ay=row+0.5, s=stride;
  double x1=(ax-dl)*s, x2=(ax+dr)*s, y1=(ay-dt)*s, y2=(ay+db)*s;
  BoxShadow o;
  o.box[0]=d2f(((x1+x2)/2.0)/640.0); o.box[1]=d2f(((y1+y2)/2.0)/640.0);
  o.box[2]=d2f((x2-x1)/640.0);       o.box[3]=d2f((y2-y1)/640.0);
  auto am=[](double v){return std::fabs(v);};
  o.pmax=std::max({am(ax*s),am(ay*s),am(dl*s),am(dr*s),am(dt*s),am(db*s),
                   am(x1),am(x2),am(y1),am(y2)});
  return o;
}
// scale id / grid geometry for an anchor index.
static void anchor_geo(int a, int& scl, int& col, int& row, int& stride){
  int gw;
  if(a<6400){ scl=0; stride=8;  gw=80; }
  else if(a<8000){ scl=1; stride=16; gw=40; a-=6400; }
  else { scl=2; stride=32; gw=20; a-=8000; }
  col=a%gw; row=a/gw;
}

template<typename T> static std::vector<T> load(const std::string& p){
  FILE* f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"missing %s\n",p.c_str());exit(2);}
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  std::vector<T> v(n/sizeof(T)); size_t got=fread(v.data(),sizeof(T),v.size(),f); (void)got; fclose(f); return v;
}

static void set_box(DUT* d, const int8_t* b){
  std::memset(&d->box_i,0,sizeof(d->box_i));
  uint8_t* r=reinterpret_cast<uint8_t*>(&d->box_i); for(int i=0;i<4;i++) r[i]=(uint8_t)b[i];
}
static void set_cls(DUT* d, const int8_t* c){
  std::memset(&d->cls_i,0,sizeof(d->cls_i));
  uint8_t* r=reinterpret_cast<uint8_t*>(&d->cls_i); for(int i=0;i<N_CLS;i++) r[i]=(uint8_t)c[i];
}

// ── chain mode ──
// Driven by tools/e2e/chain.py to run the REAL detect-head IP inside the
// end-to-end "chip" chain on a live image (not the canned stim). Reads the six
// int8 conv-tail streams + six fp16 scales from explicit paths, drives one
// frame, and dumps the gathered top-K detections (anchor + fp16 box + fp16
// logits) so chain.py can reassemble logits/pred_boxes. No goldens, no gating —
// the chain compares end-to-end detections to ORT itself.
//   env: CHAIN_BOX_S{0,1,2}, CHAIN_CLS_S{0,1,2}  (int8 [chan][spatial] .bin)
//        CHAIN_SCALES (6 float32: s_box0..2, s_cls0..2)   CHAIN_OUT (out .bin)
//   out: K records, each uint16[1+4+N_CLS] = anchor, box[4]fp16, logits[80]fp16
static int chain_mode(int argc, char** argv){
  SimCtrl<DUT> sim(argc, argv);
  sim.max_time = 40000000ull;

  std::vector<float> sc = load<float>(std::string(getenv("CHAIN_SCALES")));
  uint16_t sbox[3], scls[3];
  for(int i=0;i<3;i++){ sbox[i]=d2f(sc[i]); scls[i]=d2f(sc[3+i]); }
  const char* fbox[3]={getenv("CHAIN_BOX_S0"),getenv("CHAIN_BOX_S1"),getenv("CHAIN_BOX_S2")};
  const char* fcls[3]={getenv("CHAIN_CLS_S0"),getenv("CHAIN_CLS_S1"),getenv("CHAIN_CLS_S2")};

  std::vector<std::vector<int8_t>> boxA(N_ANCHOR, std::vector<int8_t>(4));
  std::vector<std::vector<int8_t>> clsA(N_ANCHOR, std::vector<int8_t>(N_CLS));
  int off=0;
  for(int si=0; si<3; si++){
    auto bx=load<int8_t>(std::string(fbox[si]));  // [4][n]
    auto cl=load<int8_t>(std::string(fcls[si]));   // [80][n]
    int n=SCNT[si];
    for(int p=0;p<n;p++){
      for(int c=0;c<4;c++)     boxA[off+p][c]=bx[c*n+p];
      for(int c=0;c<N_CLS;c++) clsA[off+p][c]=cl[c*n+p];
    }
    off+=n;
  }

  sim.dut->rst_ni=0; sim.dut->start_i=0; sim.dut->in_valid_i=0;
  sim.dut->s_box0_i=sbox[0]; sim.dut->s_box1_i=sbox[1]; sim.dut->s_box2_i=sbox[2];
  sim.dut->s_cls0_i=scls[0]; sim.dut->s_cls1_i=scls[1]; sim.dut->s_cls2_i=scls[2];
  sim.reset();
  sim.dut->start_i=1; sim.tick(); sim.dut->start_i=0;

  int fed=0; long guard=0;
  std::vector<uint16_t> oa; std::vector<std::array<uint16_t,4>> ob;
  std::vector<std::vector<uint16_t>> ol;
  while(true){
    if(sim.dut->in_ready_o && fed<N_ANCHOR){
      set_box(sim.dut.get(), boxA[fed].data());
      set_cls(sim.dut.get(), clsA[fed].data());
      sim.dut->in_valid_i=1;
    } else sim.dut->in_valid_i=0;
    if(sim.dut->out_valid_o){
      oa.push_back((uint16_t)sim.dut->out_anchor_o);
      std::array<uint16_t,4> bx; uint16_t* br=reinterpret_cast<uint16_t*>(&sim.dut->out_box_o);
      for(int c=0;c<4;c++) bx[c]=br[c]; ob.push_back(bx);
      std::vector<uint16_t> lg(N_CLS); uint16_t* lr=reinterpret_cast<uint16_t*>(&sim.dut->out_logits_o);
      for(int c=0;c<N_CLS;c++) lg[c]=lr[c]; ol.push_back(lg);
    }
    bool done = sim.dut->done_o;
    if(sim.dut->in_ready_o && fed<N_ANCHOR) fed++;
    sim.tick();
    if(done) break;
    if(++guard > 2000000){ fprintf(stderr,"chain detect TIMEOUT fed=%d outs=%zu\n",fed,oa.size()); break; }
  }
  int n_out=(int)oa.size();
  FILE* f=fopen(getenv("CHAIN_OUT"),"wb");
  for(int i=0;i<n_out;i++){
    fwrite(&oa[i],2,1,f);
    fwrite(ob[i].data(),2,4,f);
    fwrite(ol[i].data(),2,N_CLS,f);
  }
  fclose(f);
  fprintf(stderr,"chain detect: %d outputs -> %s\n", n_out, getenv("CHAIN_OUT"));
  return 0;
}

int main(int argc, char** argv){
  if(getenv("CHAIN_OUT")) return chain_mode(argc, argv);
  SimCtrl<DUT> sim(argc, argv);
  sim.max_time = 40000000ull;  // 6 frames x ~85k cyc topk-bound, with margin

  std::vector<float> sc = load<float>(std::string(STIM)+"scales.f32");
  uint16_t sbox[3], scls[3];
  for(int i=0;i<3;i++){ sbox[i]=d2f(sc[i]); scls[i]=d2f(sc[3+i]); }

  int total_fail=0, frames=0;

  for(int s=0;s<NSAMP;s++){
    std::string nm = std::string(STIM)+SAMPLES[s];
    // Build per-anchor box[4] and cls[80] in canonical order.
    std::vector<std::vector<int8_t>> boxA(N_ANCHOR, std::vector<int8_t>(4));
    std::vector<std::vector<int8_t>> clsA(N_ANCHOR, std::vector<int8_t>(N_CLS));
    int off=0;
    for(int si=0; si<3; si++){
      char tag[8]; snprintf(tag,sizeof(tag),"s%d",si);
      auto bx=load<int8_t>(nm+"_box_"+tag+".bin");   // [4][n]
      auto cl=load<int8_t>(nm+"_cls_"+tag+".bin");   // [80][n]
      int n=SCNT[si];
      for(int p=0;p<n;p++){
        for(int c=0;c<4;c++)     boxA[off+p][c]=bx[c*n+p];
        for(int c=0;c<N_CLS;c++) clsA[off+p][c]=cl[c*n+p];
      }
      off+=n;
    }
    // ref goldens
    auto ref_idx    = load<uint16_t>(nm+"_ref_idx.bin");      // [300]
    auto ref_gbox   = load<uint16_t>(nm+"_ref_gboxes.bin");   // [300][4]
    auto ref_glog   = load<uint16_t>(nm+"_ref_glogit.bin");   // [300][80]
    // map anchor -> ref index position
    std::vector<int> pos(N_ANCHOR,-1);
    for(int i=0;i<K;i++) pos[ref_idx[i]]=i;
    std::set<int> ref_set(ref_idx.begin(), ref_idx.end());

    // ── drive ──
    sim.dut->rst_ni=0; sim.dut->start_i=0; sim.dut->in_valid_i=0;
    sim.dut->s_box0_i=sbox[0]; sim.dut->s_box1_i=sbox[1]; sim.dut->s_box2_i=sbox[2];
    sim.dut->s_cls0_i=scls[0]; sim.dut->s_cls1_i=scls[1]; sim.dut->s_cls2_i=scls[2];
    sim.reset();

    sim.dut->start_i=1; sim.tick(); sim.dut->start_i=0;

    int fed=0; long guard=0;
    std::vector<int> got_anchor; std::vector<std::array<uint16_t,4>> got_box;
    std::vector<std::vector<uint16_t>> got_log;

    while(true){
      // feed
      if(sim.dut->in_ready_o && fed<N_ANCHOR){
        set_box(sim.dut.get(), boxA[fed].data());
        set_cls(sim.dut.get(), clsA[fed].data());
        sim.dut->in_valid_i=1;
      } else {
        sim.dut->in_valid_i=0;
      }
      // collect output BEFORE tick (combinational valid this cycle)
      if(sim.dut->out_valid_o){
        got_anchor.push_back((int)sim.dut->out_anchor_o);
        std::array<uint16_t,4> bx; uint16_t* br=reinterpret_cast<uint16_t*>(&sim.dut->out_box_o);
        for(int c=0;c<4;c++) bx[c]=br[c]; got_box.push_back(bx);
        std::vector<uint16_t> lg(N_CLS); uint16_t* lr=reinterpret_cast<uint16_t*>(&sim.dut->out_logits_o);
        for(int c=0;c<N_CLS;c++) lg[c]=lr[c]; got_log.push_back(lg);
      }
      bool done = sim.dut->done_o;
      if(sim.dut->in_ready_o && fed<N_ANCHOR) fed++;
      sim.tick();
      if(done) break;
      if(++guard > 2000000){ printf("  %s TIMEOUT (fed=%d outs=%zu)\n",SAMPLES[s],fed,got_anchor.size()); total_fail++; break; }
    }

    // ── checks ──
    int n_out=(int)got_anchor.size();
    if(n_out!=K){ printf("  %s FAIL: got %d outputs (want %d)\n",SAMPLES[s],n_out,K); total_fail++; continue; }

    std::set<int> dut_set(got_anchor.begin(), got_anchor.end());
    bool set_ok = (dut_set==ref_set);
    int logit_mism=0, box_oot=0, max_box_ulp=0;
    for(int i=0;i<n_out;i++){
      int a=got_anchor[i]; int rp=pos[a];
      if(rp<0){ continue; } // anchor not in ref set (counted via set_ok)
      // logits bit-exact (gather + dequant_n)
      for(int c=0;c<N_CLS;c++) if(got_log[i][c]!=ref_glog[rp*N_CLS+c]) logit_mism++;
      // boxes vs independent double shadow, cancellation-aware tolerance:
      // 16 ULP OR an absolute floor 8*fp16_step(pmax)/{1280,640}.
      int scl,col,row,stride; anchor_geo(a,scl,col,row,stride);
      BoxShadow bs = box_shadow(boxA[a].data(), f2d(scl==0?sbox[0]:scl==1?sbox[1]:sbox[2]),
                                col,row,stride);
      double atc = 8.0*f16step(bs.pmax)/1280.0, ats = 8.0*f16step(bs.pmax)/640.0;
      double atol[4]={atc,atc,ats,ats};
      for(int c=0;c<4;c++){
        uint16_t du=got_box[i][c], sd=bs.box[c];
        int u=f16ulp(du,sd); if(u>max_box_ulp)max_box_ulp=u;
        double ad=std::fabs(f2d(du)-f2d(sd));
        if(u>16 && ad>atol[c]) box_oot++;
      }
    }
    bool ok = set_ok && logit_mism==0 && box_oot==0;
    printf("  %-6s outs=%d set_ok=%d logit_mism=%d box_oot=%d max_box_ulp=%d common=%zu/300 cyc=%ld\n",
           SAMPLES[s], n_out, set_ok, logit_mism, box_oot, max_box_ulp,
           [&]{size_t c=0;for(int a:dut_set)if(ref_set.count(a))c++;return c;}(), guard);
    if(!ok) total_fail++;
    frames++;
  }

  sim.checks += frames; // count frames as checks for the summary
  sim.check(total_fail==0, "all frames: DUT matches int8 reference (set + logits exact, box within ULP)");
  return sim.finish();
}
