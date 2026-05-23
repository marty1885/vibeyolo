REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)

MAC_DIR  := $(REPO_ROOT)/hw/ip/mac8/rtl
DOT_DIR  := $(REPO_ROOT)/hw/ip/dotN/rtl
I2F_DIR  := $(REPO_ROOT)/hw/ip/i32_to_fp16/rtl
FMA_DIR  := $(REPO_ROOT)/hw/ip/fp16_fma/rtl
F2I_DIR  := $(REPO_ROOT)/hw/ip/fp16_to_i8_sat/rtl
RQ_DIR   := $(REPO_ROOT)/hw/ip/requant/rtl
SIL_DIR  := $(REPO_ROOT)/hw/ip/act_silu/rtl
ARQ_DIR  := $(REPO_ROOT)/hw/ip/add_rq/rtl
CL_DIR   := $(REPO_ROOT)/hw/ip/conv_layer/rtl
LB_DIR   := $(REPO_ROOT)/hw/ip/linebuf_kxk/rtl
CS_DIR   := $(REPO_ROOT)/hw/ip/conv_stage/rtl

TB_TOP   := cs_frame_tb

RTL_SRCS := \
  $(MAC_DIR)/mac8.sv $(DOT_DIR)/dotN.sv $(I2F_DIR)/i32_to_fp16.sv \
  $(FMA_DIR)/fp16_fma.sv $(F2I_DIR)/fp16_to_i8_sat.sv $(RQ_DIR)/requant.sv \
  $(SIL_DIR)/act_silu.sv $(ARQ_DIR)/add_rq.sv $(CL_DIR)/conv_layer.sv \
  $(LB_DIR)/linebuf_kxk.sv $(CS_DIR)/conv_stage.sv \
  $(CURDIR)/cs_frame_tb.sv

CC_SRCS  := $(CURDIR)/cs_frame_test.cc

# ── Per-layer parameters (set by tools/e2e/run_layer.py) ──
CIN ?= 8
COUT ?= 8
K ?= 3
STRIDE ?= 1
PAD ?= 1
H_IN ?= 6
W_IN ?= 6
P_COUT ?= 4
P_CIN ?= 4
SILU ?= 1
S_OUT_PRE ?= 0.0315
S_OUT_SILU ?= 0.0315
STIM ?= $(CURDIR)
LAYER ?= layer

GDEF := -GCIN=$(CIN) -GCOUT=$(COUT) -GK=$(K) -GSTRIDE=$(STRIDE) -GPAD=$(PAD) \
        -GH_IN=$(H_IN) -GW_IN=$(W_IN) -GP_COUT=$(P_COUT) -GP_CIN=$(P_CIN) \
        -GSILU=$(SILU) -GS_OUT_PRE=$(S_OUT_PRE) -GS_OUT_SILU=$(S_OUT_SILU) \
        -GWINIT="\"$(STIM)/weights.i8.hex\"" \
        -GSINIT="\"$(STIM)/scale_fp16.hex\"" \
        -GBINIT="\"$(STIM)/bias_fp16.hex\""

LINT_FLAGS := -Wno-WIDTHEXPAND -Wno-WIDTHCONCAT -Wno-UNUSEDSIGNAL -Wno-UNOPTFLAT \
              -Wno-PINCONNECTEMPTY $(GDEF)

# unique build dir per layer so parallel sweeps don't collide
BUILD_DIR := $(CURDIR)/build/$(LAYER)
SIM_ARGS  := $(CIN) $(COUT) $(K) $(STRIDE) $(PAD) $(H_IN) $(W_IN) \
             $(STIM)/input_i8.hex $(STIM)/out_i8.hex

include $(REPO_ROOT)/mk/verilator.mk
