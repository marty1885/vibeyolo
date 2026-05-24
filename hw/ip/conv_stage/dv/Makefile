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

TB_TOP   := conv_stage_tb

RTL_SRCS := \
  $(MAC_DIR)/mac8.sv \
  $(DOT_DIR)/dotN.sv \
  $(I2F_DIR)/i32_to_fp16.sv \
  $(FMA_DIR)/fp16_fma.sv \
  $(F2I_DIR)/fp16_to_i8_sat.sv \
  $(RQ_DIR)/requant.sv \
  $(SIL_DIR)/act_silu.sv \
  $(ARQ_DIR)/add_rq.sv \
  $(CL_DIR)/conv_layer.sv \
  $(LB_DIR)/linebuf_kxk.sv \
  $(CS_DIR)/conv_stage.sv \
  $(CURDIR)/conv_stage_tb.sv

CC_SRCS  := $(CURDIR)/conv_stage_test.cc

# ── Config matrix. Each sets matching SV (-G) and C++ (-D) parameters. ──
CFG ?= a
ifeq ($(CFG),a)        # k3 s1, multi cin/cout tile
  P := CIN=8 COUT=8 K=3 STRIDE=1 PAD=1 H_IN=6  W_IN=6  P_COUT=4 P_CIN=4 SILU=1
endif
ifeq ($(CFG),b)        # k3 s2 (stride decimation)
  P := CIN=8 COUT=8 K=3 STRIDE=2 PAD=1 H_IN=8  W_IN=8  P_COUT=4 P_CIN=4 SILU=1
endif
ifeq ($(CFG),c)        # k1 s1, single cout tile
  P := CIN=8 COUT=8 K=1 STRIDE=1 PAD=0 H_IN=5  W_IN=5  P_COUT=8 P_CIN=4 SILU=1
endif
ifeq ($(CFG),e)        # partial tiles (COUT,CIN not multiples of P)
  P := CIN=6 COUT=6 K=3 STRIDE=1 PAD=1 H_IN=5  W_IN=5  P_COUT=4 P_CIN=4 SILU=1
endif
ifeq ($(CFG),f)        # no-SiLU tail
  P := CIN=8 COUT=8 K=1 STRIDE=1 PAD=0 H_IN=5  W_IN=5  P_COUT=4 P_CIN=4 SILU=0
endif

GDEF := $(foreach kv,$(P),-G$(kv))
CDEF := $(foreach kv,$(P),-DCFG_$(kv))

LINT_FLAGS := -Wno-WIDTHEXPAND -Wno-WIDTHCONCAT -Wno-UNUSEDSIGNAL -Wno-UNOPTFLAT \
              -Wno-PINCONNECTEMPTY \
              $(GDEF) -CFLAGS "$(CDEF)"

BUILD_DIR := $(CURDIR)/build/$(CFG)

include $(REPO_ROOT)/mk/verilator.mk

.PHONY: test-all
test-all:
	@for c in a b c e f; do \
	  echo "════════ CFG=$$c ════════"; \
	  $(MAKE) --no-print-directory test CFG=$$c || exit 1; \
	done
	@echo "ALL CONFIGS PASSED"
