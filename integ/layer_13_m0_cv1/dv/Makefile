REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../..)
endif

MAC_DIR  := $(REPO_ROOT)/hw/ip/mac8/rtl
DOT_DIR  := $(REPO_ROOT)/hw/ip/dotN/rtl
I2F_DIR  := $(REPO_ROOT)/hw/ip/i32_to_fp16/rtl
FMA_DIR  := $(REPO_ROOT)/hw/ip/fp16_fma/rtl
F2I_DIR  := $(REPO_ROOT)/hw/ip/fp16_to_i8_sat/rtl
RQ_DIR   := $(REPO_ROOT)/hw/ip/requant/rtl
SIL_DIR  := $(REPO_ROOT)/hw/ip/act_silu/rtl
SCL_DIR  := $(REPO_ROOT)/integ/scale
L13_DIR  := $(REPO_ROOT)/integ/layer_13_m0_cv1/rtl

TB_TOP   := layer_13_m0_cv1_tb

RTL_SRCS := \
  $(SCL_DIR)/scale_pkg.sv \
  $(MAC_DIR)/mac8.sv \
  $(DOT_DIR)/dotN.sv \
  $(I2F_DIR)/i32_to_fp16.sv \
  $(FMA_DIR)/fp16_fma.sv \
  $(F2I_DIR)/fp16_to_i8_sat.sv \
  $(RQ_DIR)/requant.sv \
  $(SIL_DIR)/act_silu.sv \
  $(L13_DIR)/layer_13_m0_cv1.sv \
  $(CURDIR)/layer_13_m0_cv1_tb.sv

CC_SRCS  := $(CURDIR)/layer_13_m0_cv1_test.cc

LINT_FLAGS := -Wno-UNOPTFLAT -Wno-PINMISSING

include $(REPO_ROOT)/mk/verilator.mk

# Pre-existing PINMISSING warning from requant→i32_to_fp16 (shift_o pin
# added in a later refactor) trips -Wall during verilator's compile pass;
# downgrade it locally without touching the shared IP.
COMMON_FLAGS += -Wno-PINMISSING

.PHONY: stim
stim:
	python3 $(REPO_ROOT)/integ/layer_13_m0_cv1/extract.py

test: stim
