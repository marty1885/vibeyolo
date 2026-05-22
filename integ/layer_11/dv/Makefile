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
ARQ_DIR  := $(REPO_ROOT)/hw/ip/add_rq/rtl
CL_DIR   := $(REPO_ROOT)/hw/ip/conv_layer/rtl
SCL_DIR  := $(REPO_ROOT)/integ/scale
L11_DIR  := $(REPO_ROOT)/integ/layer_11/rtl

TB_TOP   := layer_11_tb

RTL_SRCS := \
  $(SCL_DIR)/scale_pkg_dv.sv \
  $(MAC_DIR)/mac8.sv \
  $(DOT_DIR)/dotN.sv \
  $(I2F_DIR)/i32_to_fp16.sv \
  $(FMA_DIR)/fp16_fma.sv \
  $(F2I_DIR)/fp16_to_i8_sat.sv \
  $(RQ_DIR)/requant.sv \
  $(SIL_DIR)/act_silu.sv \
  $(ARQ_DIR)/add_rq.sv \
  $(CL_DIR)/conv_layer.sv \
  $(L11_DIR)/layer_11.sv \
  $(CURDIR)/layer_11_tb.sv

CC_SRCS  := $(CURDIR)/layer_11_test.cc

LINT_FLAGS := -Wno-UNOPTFLAT

include $(REPO_ROOT)/mk/verilator.mk

.PHONY: stim
stim:
	python3 $(REPO_ROOT)/integ/layer_11/extract.py

test: stim
