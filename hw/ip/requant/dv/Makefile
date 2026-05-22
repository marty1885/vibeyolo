REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

I2F_DIR  := $(REPO_ROOT)/hw/ip/i32_to_fp16/rtl
FMA_DIR  := $(REPO_ROOT)/hw/ip/fp16_fma/rtl
F2I_DIR  := $(REPO_ROOT)/hw/ip/fp16_to_i8_sat/rtl
RQ_DIR   := $(REPO_ROOT)/hw/ip/requant/rtl

TB_TOP   := requant_tb
RTL_SRCS := \
  $(I2F_DIR)/i32_to_fp16.sv \
  $(FMA_DIR)/fp16_fma.sv \
  $(F2I_DIR)/fp16_to_i8_sat.sv \
  $(RQ_DIR)/requant.sv \
  $(RQ_DIR)/requant_ref.sv \
  $(CURDIR)/requant_tb.sv
CC_SRCS  := $(CURDIR)/requant_test.cc

include $(REPO_ROOT)/mk/verilator.mk
