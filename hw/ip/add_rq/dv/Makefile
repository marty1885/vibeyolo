REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

I32_DIR  := $(REPO_ROOT)/hw/ip/i32_to_fp16/rtl
FMA_DIR  := $(REPO_ROOT)/hw/ip/fp16_fma/rtl
SAT_DIR  := $(REPO_ROOT)/hw/ip/fp16_to_i8_sat/rtl
RTL_DIR  := $(REPO_ROOT)/hw/ip/add_rq/rtl

TB_TOP   := add_rq_tb
RTL_SRCS := \
  $(I32_DIR)/i32_to_fp16.sv \
  $(FMA_DIR)/fp16_fma.sv \
  $(SAT_DIR)/fp16_to_i8_sat.sv \
  $(RTL_DIR)/add_rq.sv \
  $(RTL_DIR)/add_rq_ref.sv \
  $(CURDIR)/add_rq_tb.sv
CC_SRCS  := $(CURDIR)/add_rq_test.cc

include $(REPO_ROOT)/mk/verilator.mk
