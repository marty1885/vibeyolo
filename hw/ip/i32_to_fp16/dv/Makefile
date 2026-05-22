REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

TB_TOP   := i32_to_fp16_tb
RTL_SRCS := \
  $(REPO_ROOT)/hw/ip/i32_to_fp16/rtl/i32_to_fp16.sv \
  $(REPO_ROOT)/hw/ip/i32_to_fp16/rtl/i32_to_fp16_ref.sv \
  $(CURDIR)/i32_to_fp16_tb.sv
CC_SRCS  := $(CURDIR)/i32_to_fp16_test.cc

include $(REPO_ROOT)/mk/verilator.mk
