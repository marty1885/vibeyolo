REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

TB_TOP   := fp16_fma_tb
RTL_SRCS := \
  $(REPO_ROOT)/hw/ip/fp16_fma/rtl/fp16_fma.sv \
  $(REPO_ROOT)/hw/ip/fp16_fma/rtl/fp16_fma_ref.sv \
  $(CURDIR)/fp16_fma_tb.sv
CC_SRCS  := $(CURDIR)/fp16_fma_test.cc

include $(REPO_ROOT)/mk/verilator.mk
