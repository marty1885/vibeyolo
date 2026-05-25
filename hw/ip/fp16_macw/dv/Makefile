REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

TB_TOP   := fp16_macw_tb
RTL_SRCS := \
  $(REPO_ROOT)/hw/ip/fp16_macw/rtl/fp16_macw.sv \
  $(CURDIR)/fp16_macw_tb.sv
CC_SRCS  := $(CURDIR)/fp16_macw_test.cc

LINT_FLAGS := -Wno-UNUSEDPARAM

include $(REPO_ROOT)/mk/verilator.mk
