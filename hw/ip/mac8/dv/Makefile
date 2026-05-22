REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

TB_TOP   := mac8_tb
RTL_SRCS := \
  $(REPO_ROOT)/hw/ip/mac8/rtl/mac8.sv \
  $(REPO_ROOT)/hw/ip/mac8/rtl/mac8_ref.sv \
  $(CURDIR)/mac8_tb.sv
CC_SRCS  := $(CURDIR)/mac8_test.cc

include $(REPO_ROOT)/mk/verilator.mk
