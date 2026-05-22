REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

TB_TOP   := act_silu_tb
RTL_SRCS := \
  $(REPO_ROOT)/hw/ip/act_silu/rtl/act_silu.sv \
  $(REPO_ROOT)/hw/ip/act_silu/rtl/act_silu_ref.sv \
  $(CURDIR)/act_silu_tb.sv
CC_SRCS  := $(CURDIR)/act_silu_test.cc

include $(REPO_ROOT)/mk/verilator.mk
