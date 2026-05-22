REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

TB_TOP   := act_sigmoid_tb
RTL_SRCS := \
  $(REPO_ROOT)/hw/ip/act_sigmoid/rtl/act_sigmoid.sv \
  $(REPO_ROOT)/hw/ip/act_sigmoid/rtl/act_sigmoid_ref.sv \
  $(CURDIR)/act_sigmoid_tb.sv
CC_SRCS  := $(CURDIR)/act_sigmoid_test.cc

include $(REPO_ROOT)/mk/verilator.mk
