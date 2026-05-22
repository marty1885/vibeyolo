REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

TB_TOP   := softmax16_tb
RTL_SRCS := \
  $(REPO_ROOT)/hw/ip/fp16_fma/rtl/fp16_fma.sv \
  $(REPO_ROOT)/hw/ip/softmax16/rtl/softmax16.sv \
  $(REPO_ROOT)/hw/ip/softmax16/rtl/softmax16_ref.sv \
  $(CURDIR)/softmax16_tb.sv
CC_SRCS  := $(CURDIR)/softmax16_test.cc

include $(REPO_ROOT)/mk/verilator.mk
