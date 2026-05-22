REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

TB_TOP   := fp16_to_i8_sat_tb
RTL_SRCS := \
  $(REPO_ROOT)/hw/ip/fp16_to_i8_sat/rtl/fp16_to_i8_sat.sv \
  $(REPO_ROOT)/hw/ip/fp16_to_i8_sat/rtl/fp16_to_i8_sat_ref.sv \
  $(CURDIR)/fp16_to_i8_sat_tb.sv
CC_SRCS  := $(CURDIR)/fp16_to_i8_sat_test.cc

include $(REPO_ROOT)/mk/verilator.mk
