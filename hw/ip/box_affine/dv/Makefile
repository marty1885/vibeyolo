REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

TB_TOP   := box_affine_tb
RTL_SRCS := \
  $(REPO_ROOT)/hw/ip/fp16_fma/rtl/fp16_fma.sv \
  $(REPO_ROOT)/hw/ip/i32_to_fp16/rtl/i32_to_fp16.sv \
  $(REPO_ROOT)/hw/ip/box_affine/rtl/box_affine.sv \
  $(REPO_ROOT)/hw/ip/box_affine/rtl/box_affine_ref.sv \
  $(CURDIR)/box_affine_tb.sv
CC_SRCS  := $(CURDIR)/box_affine_test.cc
LINT_FLAGS := -Wno-MULTITOP

include $(REPO_ROOT)/mk/verilator.mk
