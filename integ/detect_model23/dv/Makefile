REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../..)
endif

IP_DIR   := $(REPO_ROOT)/hw/ip
S_DIR    := $(REPO_ROOT)/integ/detect_model23/rtl
STIM_DIR := $(REPO_ROOT)/integ/detect_model23/stim

# Multi-test mode (one test) gives us the per-test FLAGS hook needed to inject
# the repo-relative stim path as a portable -DSTIM_DIR (no hardcoded machine
# path — works on any checkout, e.g. PD's).
TESTS := head

COMMON_SRCS := \
  $(IP_DIR)/i32_to_fp16/rtl/i32_to_fp16.sv \
  $(IP_DIR)/fp16_fma/rtl/fp16_fma.sv \
  $(IP_DIR)/reduce_max_n/rtl/reduce_max_n.sv \
  $(IP_DIR)/box_affine/rtl/box_affine.sv \
  $(IP_DIR)/dequant_n/rtl/dequant_n.sv \
  $(IP_DIR)/topk_fp16/rtl/topk_fp16.sv \
  $(S_DIR)/detect_head.sv \
  $(CURDIR)/detect_head_tb.sv \
  $(CURDIR)/detect_head_test.cc

head_TOP   := detect_head_tb
head_SRCS  := $(COMMON_SRCS)
# stim dir is derived at runtime from the test source's __FILE__ path
# (portable; no hardcoded path, no working-directory assumption).

LINT_SRCS := \
  $(IP_DIR)/i32_to_fp16/rtl/i32_to_fp16.sv \
  $(IP_DIR)/fp16_fma/rtl/fp16_fma.sv \
  $(IP_DIR)/reduce_max_n/rtl/reduce_max_n.sv \
  $(IP_DIR)/box_affine/rtl/box_affine.sv \
  $(IP_DIR)/dequant_n/rtl/dequant_n.sv \
  $(IP_DIR)/topk_fp16/rtl/topk_fp16.sv \
  $(S_DIR)/detect_head.sv
LINT_FLAGS := -Wno-MULTITOP -Wno-DECLFILENAME

include $(REPO_ROOT)/mk/verilator.mk

.PHONY: stim
stim: $(STIM_DIR)/.stim.stamp
$(STIM_DIR)/.stim.stamp: $(REPO_ROOT)/integ/detect_model23/extract.py
	python3 $(REPO_ROOT)/integ/detect_model23/extract.py
	touch $@
