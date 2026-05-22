REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../..)
endif

LB_DIR  := $(REPO_ROOT)/hw/ip/linebuf_kxk/rtl
MP_DIR  := $(REPO_ROOT)/hw/ip/maxpool_kxk/rtl
S_DIR   := $(REPO_ROOT)/integ/sppf_model9/rtl

TB_TOP := sppf_tb

STIM_DIR := $(REPO_ROOT)/integ/sppf_model9/stim

RTL_SRCS := \
  $(LB_DIR)/linebuf_kxk.sv \
  $(MP_DIR)/maxpool_kxk.sv \
  $(S_DIR)/sppf.sv \
  $(CURDIR)/sppf_tb.sv

CC_SRCS := $(CURDIR)/sppf_test.cc
LINT_FLAGS := -Wno-UNOPTFLAT -Wno-ASCRANGE -Wno-SELRANGE -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND
VERILATOR_THREADS ?= 1

include $(REPO_ROOT)/mk/verilator.mk

.PHONY: stim
stim: $(STIM_DIR)/.stim.stamp

$(STIM_DIR)/.stim.stamp: $(REPO_ROOT)/integ/sppf_model9/extract.py
	python3 $(REPO_ROOT)/integ/sppf_model9/extract.py
	touch $@

test: stim
