REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../..)
endif

S_DIR := $(REPO_ROOT)/integ/upsample_model14/rtl

TB_TOP := upsample_concat_tb

STIM_DIR := $(REPO_ROOT)/integ/upsample_model14/stim

RTL_SRCS := \
  $(S_DIR)/upsample_concat.sv \
  $(CURDIR)/upsample_concat_tb.sv

CC_SRCS := $(CURDIR)/upsample_concat_test.cc
LINT_FLAGS := -Wno-UNOPTFLAT -Wno-ASCRANGE -Wno-SELRANGE -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND
VERILATOR_THREADS ?= 1

include $(REPO_ROOT)/mk/verilator.mk

.PHONY: stim
stim: $(STIM_DIR)/.stim.stamp

$(STIM_DIR)/.stim.stamp: $(REPO_ROOT)/integ/upsample_model14/extract.py
	python3 $(REPO_ROOT)/integ/upsample_model14/extract.py
	touch $@

test: stim
