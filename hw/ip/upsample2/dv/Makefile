REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

RTL_DIR := $(REPO_ROOT)/hw/ip/upsample2/rtl
RTL_SRCS := \
  $(RTL_DIR)/upsample2.sv \
  $(RTL_DIR)/upsample2_ref.sv \
  $(CURDIR)/upsample2_tb.sv
CC_SRCS  := $(CURDIR)/upsample2_test.cc

# ── Multi-test mode: W=4 and W=8 ────────────────────────
TESTS := w4 w8

w4_TOP   := upsample2_tb_w4
w4_SRCS  := $(RTL_SRCS) $(CC_SRCS)
w4_FLAGS := -CFLAGS "-DTB_WIDTH=4"

w8_TOP   := upsample2_tb_w8
w8_SRCS  := $(RTL_SRCS) $(CC_SRCS)
w8_FLAGS := -CFLAGS "-DTB_WIDTH=8"

LINT_SRCS := \
  $(RTL_DIR)/upsample2.sv \
  $(RTL_DIR)/upsample2_ref.sv
LINT_FLAGS := -Wno-MULTITOP --top-module upsample2

include $(REPO_ROOT)/mk/verilator.mk
