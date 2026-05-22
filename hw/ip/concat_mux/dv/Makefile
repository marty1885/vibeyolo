REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

RTL_DIR := $(REPO_ROOT)/hw/ip/concat_mux/rtl
RTL_SRCS := \
  $(RTL_DIR)/concat_mux.sv \
  $(RTL_DIR)/concat_mux_ref.sv \
  $(CURDIR)/concat_mux_tb.sv
CC_SRCS  := $(CURDIR)/concat_mux_test.cc

# ── Multi-test mode ─────────────────────────────────────
TESTS := c2_3_p4 c8_16_p8

c2_3_p4_TOP   := concat_mux_tb_c2_3_p4
c2_3_p4_SRCS  := $(RTL_SRCS) $(CC_SRCS)
c2_3_p4_FLAGS := -CFLAGS "-DTB_CA=2 -DTB_CB=3 -DTB_PIXELS=4"

c8_16_p8_TOP   := concat_mux_tb_c8_16_p8
c8_16_p8_SRCS  := $(RTL_SRCS) $(CC_SRCS)
c8_16_p8_FLAGS := -CFLAGS "-DTB_CA=8 -DTB_CB=16 -DTB_PIXELS=8"

LINT_SRCS := \
  $(RTL_DIR)/concat_mux.sv \
  $(RTL_DIR)/concat_mux_ref.sv
LINT_FLAGS := -Wno-MULTITOP --top-module concat_mux

include $(REPO_ROOT)/mk/verilator.mk
