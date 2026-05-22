REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

RTL_DIR := $(REPO_ROOT)/hw/ip/skip_buf/rtl
RTL_SRCS := \
  $(RTL_DIR)/skip_buf.sv \
  $(RTL_DIR)/skip_buf_ref.sv \
  $(CURDIR)/skip_buf_tb.sv
CC_SRCS  := $(CURDIR)/skip_buf_test.cc

# ── Multi-test mode: small (4x4x4=64) and large (8x8x8=512) ─
TESTS := small large

small_TOP   := skip_buf_tb_small
small_SRCS  := $(RTL_SRCS) $(CC_SRCS)
small_FLAGS := -CFLAGS "-DTB_CONFIG=4"

large_TOP   := skip_buf_tb_large
large_SRCS  := $(RTL_SRCS) $(CC_SRCS)
large_FLAGS := -CFLAGS "-DTB_CONFIG=8"

LINT_SRCS := \
  $(RTL_DIR)/skip_buf.sv \
  $(RTL_DIR)/skip_buf_ref.sv
LINT_FLAGS := -Wno-MULTITOP --top-module skip_buf

include $(REPO_ROOT)/mk/verilator.mk
