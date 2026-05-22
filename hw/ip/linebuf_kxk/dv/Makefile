REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

RTL_DIR := $(REPO_ROOT)/hw/ip/linebuf_kxk/rtl
RTL_SRCS := \
  $(RTL_DIR)/linebuf_kxk.sv \
  $(RTL_DIR)/linebuf_kxk_ref.sv \
  $(CURDIR)/linebuf_kxk_tb.sv
CC_SRCS  := $(CURDIR)/linebuf_kxk_test.cc

# ── Multi-test mode ─────
TESTS := k3w4 k5w8 k3w8c16 k3w8c64

k3w4_TOP   := linebuf_kxk_tb_k3w4
k3w4_SRCS  := $(RTL_SRCS) $(CC_SRCS)
k3w4_FLAGS := -CFLAGS "-DTB_CFG_K3W4"

k5w8_TOP   := linebuf_kxk_tb_k5w8
k5w8_SRCS  := $(RTL_SRCS) $(CC_SRCS)
k5w8_FLAGS := -CFLAGS "-DTB_CFG_K5W8"

k3w8c16_TOP   := linebuf_kxk_tb_k3w8c16
k3w8c16_SRCS  := $(RTL_SRCS) $(CC_SRCS)
k3w8c16_FLAGS := -CFLAGS "-DTB_CFG_K3W8C16"

k3w8c64_TOP   := linebuf_kxk_tb_k3w8c64
k3w8c64_SRCS  := $(RTL_SRCS) $(CC_SRCS)
k3w8c64_FLAGS := -CFLAGS "-DTB_CFG_K3W8C64"

LINT_SRCS := \
  $(RTL_DIR)/linebuf_kxk.sv \
  $(RTL_DIR)/linebuf_kxk_ref.sv
LINT_FLAGS := -Wno-MULTITOP --top-module linebuf_kxk

include $(REPO_ROOT)/mk/verilator.mk
