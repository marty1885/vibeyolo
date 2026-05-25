REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

RTL_DIR := $(REPO_ROOT)/hw/ip/sram_beh/rtl
RTL_SRCS := \
  $(RTL_DIR)/sram_beh.sv \
  $(RTL_DIR)/sram_beh_banked.sv \
  $(CURDIR)/sram_beh_tb.sv
CC_SRCS  := $(CURDIR)/sram_beh_test.cc

# ── Multi-test mode: read latency 1 and 2 (both force NW>1 and ND>1) ─
TESTS := lat1 lat2

lat1_TOP   := sram_beh_tb_lat1
lat1_SRCS  := $(RTL_SRCS) $(CC_SRCS)
lat1_FLAGS := -CFLAGS "-DTB_CONFIG=1"

lat2_TOP   := sram_beh_tb_lat2
lat2_SRCS  := $(RTL_SRCS) $(CC_SRCS)
lat2_FLAGS := -CFLAGS "-DTB_CONFIG=2"

LINT_SRCS := \
  $(RTL_DIR)/sram_beh.sv \
  $(RTL_DIR)/sram_beh_banked.sv
LINT_FLAGS := -Wno-MULTITOP --top-module sram_beh_banked

include $(REPO_ROOT)/mk/verilator.mk
