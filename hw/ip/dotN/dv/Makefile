REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

RTL_DIR  := $(REPO_ROOT)/hw/ip/dotN/rtl
MAC8_DIR := $(REPO_ROOT)/hw/ip/mac8/rtl
TB_SV    := $(CURDIR)/dotN_tb.sv
TEST_CC  := $(CURDIR)/dotN_test.cc

COMMON_SRCS := \
  $(MAC8_DIR)/mac8.sv \
  $(RTL_DIR)/dotN.sv \
  $(RTL_DIR)/dotN_ref.sv \
  $(TB_SV) \
  $(TEST_CC)

# One build per N. We test at least the spec-required pow2 configs.
TESTS := n4 n16

n4_TOP   := dotN_tb
n4_SRCS  := $(COMMON_SRCS)
n4_FLAGS := -GN=4 -CFLAGS "-DDOTN_N=4"

n16_TOP  := dotN_tb
n16_SRCS := $(COMMON_SRCS)
n16_FLAGS := -GN=16 -CFLAGS "-DDOTN_N=16"

# Lint at default N=16; parameter only changes widths and tree depth.
LINT_SRCS := \
  $(MAC8_DIR)/mac8.sv \
  $(RTL_DIR)/dotN.sv \
  $(RTL_DIR)/dotN_ref.sv \
  $(TB_SV)
LINT_FLAGS := -Wno-MULTITOP

include $(REPO_ROOT)/mk/verilator.mk
