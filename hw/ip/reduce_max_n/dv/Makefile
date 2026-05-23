REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

RTL_DIR := $(REPO_ROOT)/hw/ip/reduce_max_n/rtl
TB_SV   := $(CURDIR)/reduce_max_n_tb.sv
TEST_CC := $(CURDIR)/reduce_max_n_test.cc

COMMON_SRCS := \
  $(RTL_DIR)/reduce_max_n.sv \
  $(RTL_DIR)/reduce_max_n_ref.sv \
  $(TB_SV) \
  $(TEST_CC)

# One build per N: n80 is the production detect-head shape; n16/n4 prove
# the parameterization scales (odd/even tree depths, narrow ports).
TESTS := n80 n16 n4

n80_TOP   := reduce_max_n_tb
n80_SRCS  := $(COMMON_SRCS)
n80_FLAGS := -GN=80 -CFLAGS "-DRM_N=80"

n16_TOP   := reduce_max_n_tb
n16_SRCS  := $(COMMON_SRCS)
n16_FLAGS := -GN=16 -CFLAGS "-DRM_N=16"

n4_TOP    := reduce_max_n_tb
n4_SRCS   := $(COMMON_SRCS)
n4_FLAGS  := -GN=4 -CFLAGS "-DRM_N=4"

# Lint at the default N=80; the parameter only changes the bus width.
LINT_SRCS := \
  $(RTL_DIR)/reduce_max_n.sv \
  $(RTL_DIR)/reduce_max_n_ref.sv \
  $(TB_SV)
LINT_FLAGS := -Wno-MULTITOP

include $(REPO_ROOT)/mk/verilator.mk
