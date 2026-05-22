REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

RTL_DIR := $(REPO_ROOT)/hw/ip/maxpool_kxk/rtl
TB_SV   := $(CURDIR)/maxpool_kxk_tb.sv
TEST_CC := $(CURDIR)/maxpool_kxk_test.cc

COMMON_SRCS := \
  $(RTL_DIR)/maxpool_kxk.sv \
  $(RTL_DIR)/maxpool_kxk_ref.sv \
  $(TB_SV) \
  $(TEST_CC)

# One build per K. Multi-test mode in mk/verilator.mk gives us
# separate obj_<t> dirs and per-test FLAGS; we use those to pass both
# the SV parameter override (-GK=...) and the matching C++ define
# (-DK=...) so the test source agrees with the elaborated DUT width.
TESTS := k2 k3 k5

k2_TOP   := maxpool_kxk_tb
k2_SRCS  := $(COMMON_SRCS)
k2_FLAGS := -GK=2 -CFLAGS "-DMAXPOOL_K=2"

k3_TOP   := maxpool_kxk_tb
k3_SRCS  := $(COMMON_SRCS)
k3_FLAGS := -GK=3 -CFLAGS "-DMAXPOOL_K=3"

k5_TOP   := maxpool_kxk_tb
k5_SRCS  := $(COMMON_SRCS)
k5_FLAGS := -GK=5 -CFLAGS "-DMAXPOOL_K=5"

# Lint at the default K=5; the parameter only changes the bus width.
LINT_SRCS := \
  $(RTL_DIR)/maxpool_kxk.sv \
  $(RTL_DIR)/maxpool_kxk_ref.sv \
  $(TB_SV)
LINT_FLAGS := -Wno-MULTITOP

include $(REPO_ROOT)/mk/verilator.mk
