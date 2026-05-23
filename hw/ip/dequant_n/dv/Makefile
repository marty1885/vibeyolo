REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

RTL_DIR := $(REPO_ROOT)/hw/ip/dequant_n/rtl
DEP_DIR := $(REPO_ROOT)/hw/ip
TB_SV   := $(CURDIR)/dequant_n_tb.sv
TEST_CC := $(CURDIR)/dequant_n_test.cc

COMMON_SRCS := \
  $(DEP_DIR)/i32_to_fp16/rtl/i32_to_fp16.sv \
  $(DEP_DIR)/fp16_fma/rtl/fp16_fma.sv \
  $(RTL_DIR)/dequant_n.sv \
  $(RTL_DIR)/dequant_n_ref.sv \
  $(TB_SV) \
  $(TEST_CC)

# n80 is the detect-head logit-gather shape; n16 proves the parameterization.
TESTS := n80 n16

n80_TOP   := dequant_n_tb
n80_SRCS  := $(COMMON_SRCS)
n80_FLAGS := -GN=80 -CFLAGS "-DDQ_N=80"

n16_TOP   := dequant_n_tb
n16_SRCS  := $(COMMON_SRCS)
n16_FLAGS := -GN=16 -CFLAGS "-DDQ_N=16"

LINT_SRCS := \
  $(DEP_DIR)/i32_to_fp16/rtl/i32_to_fp16.sv \
  $(DEP_DIR)/fp16_fma/rtl/fp16_fma.sv \
  $(RTL_DIR)/dequant_n.sv \
  $(RTL_DIR)/dequant_n_ref.sv \
  $(TB_SV)
LINT_FLAGS := -Wno-MULTITOP

include $(REPO_ROOT)/mk/verilator.mk
