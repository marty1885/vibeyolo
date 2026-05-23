REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

RTL_SRCS_COMMON := \
  $(REPO_ROOT)/hw/ip/topk_fp16/rtl/topk_fp16.sv \
  $(REPO_ROOT)/hw/ip/topk_fp16/rtl/topk_fp16_ref.sv \
  $(CURDIR)/topk_fp16_tb.sv

CC_SRC := $(CURDIR)/topk_fp16_test.cc

TESTS := small big

small_TOP   := topk_fp16_tb_small
small_SRCS  := $(RTL_SRCS_COMMON) $(CC_SRC)
small_FLAGS := -CFLAGS "-DCFG_SMALL" -Wno-DECLFILENAME

big_TOP   := topk_fp16_tb_big
big_SRCS  := $(RTL_SRCS_COMMON) $(CC_SRC)
big_FLAGS := -CFLAGS "-DCFG_BIG" -Wno-DECLFILENAME

LINT_SRCS  := $(RTL_SRCS_COMMON)
LINT_FLAGS := -Wno-MULTITOP -Wno-DECLFILENAME

include $(REPO_ROOT)/mk/verilator.mk
