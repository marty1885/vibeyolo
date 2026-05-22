REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

TB_TOP   := flash_attn_tb
RTL_SRCS := \
  $(REPO_ROOT)/hw/ip/fp16_fma/rtl/fp16_fma.sv \
  $(REPO_ROOT)/hw/ip/flash_attn/rtl/flash_attn.sv \
  $(REPO_ROOT)/hw/ip/flash_attn/rtl/flash_attn_ref.sv \
  $(CURDIR)/flash_attn_tb.sv
CC_SRCS  := $(CURDIR)/flash_attn_test.cc

# Wno-UNOPTFLAT: large combinational reduce_add chain through fp16_add
# functions trips the verilator optimization heuristic. Wno-WIDTHTRUNC
# silences the parametric address width slicing in flash_attn_tb.
LINT_FLAGS := -Wno-UNOPTFLAT -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL

include $(REPO_ROOT)/mk/verilator.mk
