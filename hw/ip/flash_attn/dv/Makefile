REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

FMA_DIR := $(REPO_ROOT)/hw/ip/fp16_fma/rtl
FA_DIR  := $(REPO_ROOT)/hw/ip/flash_attn/rtl

COMMON_RTL := \
  $(FMA_DIR)/fp16_fma.sv \
  $(FA_DIR)/flash_attn.sv \
  $(FA_DIR)/flash_attn_ref.sv

TEST_CC := $(CURDIR)/flash_attn_test.cc

LINT_SRCS := \
  $(FMA_DIR)/fp16_fma.sv \
  $(FA_DIR)/flash_attn.sv \
  $(FA_DIR)/flash_attn_ref.sv
LINT_FLAGS := -Wno-MULTITOP -Wno-UNOPTFLAT -Wno-ENUMVALUE -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDPARAM -Wno-LATCH -Wno-UNUSEDSIGNAL -Wno-WIDTHCONCAT

TESTS := small dbg mid1 mid2 prod

# ─── dbg: like small but N=16 (4 col-tiles per row-tile) ───
dbg_TOP   := flash_attn_tb_dbg
dbg_SRCS  := $(COMMON_RTL) $(CURDIR)/flash_attn_tb_dbg.sv $(TEST_CC)
dbg_FLAGS := $(LINT_FLAGS) \
  -CFLAGS "-DCFG_HEADER=\\\"Vflash_attn_tb_dbg.h\\\" \
           -DCFG_DUT_T=Vflash_attn_tb_dbg \
           -DCFG_NAME=\\\"dbg\\\" \
           -DCFG_HEADS=1 -DCFG_N=16 -DCFG_DIM_Q=16 -DCFG_DIM_V=4 \
           -DCFG_FRAMES=2 -DCFG_MAX_CYC=60000 -DCFG_TEMP_FP16=0x31A8"


# ─── small ───
small_TOP   := flash_attn_tb_small
small_SRCS  := $(COMMON_RTL) $(CURDIR)/flash_attn_tb_small.sv $(TEST_CC)
small_FLAGS := $(LINT_FLAGS) \
  -CFLAGS "-DCFG_HEADER=\\\"Vflash_attn_tb_small.h\\\" \
           -DVflash_attn_tb_CFG_h=Vflash_attn_tb_small_h \
           -DCFG_DUT_T=Vflash_attn_tb_small \
           -DCFG_NAME=\\\"small\\\" \
           -DCFG_HEADS=1 -DCFG_N=8 -DCFG_DIM_Q=4 -DCFG_DIM_V=4 \
           -DCFG_FRAMES=4 -DCFG_MAX_CYC=60000 -DCFG_TEMP_FP16=0x31A8"

# ─── mid1 ───
mid1_TOP   := flash_attn_tb_mid1
mid1_SRCS  := $(COMMON_RTL) $(CURDIR)/flash_attn_tb_mid1.sv $(TEST_CC)
mid1_FLAGS := $(LINT_FLAGS) \
  -CFLAGS "-DCFG_HEADER=\\\"Vflash_attn_tb_mid1.h\\\" \
           -DCFG_DUT_T=Vflash_attn_tb_mid1 \
           -DCFG_NAME=\\\"mid1\\\" \
           -DCFG_HEADS=2 -DCFG_N=64 -DCFG_DIM_Q=16 -DCFG_DIM_V=32 \
           -DCFG_FRAMES=2 -DCFG_MAX_CYC=400000 -DCFG_TEMP_FP16=0x3400"

# ─── mid2 ───
mid2_TOP   := flash_attn_tb_mid2
mid2_SRCS  := $(COMMON_RTL) $(CURDIR)/flash_attn_tb_mid2.sv $(TEST_CC)
mid2_FLAGS := $(LINT_FLAGS) \
  -CFLAGS "-DCFG_HEADER=\\\"Vflash_attn_tb_mid2.h\\\" \
           -DCFG_DUT_T=Vflash_attn_tb_mid2 \
           -DCFG_NAME=\\\"mid2\\\" \
           -DCFG_HEADS=1 -DCFG_N=128 -DCFG_DIM_Q=32 -DCFG_DIM_V=32 \
           -DCFG_FRAMES=1 -DCFG_MAX_CYC=400000 -DCFG_TEMP_FP16=0x31A8"

# ─── prod ───
prod_TOP   := flash_attn_tb_prod
prod_SRCS  := $(COMMON_RTL) $(CURDIR)/flash_attn_tb_prod.sv $(TEST_CC)
prod_FLAGS := $(LINT_FLAGS) \
  -CFLAGS "-DCFG_HEADER=\\\"Vflash_attn_tb_prod.h\\\" \
           -DCFG_DUT_T=Vflash_attn_tb_prod \
           -DCFG_NAME=\\\"prod\\\" \
           -DCFG_HEADS=2 -DCFG_N=400 -DCFG_DIM_Q=32 -DCFG_DIM_V=64 \
           -DCFG_FRAMES=1 -DCFG_MAX_CYC=400000 -DCFG_TEMP_FP16=0x31A8"

include $(REPO_ROOT)/mk/verilator.mk
