REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

MAC_DIR  := $(REPO_ROOT)/hw/ip/mac8/rtl
DOT_DIR  := $(REPO_ROOT)/hw/ip/dotN/rtl
I2F_DIR  := $(REPO_ROOT)/hw/ip/i32_to_fp16/rtl
FMA_DIR  := $(REPO_ROOT)/hw/ip/fp16_fma/rtl
F2I_DIR  := $(REPO_ROOT)/hw/ip/fp16_to_i8_sat/rtl
RQ_DIR   := $(REPO_ROOT)/hw/ip/requant/rtl
SIL_DIR  := $(REPO_ROOT)/hw/ip/act_silu/rtl
ARQ_DIR  := $(REPO_ROOT)/hw/ip/add_rq/rtl
CL_DIR   := $(REPO_ROOT)/hw/ip/conv_layer/rtl

TB_SV    := $(CURDIR)/conv_layer_tb.sv
TEST_CC  := $(CURDIR)/conv_layer_test.cc

COMMON_SRCS := \
  $(MAC_DIR)/mac8.sv \
  $(DOT_DIR)/dotN.sv \
  $(I2F_DIR)/i32_to_fp16.sv \
  $(FMA_DIR)/fp16_fma.sv \
  $(F2I_DIR)/fp16_to_i8_sat.sv \
  $(RQ_DIR)/requant.sv \
  $(SIL_DIR)/act_silu.sv \
  $(ARQ_DIR)/add_rq.sv \
  $(CL_DIR)/conv_layer.sv \
  $(CL_DIR)/conv_layer_ref.sv \
  $(TB_SV) \
  $(TEST_CC)

LINT_SRCS := \
  $(MAC_DIR)/mac8.sv \
  $(DOT_DIR)/dotN.sv \
  $(I2F_DIR)/i32_to_fp16.sv \
  $(FMA_DIR)/fp16_fma.sv \
  $(F2I_DIR)/fp16_to_i8_sat.sv \
  $(RQ_DIR)/requant.sv \
  $(SIL_DIR)/act_silu.sv \
  $(ARQ_DIR)/add_rq.sv \
  $(CL_DIR)/conv_layer.sv \
  $(CL_DIR)/conv_layer_ref.sv \
  $(TB_SV)
LINT_FLAGS := -Wno-MULTITOP -Wno-UNOPTFLAT

TESTS := c1 c2 c3 c4

# C1: layer_2 family — K=1, CIN=32, COUT=32, P_COUT=16, P_CIN=8.
c1_TOP   := conv_layer_tb
c1_SRCS  := $(COMMON_SRCS)
c1_FLAGS := \
  -GCIN=32 -GCOUT=32 -GK=1 -GSTRIDE=1 -GPAD=0 -GP_COUT=16 -GP_CIN=8 \
  -GRESIDUAL=0 -GSILU=1 \
  -CFLAGS "-DCFG_NAME=\\\"C1_K1_32x32\\\" -DCFG_CIN=32 -DCFG_COUT=32 -DCFG_K=1 -DCFG_STRIDE=1 -DCFG_P_COUT=16 -DCFG_P_CIN=8 -DCFG_RESIDUAL=0 -DCFG_SILU=1"

# C2: layer_3 family — K=3, CIN=16, COUT=8, P_COUT=4, P_CIN=4.
c2_TOP   := conv_layer_tb
c2_SRCS  := $(COMMON_SRCS)
c2_FLAGS := \
  -GCIN=16 -GCOUT=8 -GK=3 -GSTRIDE=1 -GPAD=1 -GP_COUT=4 -GP_CIN=4 \
  -GRESIDUAL=0 -GSILU=1 \
  -CFLAGS "-DCFG_NAME=\\\"C2_K3_16x8\\\" -DCFG_CIN=16 -DCFG_COUT=8 -DCFG_K=3 -DCFG_STRIDE=1 -DCFG_P_COUT=4 -DCFG_P_CIN=4 -DCFG_RESIDUAL=0 -DCFG_SILU=1"

# C3: layer_11 family — K=3, STRIDE=2, CIN=128, COUT=128, P_COUT=32, P_CIN=16.
c3_TOP   := conv_layer_tb
c3_SRCS  := $(COMMON_SRCS)
c3_FLAGS := \
  -GCIN=128 -GCOUT=128 -GK=3 -GSTRIDE=2 -GPAD=1 -GP_COUT=32 -GP_CIN=16 \
  -GRESIDUAL=0 -GSILU=1 \
  -CFLAGS "-DCFG_NAME=\\\"C3_K3s2_128x128\\\" -DCFG_CIN=128 -DCFG_COUT=128 -DCFG_K=3 -DCFG_STRIDE=2 -DCFG_P_COUT=32 -DCFG_P_CIN=16 -DCFG_RESIDUAL=0 -DCFG_SILU=1"

# C4: layer_4/9 family — K=3, RESIDUAL=1, CIN=16, COUT=32, P_COUT=8, P_CIN=4.
c4_TOP   := conv_layer_tb
c4_SRCS  := $(COMMON_SRCS)
c4_FLAGS := \
  -GCIN=16 -GCOUT=32 -GK=3 -GSTRIDE=1 -GPAD=1 -GP_COUT=8 -GP_CIN=4 \
  -GRESIDUAL=1 -GSILU=1 \
  -CFLAGS "-DCFG_NAME=\\\"C4_K3_res_16x32\\\" -DCFG_CIN=16 -DCFG_COUT=32 -DCFG_K=3 -DCFG_STRIDE=1 -DCFG_P_COUT=8 -DCFG_P_CIN=4 -DCFG_RESIDUAL=1 -DCFG_SILU=1"

include $(REPO_ROOT)/mk/verilator.mk
