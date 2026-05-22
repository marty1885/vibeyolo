REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

MAC_DIR := $(REPO_ROOT)/hw/ip/mac8/rtl
DOT_DIR := $(REPO_ROOT)/hw/ip/dotN/rtl
I2F_DIR := $(REPO_ROOT)/hw/ip/i32_to_fp16/rtl
FMA_DIR := $(REPO_ROOT)/hw/ip/fp16_fma/rtl
F2I_DIR := $(REPO_ROOT)/hw/ip/fp16_to_i8_sat/rtl
RQ_DIR  := $(REPO_ROOT)/hw/ip/requant/rtl
SIL_DIR := $(REPO_ROOT)/hw/ip/act_silu/rtl
ARQ_DIR := $(REPO_ROOT)/hw/ip/add_rq/rtl
CL_DIR  := $(REPO_ROOT)/hw/ip/conv_layer/rtl
L_DIR   := $(REPO_ROOT)/integ/generated/layer_21_model7/rtl

TB_TOP := layer_21_tb

STIM_DIR := $(REPO_ROOT)/integ/generated/layer_21_model7/stim

RTL_SRCS := \
  $(REPO_ROOT)/integ/generated/scale/scale_pkg_balanced_dv.sv \
  $(STIM_DIR)/s_out_params.sv \
  $(MAC_DIR)/mac8.sv \
  $(DOT_DIR)/dotN.sv \
  $(I2F_DIR)/i32_to_fp16.sv \
  $(FMA_DIR)/fp16_fma.sv \
  $(F2I_DIR)/fp16_to_i8_sat.sv \
  $(RQ_DIR)/requant.sv \
  $(SIL_DIR)/act_silu.sv \
  $(ARQ_DIR)/add_rq.sv \
  $(CL_DIR)/conv_layer.sv \
  $(L_DIR)/layer_21.sv \
  $(CURDIR)/layer_21_tb.sv

CC_SRCS := $(CURDIR)/layer_21_test.cc
LINT_FLAGS := -Wno-UNOPTFLAT -Wno-ASCRANGE -Wno-SELRANGE

include $(REPO_ROOT)/mk/verilator.mk

.PHONY: stim
stim: $(STIM_DIR)/.stim.stamp

# Always re-run extract.py before building the verilator binary so that
# stim/s_out_params.sv is up-to-date. The stamp keeps `make` happy with
# the file-as-prereq pattern; the verilator binary also depends on the
# s_out_params.sv source file directly via RTL_SRCS.
$(STIM_DIR)/.stim.stamp: $(REPO_ROOT)/integ/generated/layer_21_model7/extract.py
	python3 $(REPO_ROOT)/integ/generated/layer_21_model7/extract.py
	touch $@

# Force the binary build to wait on extract.py completion. Without this,
# make -j can race: the binary starts building against the placeholder
# s_out_params.sv before extract.py rewrites it.
$(STIM_DIR)/s_out_params.sv: $(STIM_DIR)/.stim.stamp
	@true

test: stim
