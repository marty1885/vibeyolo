REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

# Small-but-non-trivial sweep config for sim speed.
WROM_KH ?= 3
WROM_KW ?= 3
WROM_IC ?= 4
WROM_OC ?= 8

HEX_PREFIX := $(CURDIR)/build/wrom_small
HEX_FILES  := $(HEX_PREFIX).w.hex $(HEX_PREFIX).s.hex $(HEX_PREFIX).b.hex

# We bypass mk/verilator.mk's single-test mode and drive Verilator
# directly here because we need to (a) generate the $readmemh hex files
# before elaboration and (b) inject parameter overrides into the build
# command. The structure mirrors verilator.mk's single-test rule.

VERILATOR ?= verilator
VERILATOR_JOBS ?= $(shell procs=$$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1); jobs=$$((procs * 4 / 5)); if [ $$jobs -lt 1 ]; then jobs=1; fi; echo $$jobs)

BUILD_DIR := $(CURDIR)/build
OBJ_DIR   := $(BUILD_DIR)/obj

TB_TOP   := weight_rom_tb
RTL_SRCS := \
  $(REPO_ROOT)/hw/ip/weight_rom/rtl/weight_rom.sv \
  $(REPO_ROOT)/hw/ip/weight_rom/rtl/weight_rom_ref.sv \
  $(CURDIR)/weight_rom_tb.sv
CC_SRCS  := $(CURDIR)/weight_rom_test.cc

COMMON_FLAGS := \
  -sv --cc --exe --build -j $(VERILATOR_JOBS) \
  -Wall \
  --assert \
  --coverage \
  -CFLAGS "-std=c++17 -I$(REPO_ROOT)/dv/common"

PARAM_FLAGS := \
  -GKh=$(WROM_KH) -GKw=$(WROM_KW) -GIc=$(WROM_IC) -GOc=$(WROM_OC) \
  -GInitFile='"$(HEX_PREFIX)"' \
  -CFLAGS "-DWROM_KH=$(WROM_KH) -DWROM_KW=$(WROM_KW) -DWROM_IC=$(WROM_IC) -DWROM_OC=$(WROM_OC)"

.PHONY: test lint clean coverage-report

test: $(OBJ_DIR)/V$(TB_TOP)
	@echo "──── run $(TB_TOP) ────"
	@$< +verilator+coverage+file+$(OBJ_DIR)/coverage.dat
	@if [ -f $(OBJ_DIR)/coverage.dat ]; then \
	  verilator_coverage --annotate $(BUILD_DIR)/coverage \
	    $(OBJ_DIR)/coverage.dat; \
	fi

lint:
	@echo "──── lint $(TB_TOP) ────"
	@$(VERILATOR) --lint-only -sv -Wall $(PARAM_FLAGS) \
	  --top-module $(TB_TOP) \
	  $(RTL_SRCS)

coverage-report:
	@if [ -f $(OBJ_DIR)/coverage.dat ]; then \
	  printf "  %-40s " "$(TB_TOP)"; \
	  verilator_coverage --annotate $(BUILD_DIR)/coverage \
	    $(OBJ_DIR)/coverage.dat \
	    | sed 's/^Total coverage //' | head -1; \
	fi

$(HEX_FILES): $(CURDIR)/gen_hex.py
	@mkdir -p $(dir $(HEX_PREFIX))
	python3 $(CURDIR)/gen_hex.py \
	  --kh $(WROM_KH) --kw $(WROM_KW) --ic $(WROM_IC) --oc $(WROM_OC) \
	  --out $(HEX_PREFIX)

$(OBJ_DIR)/V$(TB_TOP): $(RTL_SRCS) $(CC_SRCS) $(HEX_FILES)
	@mkdir -p $(BUILD_DIR)
	$(VERILATOR) $(COMMON_FLAGS) $(PARAM_FLAGS) \
	  --Mdir $(OBJ_DIR) \
	  --top-module $(TB_TOP) \
	  -o V$(TB_TOP) \
	  $(RTL_SRCS) $(CC_SRCS)

clean:
	rm -rf $(BUILD_DIR)
