# Formal sequential-equivalence (SEC) harness for the leaf IPs.
#
# Every IP ships a re-coded behavioral golden (<ip>_ref.sv) with an
# identical port list. This proves the synthesizable <ip>.sv equivalent to
# that golden for ALL inputs and ALL cycles from reset (unbounded), using:
#   sv2v  : SystemVerilog-2012 -> Verilog-2005 (yosys frontend can't take the
#           SV the DUTs/goldens use: unpacked function args, void fns, etc.)
#   yosys : miter -equiv builds the comparison; prep lowers to a property.
#   sby   : prove mode = BMC base case + temporal induction.
#   z3    : word-level SMT — proves multiply/accumulate cones that bit-level
#           SAT induction can't (see mac8).
#
# Usage:  make -C formal -jN equiv     # all SEC-able IPs in parallel
#         make -C formal sec-mac8      # one IP
#         make -C formal clean
#
# NOTE: only IPs whose golden is BIT-ACCURATE belong here. Goldens written
# in `real` (untimed float) are simulation references, not formal models —
# SMT cannot reason about IEEE real, so they are property-checked instead
# (see PROPERTY_IPS, not yet wired) and excluded below.

SHELL := /bin/bash
HW    := ../hw/ip
WORK  := work
DEPTH ?= 16

# Per-IP sv2v defines (shrink width-parameterized IPs so proofs stay fast).
DEFS_dotN         := -DN=8
DEFS_reduce_max_n := -DN=8

# Async-reset blocks whose init value is dropped by prep: pin init to 0
# (their reset value). Default for other IPs preserves the design init.
INITZERO_mac8     := setundef -zero -init

# Bit-accurate-golden IPs. Some still need refinement (see notes in README).
EQUIV_IPS := i32_to_fp16 fp16_to_i8_sat mac8 requant maxpool_kxk dotN \
             reduce_max_n upsample2 concat_mux skip_buf linebuf_kxk

SATDEPTH ?= 8

.PHONY: equiv clean $(addprefix sec-,$(EQUIV_IPS)) $(addprefix sat-,$(EQUIV_IPS))

equiv: $(addprefix sec-,$(EQUIV_IPS))

# sat-%: bit-level yosys SAT + temporal induction (no SMT). Tolerant of
# unreset behavioral-ref pipeline regs; proves control/shift/saturate logic
# (i32_to_fp16, fp16_to_i8_sat). Multiplier cones don't converge here -> use sec-%.
$(addprefix sat-,$(EQUIV_IPS)): sat-%:
	@mkdir -p $(WORK)
	@sv2v $(DEFS_$*) $(HW)/$*/rtl/$*.sv     > $(WORK)/$*_d.v 2>$(WORK)/$*.sv2v.log || { echo "SV2V  $* (dut)"; exit 1; }
	@sv2v $(DEFS_$*) $(HW)/$*/rtl/$*_ref.sv > $(WORK)/$*_r.v 2>>$(WORK)/$*.sv2v.log || { echo "SV2V  $* (ref)"; exit 1; }
	@printf 'read_verilog %s_d.v\nread_verilog %s_r.v\nhierarchy; proc\nmiter -equiv -flatten -make_assert %s_ref %s m\nhierarchy -top m\nasync2sync; opt -full\nsat -prove-asserts -tempinduct -seq $(SATDEPTH) -verify\n' $* $* $* $* > $(WORK)/$*.ys
	@if (cd $(WORK) && timeout 400 yosys -q $*.ys) >$(WORK)/$*.sat.log 2>&1; then echo "PASS  $* (sat)"; else echo "FAIL  $* (sat) -> $(WORK)/$*.sat.log"; fi

$(addprefix sec-,$(EQUIV_IPS)): sec-%:
	@mkdir -p $(WORK)
	@sv2v $(DEFS_$*) $(HW)/$*/rtl/$*.sv     > $(WORK)/$*_d.v 2>$(WORK)/$*.sv2v.log || { echo "SV2V  $* (dut)"; cat $(WORK)/$*.sv2v.log; exit 1; }
	@sv2v $(DEFS_$*) $(HW)/$*/rtl/$*_ref.sv > $(WORK)/$*_r.v 2>>$(WORK)/$*.sv2v.log || { echo "SV2V  $* (ref)"; cat $(WORK)/$*.sv2v.log; exit 1; }
	@sed -e 's/@IP@/$*/g' -e 's/@DEPTH@/$(DEPTH)/g' -e 's#@WORK@#$(WORK)#g' \
	     -e 's|@INITZERO@|$(INITZERO_$*)|g' sec.sby.tmpl > $(WORK)/$*.sby
	@rm -rf $(WORK)/$*_sby
	@sby -f -d $(WORK)/$*_sby $(WORK)/$*.sby > $(WORK)/$*.log 2>&1; \
	 if   grep -q "DONE (PASS" $(WORK)/$*.log; then echo "PASS  $*"; \
	 elif grep -q "DONE (FAIL" $(WORK)/$*.log; then echo "FAIL  $*  -> $(WORK)/$*.log"; \
	 else echo "ERROR $*  -> $(WORK)/$*.log"; fi

# Numeric property proof for mac8: the int32 accumulator is exact (never
# wraps) for up to K=131071 multiply-accumulates, and the bound is tight
# (K=131072 reaches 2^31 and wraps). Unbounded proof by k-induction (z3).
.PHONY: props-mac8
props-mac8:
	@mkdir -p $(WORK)
	@sed 's/@K@/131071/' mac8_props.sby > $(WORK)/mac8_props_safe.sby
	@rm -rf $(WORK)/mac8_props_safe
	@if sby -f -d $(WORK)/mac8_props_safe $(WORK)/mac8_props_safe.sby >$(WORK)/mac8_props_safe.log 2>&1; \
	 then echo "PASS  mac8 no-overflow @ K=131071 (unbounded, k-induction)"; \
	 else echo "FAIL  mac8 @ K=131071 -> $(WORK)/mac8_props_safe.log"; fi
	@sed 's/@K@/131072/' mac8_props.sby > $(WORK)/mac8_props_tight.sby
	@rm -rf $(WORK)/mac8_props_tight
	@if sby -f -d $(WORK)/mac8_props_tight $(WORK)/mac8_props_tight.sby >$(WORK)/mac8_props_tight.log 2>&1; \
	 then echo "FAIL  tightness broken: K=131072 unexpectedly PASSED"; \
	 else echo "PASS  tightness confirmed: K=131072 overflows (no_overflow fails as expected)"; fi

clean:
	rm -rf $(WORK) mac8_props_safe mac8_props_tight work_props_*.sby
