#!/usr/bin/env bash
# Route individual logic IPs on ASAP7, one at a time (sequential -> bounded RAM).
# Flow per IP: resolve synth RTL from dv/Makefile RTL_SRCS (drop _tb/_ref) ->
# sv2v -> yosys (synth+abc map, 1ns) -> OpenROAD place+CTS+global_route.
# Each OpenROAD run is capped with `ulimit -v` so a blowup kills the process,
# not the machine.  Usage: pnr/route_ips.sh [ip1 ip2 ...]
set -u
cd "$(git rev-parse --show-toplevel)"
PDK=/home/marty/Documents/aif/pdk/asap7
LIB=$PDK/lib/asap7sc7p5t_RVT_TT_merged.lib
MEM_KB=22000000                       # ~21 GB virtual-memory cap per tool

IPS=("$@")
if [ ${#IPS[@]} -eq 0 ]; then
  IPS=(fp16_fma i32_to_fp16 fp16_to_i8_sat mac8 requant dequant_n \
       reduce_max_n act_silu act_sigmoid box_affine box_decode softmax16 add_rq)
fi

SUMMARY=pnr/IP_PNR_SUMMARY.md
echo "| IP | cells | area µm² | WNS ps | Fmax GHz | status |"  >  $SUMMARY
echo "|----|------:|---------:|-------:|---------:|--------|"   >> $SUMMARY

for BLK in "${IPS[@]}"; do
  echo "==================== $BLK ===================="
  WORK=pnr/$BLK; mkdir -p "$WORK"
  MK=hw/ip/$BLK/dv/Makefile
  if [ ! -f "$MK" ]; then echo "| $BLK | - | - | - | - | no dv/Makefile |" >> $SUMMARY; continue; fi

  # resolve the synth file list via make, trying the var names used across IPs
  # (LINT_SRCS is curated RTL-only; else RTL_SRCS / COMMON_SRCS). Drop _tb/_ref.
  SRCS=""
  for VAR in LINT_SRCS RTL_SRCS COMMON_SRCS; do
    SRCS=$(make -C "hw/ip/$BLK/dv" -f Makefile \
           -f <(printf 'printsrcs:\n\t@echo $(%s)\n' "$VAR") printsrcs 2>/dev/null \
           | tr ' ' '\n' | grep '\.sv$' | grep -vE '_tb\.sv$|_ref\.sv$')
    [ -n "$SRCS" ] && break
  done
  if [ -z "$SRCS" ]; then echo "| $BLK | - | - | - | - | no RTL_SRCS |" >> $SUMMARY; continue; fi
  echo "  srcs: $(echo $SRCS | xargs -n1 basename | tr '\n' ' ')"

  # 1) sv2v -> single Verilog-2005 file
  if ! sv2v $SRCS > "$WORK/$BLK.v" 2> "$WORK/sv2v.log"; then
    echo "| $BLK | - | - | - | - | sv2v fail |" >> $SUMMARY; continue; fi

  # 2) yosys synth + ASAP7 map (1 ns target). HIERARCHICAL (no -flatten): ABC
  #    maps each unique submodule (e.g. fp16_fma) ONCE and reuses it, instead of
  #    re-optimising 24 inlined copies — minutes vs tens of minutes for composite
  #    IPs (box_decode = 24× fp16_fma). OpenROAD flattens the instance tree itself.
  yosys -q -p "
    read_verilog $WORK/$BLK.v
    synth -top $BLK
    dfflibmap -liberty $LIB
    abc -liberty $LIB -D 1000
    opt_clean -purge
    write_verilog $WORK/$BLK.netlist.v
  " > "$WORK/yosys.log" 2>&1
  if [ ! -s "$WORK/$BLK.netlist.v" ]; then
    echo "| $BLK | - | - | - | - | yosys fail |" >> $SUMMARY; continue; fi

  # 3) OpenROAD place+route, memory-capped + hard time-out (no more session hangs)
  ( ulimit -v $MEM_KB; BLK=$BLK timeout 600 openroad -exit pnr/block_pnr.tcl ) \
      > "$WORK/run.log" 2>&1
  rc=$?
  if [ $rc -eq 124 ]; then
    echo "| $BLK | - | - | - | - | route timeout (>10min) |" >> $SUMMARY; continue; fi
  if [ $rc -ne 0 ]; then
    echo "| $BLK | - | - | - | - | openroad rc=$rc (cap/err) |" >> $SUMMARY; continue; fi

  # 4) scrape results
  area=$(grep -iE "Design area" "$WORK/run.log" | grep -oE "[0-9]+ u" | grep -oE "[0-9]+" | head -1)
  cells=$(grep -ciE "^  - |_ASAP7" "$WORK/$BLK.netlist.v" 2>/dev/null)
  wns=$(grep -A2 "worst slack" "$WORK/run.log" | grep -oE "\-?[0-9]+\.[0-9]+" | head -1)
  # WNS (ps) at 1000 ps period -> Fmax = 1000/(1000-WNS)
  fmax=$(awk -v w="${wns:-0}" 'BEGIN{p=1000; printf "%.2f", 1000/(p - w)}')
  echo "| $BLK | ${cells:-?} | ${area:-?} | ${wns:-?} | $fmax | routed |" >> $SUMMARY
  echo "  -> WNS ${wns:-?} ps, area ${area:-?} µm², Fmax ${fmax} GHz"
done

echo; echo "=== summary ==="; cat $SUMMARY
