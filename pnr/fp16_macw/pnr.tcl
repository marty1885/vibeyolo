# OpenROAD P&R — fp16_macw on ASAP7 (RVT, TT). Real floorplan + post-route timing.
# Run: openroad -exit pnr/fp16_macw/pnr.tcl   (cwd = repo root)
set PDK  /home/marty/Documents/aif/pdk/asap7
set WORK pnr/fp16_macw

read_lef     $PDK/lef/asap7_tech_1x.lef
read_lef     $PDK/lef/asap7_R_1x.lef
read_liberty $PDK/lib/asap7sc7p5t_RVT_TT_merged.lib
read_verilog $WORK/fp16_macw.netlist.v
link_design  fp16_macw

# 1 GHz target (liberty time_unit = 1 ps → period 1000).
create_clock -name clk -period 1000 [get_ports clk_i]
set_input_delay  -clock clk 50 [all_inputs -no_clocks]
set_output_delay -clock clk 50 [all_outputs]

# ASAP7 per-layer RC (ORFS values) so parasitic estimation is meaningful.
set_wire_rc -signal -layer M3
set_wire_rc -clock  -layer M5

# ── floorplan ──────────────────────────────────────────────────────────────
initialize_floorplan -utilization 55 -aspect_ratio 1.0 \
                     -core_space 0.5 -site asap7sc7p5t
source $PDK/make_tracks.tcl

# ── placement ──────────────────────────────────────────────────────────────
place_pins -hor_layers M4 -ver_layers M5
global_placement -density 0.65 -pad_left 1 -pad_right 1
estimate_parasitics -placement
puts "=== POST-GLOBAL-PLACE TIMING (pre-repair) ==="
report_worst_slack -max
report_tns

# standard timing repair: fix max-cap/slew/fanout + buffer/resize for setup
repair_design
detailed_placement
check_placement -verbose

# ── clock tree ───────────────────────────────────────────────────────────────
clock_tree_synthesis -buf_list BUFx4_ASAP7_75t_R -root_buf BUFx4_ASAP7_75t_R \
                     -sink_clustering_enable
set_propagated_clock [all_clocks]
detailed_placement

# ── route ────────────────────────────────────────────────────────────────────
set_routing_layers -signal M2-M7 -clock M2-M7
global_route
estimate_parasitics -global_routing

puts "=== POST-ROUTE TIMING (estimated parasitics, RC from global route) ==="
report_worst_slack -max
report_tns
report_checks -path_delay max -digits 3

# ── area / utilization ───────────────────────────────────────────────────────
puts "=== AREA / UTILISATION ==="
report_design_area

# ── floorplan image + DEF ─────────────────────────────────────────────────────
write_def $WORK/fp16_macw.def
catch {gui::save_image $WORK/fp16_macw_floorplan.png} err
puts "save_image: $err"
exit
