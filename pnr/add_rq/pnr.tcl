# OpenROAD P&R — add_rq on ASAP7 (RVT, TT). Composite block: 2× i32_to_fp16,
# 4× fp16_fma, 1× fp16_to_i8_sat. Hierarchy preserved → IP-colorable placement.
set PDK  /home/marty/Documents/aif/pdk/asap7
set WORK pnr/add_rq

read_lef     $PDK/lef/asap7_tech_1x.lef
read_lef     $PDK/lef/asap7_R_1x.lef
read_liberty $PDK/lib/asap7sc7p5t_RVT_TT_merged.lib
read_verilog $WORK/add_rq.netlist.v
link_design  add_rq

create_clock -name clk -period 1000 [get_ports clk_i]
set_input_delay  -clock clk 50 [all_inputs -no_clocks]
set_output_delay -clock clk 50 [all_outputs]
set_wire_rc -signal -layer M3
set_wire_rc -clock  -layer M5

initialize_floorplan -utilization 55 -aspect_ratio 1.0 -core_space 0.5 -site asap7sc7p5t
source $PDK/make_tracks.tcl

place_pins -hor_layers M4 -ver_layers M5
global_placement -density 0.65 -pad_left 1 -pad_right 1
estimate_parasitics -placement
repair_design
detailed_placement

clock_tree_synthesis -buf_list BUFx4_ASAP7_75t_R -root_buf BUFx4_ASAP7_75t_R -sink_clustering_enable
set_propagated_clock [all_clocks]
detailed_placement

set_routing_layers -signal M2-M7 -clock M2-M7
global_route
estimate_parasitics -global_routing

puts "=== POST-ROUTE TIMING ==="
report_worst_slack -max
report_tns
puts "=== AREA ==="
report_design_area
write_def $WORK/add_rq.def
exit
