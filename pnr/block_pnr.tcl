# Generic per-IP P&R on ASAP7 (RVT, TT). Block name from $env(BLK).
# Run via pnr/route_ips.sh (sets BLK, ulimit). Reports post-route Fmax + area.
set BLK  $env(BLK)
set PDK  /home/marty/Documents/aif/pdk/asap7
set WORK pnr/$BLK

read_lef     $PDK/lef/asap7_tech_1x.lef
read_lef     $PDK/lef/asap7_R_1x.lef
read_liberty $PDK/lib/asap7sc7p5t_RVT_TT_merged.lib
read_verilog $WORK/$BLK.netlist.v
link_design  $BLK

# 1 GHz target (liberty time_unit = 1 ps -> period 1000).
create_clock -name clk -period 1000 [get_ports clk_i]
set_input_delay  -clock clk 50 [all_inputs -no_clocks]
set_output_delay -clock clk 50 [all_outputs]
set_wire_rc -signal -layer M3
set_wire_rc -clock  -layer M5

# Cap fanout so repair_design builds buffer trees for the reset and tie-0/1 nets.
# Without this, a 24×fp16_fma block (box_decode) leaves rst_ni at ~68k fanout and
# tie nets at ~24k, which makes global_route build huge Steiner trees and hang.
set_max_fanout 40 [current_design]

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

puts "=== RESULT $BLK ==="
report_worst_slack -max
report_design_area
write_def $WORK/$BLK.def
exit
