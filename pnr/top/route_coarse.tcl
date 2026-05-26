# COARSE top-level global route — memory-bounded.
# Run ONLY under a ulimit -v cap (see pnr/top/route_coarse.sh): top-level routing
# allocates a die-wide GCell grid, so we restrict to 2 signal layers (M4 pins +
# M5 orthogonal) instead of M2-M7. Worst case the ulimit kills the process, not
# the machine ([[footgun_top_global_route_ooms]]).
set PDK /home/marty/Documents/aif/pdk/asap7
read_lef $PDK/lef/asap7_tech_1x.lef
# NOTE: standard-cell LEF (asap7_R_1x.lef) intentionally NOT read — the top level
# has no std cells, only the block hard-macros in blocks.lef. Skipping it drops
# 212 macro defs from the DB to save memory for the routing grid.
read_lef [expr {[info exists env(LEF)] ? $env(LEF) : "pnr/top/blocks.lef"}]
set DEF [expr {[info exists env(DEF)] ? $env(DEF) : "pnr/top/placed_routable.def"}]
set OUT [expr {[info exists env(OUT)] ? $env(OUT) : "pnr/top/routed_coarse.def"}]
read_def $DEF
source $PDK/make_tracks.tcl

# 2-layer coarse route: M4 (horizontal, pin layer) + M5 (vertical).
set_routing_layers -signal M4-M5
puts "=== coarse global_route (M4-M5) on $DEF ==="
global_route -allow_congestion -verbose
report_design_area
report_wire_length -net * -global_route
write_def $OUT
exit
