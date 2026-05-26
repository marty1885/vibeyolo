# ⚠️  DO NOT RUN — global-routing this top-level floorplan OOMs the machine.
# A multi-mm² die builds a die-wide M2-M7 routing/congestion grid (~1e5 × 1e5
# GCells) that exhausts RAM (same wall as full-chip flat P&R). Top-level routed
# wirelength is estimated analytically instead — see tools/macro_floorplan.py
# (dataflow HPWL on the placement). Per-block routing stays feasible: pnr/<blk>.
puts "Refusing to top-level global_route: OOM risk. Use HPWL estimate."
exit 1
