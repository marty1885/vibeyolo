# =============================================================================
# yolo26n_top.sdc — starter constraints
#
# Initial SDC for PD bringup. Numbers are placeholders (1 GHz functional /
# 10 MHz scan / 30% IO budget); PD will tighten with library data and the
# final floorplan. Frozen ports below match yolo26n_top.sv exactly.
# =============================================================================

# -----------------------------------------------------------------------------
# Clocks
# -----------------------------------------------------------------------------
create_clock -name clk      -period 1.000 [get_ports clk_i]
create_clock -name test_clk -period 100.0 [get_ports test_clk_i]

set_clock_groups -asynchronous -group {clk} -group {test_clk}

# Generated clock from the integrated clock gate
create_generated_clock -name clk_core \
    -source [get_ports clk_i] -divide_by 1 \
    [get_pins u_clk_gate_core/clk_o]

# -----------------------------------------------------------------------------
# Reset
# -----------------------------------------------------------------------------
set_false_path -from [get_ports rst_ni] -to [get_pins u_rst_sync/sync_q_reg[*]/CDN]

# -----------------------------------------------------------------------------
# DFT
# -----------------------------------------------------------------------------
set_case_analysis 0 [get_ports scan_mode_i]
set_case_analysis 0 [get_ports scan_en_i]
# scan-mode timing is closed by the scan-insertion flow; mark as a mode signal.

# -----------------------------------------------------------------------------
# I/O delays — placeholder 30% of clock period in each direction.
# Adjust once pad library is in place.
# -----------------------------------------------------------------------------
set io_budget 0.300

set in_ports  [list \
    s_axis_pix_tvalid s_axis_pix_tdata s_axis_pix_tuser s_axis_pix_tlast \
    m_axis_det_tready \
    s_axil_awvalid s_axil_awaddr s_axil_awprot \
    s_axil_wvalid  s_axil_wdata  s_axil_wstrb \
    s_axil_bready \
    s_axil_arvalid s_axil_araddr s_axil_arprot \
    s_axil_rready \
    clk_gate_en_i bist_run_i]

set out_ports [list \
    s_axis_pix_tready \
    m_axis_det_tvalid m_axis_det_tdata m_axis_det_tlast \
    s_axil_awready s_axil_wready s_axil_bvalid s_axil_bresp \
    s_axil_arready s_axil_rvalid s_axil_rdata s_axil_rresp \
    irq_o bist_done_o bist_fail_o]

set_input_delay  -clock clk $io_budget [get_ports $in_ports]
set_output_delay -clock clk $io_budget [get_ports $out_ports]

# Scan-chain ports run on test_clk
set_input_delay  -clock test_clk $io_budget [get_ports scan_in_i[*]]
set_output_delay -clock test_clk $io_budget [get_ports scan_out_o[*]]

# -----------------------------------------------------------------------------
# False paths — config straps that don't need timing closure
# -----------------------------------------------------------------------------
set_false_path -through [get_pins u_csr/o_in_zp_a[*]]
set_false_path -through [get_pins u_csr/o_topk[*]]
set_false_path -through [get_pins u_csr/o_score_thresh[*]]
set_false_path -through [get_pins u_csr/o_emit_empty]
set_false_path -through [get_pins u_csr/o_irq_en]

# -----------------------------------------------------------------------------
# keep_hierarchy
# -----------------------------------------------------------------------------
set_dont_touch [get_cells u_core]
