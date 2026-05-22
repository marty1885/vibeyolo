// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// yolo26n_top — chip-level top.
//
// ============================================================================
// FROZEN INTERFACE — DO NOT MODIFY.
// PD owns this boundary. All future logic changes happen inside
// `yolo26n_core`. New blocks may extend `yolo26n_core` ports; the chip
// boundary above does not move.
// ============================================================================
//
// Clocking:    single functional domain `clk_i`. A separate `test_clk_i`
//              feeds scan/BIST when `scan_mode_i` is asserted. An internal
//              `prim_clk_gate` drives the core; `scan_en_i` bypasses it.
// Reset:       `rst_ni` async-assert / sync-deassert via internal
//              `prim_rst_sync`. PD swaps the synchronizer for a tech cell.
// I/O:         All AXIS and AXI-Lite signals pass through skid buffers /
//              flop stages at the boundary. Every chip pin is driven by /
//              captured by a register — no combinational paths cross the
//              pad ring.
// DFT:         scan_en, scan_mode, test_clk, scan_in[N_SCAN_CHAINS],
//              scan_out[N_SCAN_CHAINS], bist_run, bist_done, bist_fail.
//              N_SCAN_CHAINS is a top-level parameter (default 8) so PD can
//              renegotiate without touching ports beyond that bus width.
//
// Image in:    AXI4-Stream slave `s_axis_pix`. 2 px/cycle, 3 ch u8 each.
//                TDATA[47:0] = {pix1_b, pix1_g, pix1_r, pix0_b, pix0_g, pix0_r}
//                TDATA[63:48] reserved (tie low at SoC).
//                TUSER[0] = SOF on the first beat of a frame.
//                TLAST    = EOL on the last beat of each row.
// Det out:     AXI4-Stream master `m_axis_det`. 1 det/beat.
//                TDATA[6:0]    class_id
//                TDATA[7]      reserved
//                TDATA[23:8]   score fp16
//                TDATA[39:24]  x1 fp16
//                TDATA[55:40]  y1 fp16
//                TDATA[71:56]  x2 fp16
//                TDATA[87:72]  y2 fp16
//                TDATA[127:88] reserved
//                TLAST = last det of frame.
// TID/TDEST:   NOT implemented on either AXIS. SoC integrator must tie
//              upstream TID/TDEST inputs and ignore the absence on outputs.
// Config:      AXI4-Lite slave, 12-bit addr, 32-bit data. See README for map.
// IRQ:         Level-high; `STATUS.done & CTRL.irq_en`.
// ============================================================================

`ifndef YOLO26N_VERSION_MAJOR
  `define YOLO26N_VERSION_MAJOR 8'd0
`endif
`ifndef YOLO26N_VERSION_MINOR
  `define YOLO26N_VERSION_MINOR 8'd2
`endif

module yolo26n_top #(
  parameter int unsigned N_SCAN_CHAINS = 8
) (
  // --------------------------------------------------------------------------
  // Clock / reset
  // --------------------------------------------------------------------------
  input  logic                       clk_i,
  input  logic                       rst_ni,        // async assert
  input  logic                       test_clk_i,    // free-running, scan/BIST
  input  logic                       clk_gate_en_i, // 1 = leave clock on; 0 = allow ICG to gate when idle

  // --------------------------------------------------------------------------
  // DFT
  // --------------------------------------------------------------------------
  input  logic                       scan_en_i,
  input  logic                       scan_mode_i,
  input  logic  [N_SCAN_CHAINS-1:0]  scan_in_i,
  output logic  [N_SCAN_CHAINS-1:0]  scan_out_o,
  input  logic                       bist_run_i,
  output logic                       bist_done_o,
  output logic                       bist_fail_o,

  // --------------------------------------------------------------------------
  // AXI4-Stream slave — image input
  // --------------------------------------------------------------------------
  input  logic                       s_axis_pix_tvalid,
  output logic                       s_axis_pix_tready,
  input  logic  [63:0]               s_axis_pix_tdata,
  input  logic  [0:0]                s_axis_pix_tuser,   // [0] = SOF
  input  logic                       s_axis_pix_tlast,   // EOL

  // --------------------------------------------------------------------------
  // AXI4-Stream master — detection output
  // --------------------------------------------------------------------------
  output logic                       m_axis_det_tvalid,
  input  logic                       m_axis_det_tready,
  output logic  [127:0]              m_axis_det_tdata,
  output logic                       m_axis_det_tlast,

  // --------------------------------------------------------------------------
  // AXI4-Lite slave — config / status
  // --------------------------------------------------------------------------
  input  logic                       s_axil_awvalid,
  output logic                       s_axil_awready,
  input  logic  [11:0]               s_axil_awaddr,
  input  logic  [2:0]                s_axil_awprot,

  input  logic                       s_axil_wvalid,
  output logic                       s_axil_wready,
  input  logic  [31:0]               s_axil_wdata,
  input  logic  [3:0]                s_axil_wstrb,

  output logic                       s_axil_bvalid,
  input  logic                       s_axil_bready,
  output logic  [1:0]                s_axil_bresp,

  input  logic                       s_axil_arvalid,
  output logic                       s_axil_arready,
  input  logic  [11:0]               s_axil_araddr,
  input  logic  [2:0]                s_axil_arprot,

  output logic                       s_axil_rvalid,
  input  logic                       s_axil_rready,
  output logic  [31:0]               s_axil_rdata,
  output logic  [1:0]                s_axil_rresp,

  // --------------------------------------------------------------------------
  // Misc
  // --------------------------------------------------------------------------
  output logic                       irq_o
);

  // ==========================================================================
  // Reset synchronizer (PD: swap for tech cell)
  // ==========================================================================
  logic rst_sync_n;

  prim_rst_sync u_rst_sync (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .rst_no (rst_sync_n)
  );

  // ==========================================================================
  // Top-level clock gate
  //
  // Gated clock drives the core only. Boundary register slices and the CSR
  // run on the ungated `clk_i` so AXI bus access stays alive even when the
  // pipeline is idle. `scan_en_i` (or `scan_mode_i`) keeps the gate open.
  // ==========================================================================
  logic core_busy;
  logic clk_core;
  logic clk_gate_en_eff;
  assign clk_gate_en_eff = clk_gate_en_i | core_busy;

  prim_clk_gate u_clk_gate_core (
    .clk_i     (clk_i),
    .en_i      (clk_gate_en_eff),
    .test_en_i (scan_en_i | scan_mode_i),
    .clk_o     (clk_core)
  );

  // ==========================================================================
  // Boundary register slices — every AXIS pin driven by / captured into flops.
  // ==========================================================================
  logic         pix_in_tvalid;
  logic         pix_in_tready;
  logic [63:0]  pix_in_tdata;
  logic [0:0]   pix_in_tuser;
  logic         pix_in_tlast;

  axis_skid #(.DATA_W(64), .USER_W(1)) u_pix_in_skid (
    .clk_i      (clk_i),
    .rst_ni     (rst_sync_n),
    .s_tvalid_i (s_axis_pix_tvalid),
    .s_tready_o (s_axis_pix_tready),
    .s_tdata_i  (s_axis_pix_tdata),
    .s_tuser_i  (s_axis_pix_tuser),
    .s_tlast_i  (s_axis_pix_tlast),
    .m_tvalid_o (pix_in_tvalid),
    .m_tready_i (pix_in_tready),
    .m_tdata_o  (pix_in_tdata),
    .m_tuser_o  (pix_in_tuser),
    .m_tlast_o  (pix_in_tlast)
  );

  logic         det_out_tvalid;
  logic         det_out_tready;
  logic [127:0] det_out_tdata;
  logic         det_out_tlast;

  axis_skid #(.DATA_W(128), .USER_W(1)) u_det_out_skid (
    .clk_i      (clk_i),
    .rst_ni     (rst_sync_n),
    .s_tvalid_i (det_out_tvalid),
    .s_tready_o (det_out_tready),
    .s_tdata_i  (det_out_tdata),
    .s_tuser_i  (1'b0),
    .s_tlast_i  (det_out_tlast),
    .m_tvalid_o (m_axis_det_tvalid),
    .m_tready_i (m_axis_det_tready),
    .m_tdata_o  (m_axis_det_tdata),
    /* verilator lint_off PINCONNECTEMPTY */
    .m_tuser_o  ( ),
    /* verilator lint_on PINCONNECTEMPTY */
    .m_tlast_o  (m_axis_det_tlast)
  );

  // ==========================================================================
  // AXI-Lite — CSR is already fully synchronous w/ depth-1 handshake; pin
  // it to ungated `clk_i` so software can poll STATUS even when the core
  // is clock-gated.
  // ==========================================================================
  logic        cfg_start_pulse;
  logic        cfg_abort_pulse;
  logic        cfg_irq_en;
  logic        cfg_emit_empty;
  logic [7:0]  cfg_in_zp_a;
  logic [15:0] cfg_topk;
  logic [15:0] cfg_score_thresh;

  logic        sts_busy;
  logic        sts_done;
  logic        sts_err;
  logic [7:0]  sts_frame_id;

  yolo26n_csr u_csr (
    .clk_i          (clk_i),
    .rst_ni         (rst_sync_n),

    .s_axil_awvalid (s_axil_awvalid),
    .s_axil_awready (s_axil_awready),
    .s_axil_awaddr  (s_axil_awaddr),
    .s_axil_awprot  (s_axil_awprot),
    .s_axil_wvalid  (s_axil_wvalid),
    .s_axil_wready  (s_axil_wready),
    .s_axil_wdata   (s_axil_wdata),
    .s_axil_wstrb   (s_axil_wstrb),
    .s_axil_bvalid  (s_axil_bvalid),
    .s_axil_bready  (s_axil_bready),
    .s_axil_bresp   (s_axil_bresp),
    .s_axil_arvalid (s_axil_arvalid),
    .s_axil_arready (s_axil_arready),
    .s_axil_araddr  (s_axil_araddr),
    .s_axil_arprot  (s_axil_arprot),
    .s_axil_rvalid  (s_axil_rvalid),
    .s_axil_rready  (s_axil_rready),
    .s_axil_rdata   (s_axil_rdata),
    .s_axil_rresp   (s_axil_rresp),

    .o_start_pulse  (cfg_start_pulse),
    .o_abort_pulse  (cfg_abort_pulse),
    .o_irq_en       (cfg_irq_en),
    .o_emit_empty   (cfg_emit_empty),
    .o_in_zp_a      (cfg_in_zp_a),
    .o_topk         (cfg_topk),
    .o_score_thresh (cfg_score_thresh),

    .i_busy         (sts_busy),
    .i_done         (sts_done),
    .i_err          (sts_err),
    .i_frame_id     (sts_frame_id),

    .i_id_magic     (32'h59_4F_4C_4F),
    .i_version      ({16'h0, `YOLO26N_VERSION_MAJOR, `YOLO26N_VERSION_MINOR})
  );

  assign irq_o     = cfg_irq_en & sts_done;
  assign core_busy = sts_busy;

  // ==========================================================================
  // Core — keep_hierarchy so PD can floorplan the heavy datapath as a block.
  // ==========================================================================
  (* keep_hierarchy = "yes" *)
  yolo26n_core u_core (
    .clk_i          (clk_core),
    .rst_ni         (rst_sync_n),

    .pix_tvalid_i   (pix_in_tvalid),
    .pix_tready_o   (pix_in_tready),
    .pix_tdata_i    (pix_in_tdata[47:0]),
    .pix_sof_i      (pix_in_tuser[0]),
    .pix_eol_i      (pix_in_tlast),

    .det_tvalid_o   (det_out_tvalid),
    .det_tready_i   (det_out_tready),
    .det_tdata_o    (det_out_tdata),
    .det_tlast_o    (det_out_tlast),

    .start_pulse_i  (cfg_start_pulse),
    .abort_pulse_i  (cfg_abort_pulse),
    .emit_empty_i   (cfg_emit_empty),
    .in_zp_a_i      (cfg_in_zp_a),
    .topk_i         (cfg_topk),
    .score_thresh_i (cfg_score_thresh),

    .busy_o         (sts_busy),
    .done_o         (sts_done),
    .err_o          (sts_err),
    .frame_id_o     (sts_frame_id)
  );

  // ==========================================================================
  // DFT placeholders.
  //
  // The scan chains will be stitched by the DFT/scan-insertion flow after PD
  // partitioning. Today these signals fan in/out as flopped no-ops so the
  // boundary timing is closed regardless of internal status.
  // BIST hooks land when the SRAM macros are committed (see MEMORIES.md).
  // ==========================================================================
  logic [N_SCAN_CHAINS-1:0] scan_out_q;
  logic                     bist_done_q;
  logic                     bist_fail_q;

  always_ff @(posedge test_clk_i or negedge rst_sync_n) begin
    if (!rst_sync_n) begin
      scan_out_q  <= '0;
      bist_done_q <= 1'b0;
      bist_fail_q <= 1'b0;
    end else begin
      scan_out_q  <= scan_in_i;   // bypass; real chains stitched by DFT flow
      bist_done_q <= bist_run_i;
      bist_fail_q <= 1'b0;
    end
  end

  assign scan_out_o  = scan_out_q;
  assign bist_done_o = bist_done_q;
  assign bist_fail_o = bist_fail_q;

  // unused tie-offs — kept so lint stays clean while features land
  logic _unused;
  assign _unused = ^{scan_mode_i, pix_in_tdata[63:48]};

endmodule
