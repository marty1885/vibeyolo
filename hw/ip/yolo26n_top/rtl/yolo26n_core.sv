// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// yolo26n_core — implementation shell behind the frozen `yolo26n_top` boundary.
//
// This module's PORTS are the contract between PD/top and the integration team:
// they are stable, but the BODY is expected to grow as block-level glue lands
// (SPPF, upsample, attention, detect head, 102 conv layers, skip-FIFOs).
//
// Current state: skeletal pass-through. Accepts the image stream, drains it,
// and emits zero detections per frame (TLAST asserted with TVALID when
// `emit_empty_i` is set). This lets PD synthesize the full chip boundary today
// while integration continues underneath.
//
// Naming and the {pix, det, ctrl, sts} groupings here MUST stay aligned with
// `yolo26n_top.sv`. PD does not look at this file; integration owns it.

module yolo26n_core (
  input  logic         clk_i,
  input  logic         rst_ni,

  // Image input (de-AXIS'd: TDATA[47:0] only, plus SOF/EOL sidebands)
  input  logic         pix_tvalid_i,
  output logic         pix_tready_o,
  input  logic [47:0]  pix_tdata_i,   // {pix1_b,pix1_g,pix1_r, pix0_b,pix0_g,pix0_r}
  input  logic         pix_sof_i,
  input  logic         pix_eol_i,

  // Detection output
  output logic         det_tvalid_o,
  input  logic         det_tready_i,
  output logic [127:0] det_tdata_o,
  output logic         det_tlast_o,

  // Ctrl
  input  logic         start_pulse_i,
  input  logic         abort_pulse_i,
  input  logic         emit_empty_i,
  input  logic [7:0]   in_zp_a_i,
  input  logic [15:0]  topk_i,
  input  logic [15:0]  score_thresh_i,

  // Status
  output logic         busy_o,
  output logic         done_o,
  output logic         err_o,
  output logic [7:0]   frame_id_o
);

  // ==========================================================================
  // TODO(integration): instantiate
  //   - L0..L101 conv layer wrappers under `scale_pkg::LAYER_<i>_*`
  //   - SPPF block between L31 and L32
  //   - upsample+concat at neck P4->P3 and P3->detect
  //   - PSA / A2C2f attention around model.10 and model.22
  //   - detect head + learned top-K across {80,40,20}^2
  //   - skip-connection FIFOs (P3 ~410KB, P4 ~205KB)
  //   - per-layer weight/scale/bias ROMs and broadcast network
  //
  // The interface above is frozen by `yolo26n_top.sv` and PD constraints; do
  // not change port names, widths, or directions.
  // ==========================================================================

  // --------------------------------------------------------------------------
  // Skeletal placeholder: drain pixels, emit empty frame on each EOF.
  // --------------------------------------------------------------------------
  typedef enum logic [1:0] {
    S_IDLE, S_INGEST, S_EMIT, S_DONE
  } state_e;

  state_e      state_q, state_d;
  logic        frame_active_q;
  logic [7:0]  frame_id_q;
  logic        err_q;

  // accept image whenever core is running
  assign pix_tready_o = (state_q == S_INGEST) | (state_q == S_IDLE);

  // detection output: one terminating empty beat per frame when emit_empty
  assign det_tvalid_o = (state_q == S_EMIT);
  assign det_tlast_o  = (state_q == S_EMIT);
  assign det_tdata_o  = 128'h0;

  always_comb begin
    state_d = state_q;
    unique case (state_q)
      S_IDLE: begin
        if (start_pulse_i) state_d = S_INGEST;
      end
      S_INGEST: begin
        // crude EOF: rely on driver to assert one frame's worth of pix beats
        // and drop TVALID. A future revision replaces this with a pixel
        // counter sized to scale_pkg::LAYER_0_H * LAYER_0_W / 2.
        if (abort_pulse_i)                           state_d = S_DONE;
        else if (!pix_tvalid_i && frame_active_q && pix_eol_i_q)
                                                     state_d = emit_empty_i ? S_EMIT : S_DONE;
      end
      S_EMIT: begin
        if (det_tready_i) state_d = S_DONE;
      end
      S_DONE: begin
        state_d = S_IDLE;
      end
      default: state_d = S_IDLE;
    endcase
  end

  logic pix_eol_i_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q        <= S_IDLE;
      frame_active_q <= 1'b0;
      frame_id_q     <= 8'h0;
      err_q          <= 1'b0;
      pix_eol_i_q    <= 1'b0;
    end else begin
      state_q     <= state_d;
      pix_eol_i_q <= pix_eol_i & pix_tvalid_i & pix_tready_o;

      if (pix_tvalid_i & pix_tready_o & pix_sof_i)
        frame_active_q <= 1'b1;
      else if (state_q == S_DONE)
        frame_active_q <= 1'b0;

      if (state_q == S_DONE) frame_id_q <= frame_id_q + 8'd1;

      if (abort_pulse_i) err_q <= 1'b1;
    end
  end

  assign busy_o     = (state_q != S_IDLE);
  assign done_o     = (state_q == S_DONE);
  assign err_o      = err_q;
  assign frame_id_o = frame_id_q;

  // unused (will be consumed when real datapath lands)
  logic _unused;
  assign _unused = ^{pix_tdata_i, in_zp_a_i, topk_i, score_thresh_i};

endmodule
