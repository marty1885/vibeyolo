// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// yolo26n_csr — AXI4-Lite (12-bit addr, 32-bit data) register file for the
// chip-level CTRL / STATUS / TOPK / SCRATCH / ID / VERSION registers.
//
// Single-cycle accept on AW+W (both must be valid to fire), single-cycle B.
// AR -> R is single-cycle. No outstanding tracking (depth-1).

module yolo26n_csr (
  input  logic         clk_i,
  input  logic         rst_ni,

  // AXI4-Lite
  input  logic         s_axil_awvalid,
  output logic         s_axil_awready,
  input  logic [11:0]  s_axil_awaddr,
  input  logic [2:0]   s_axil_awprot,

  input  logic         s_axil_wvalid,
  output logic         s_axil_wready,
  input  logic [31:0]  s_axil_wdata,
  input  logic [3:0]   s_axil_wstrb,

  output logic         s_axil_bvalid,
  input  logic         s_axil_bready,
  output logic [1:0]   s_axil_bresp,

  input  logic         s_axil_arvalid,
  output logic         s_axil_arready,
  input  logic [11:0]  s_axil_araddr,
  input  logic [2:0]   s_axil_arprot,

  output logic         s_axil_rvalid,
  input  logic         s_axil_rready,
  output logic [31:0]  s_axil_rdata,
  output logic [1:0]   s_axil_rresp,

  // CTRL outputs
  output logic         o_start_pulse,
  output logic         o_abort_pulse,
  output logic         o_irq_en,
  output logic         o_emit_empty,
  output logic [7:0]   o_in_zp_a,
  output logic [15:0]  o_topk,
  output logic [15:0]  o_score_thresh,

  // STATUS inputs
  input  logic         i_busy,
  input  logic         i_done,
  input  logic         i_err,
  input  logic [7:0]   i_frame_id,

  // RO
  input  logic [31:0]  i_id_magic,
  input  logic [31:0]  i_version
);

  localparam logic [11:0] ADDR_CTRL    = 12'h000;
  localparam logic [11:0] ADDR_STATUS  = 12'h004;
  localparam logic [11:0] ADDR_TOPK    = 12'h008;
  localparam logic [11:0] ADDR_SCRATCH = 12'h00C;
  localparam logic [11:0] ADDR_ID      = 12'h010;
  localparam logic [11:0] ADDR_VERSION = 12'h014;

  // CTRL fields
  logic        ctrl_irq_en_q;
  logic        ctrl_emit_empty_q;
  logic [7:0]  ctrl_in_zp_a_q;

  // TOPK
  logic [15:0] topk_q;
  logic [15:0] score_thresh_q;

  // SCRATCH
  logic [31:0] scratch_q;

  // sticky status
  logic        done_sticky_q;
  logic        err_sticky_q;

  // --------------------------------------------------------------------------
  // Write channel
  // --------------------------------------------------------------------------
  logic write_fire;
  assign write_fire        = s_axil_awvalid & s_axil_wvalid & s_axil_awready & s_axil_wready;
  assign s_axil_awready    = ~s_axil_bvalid;     // simple: accept when no pending response
  assign s_axil_wready     = ~s_axil_bvalid;

  // pulses
  logic start_pulse_d, abort_pulse_d;
  logic done_w1c, err_w1c;

  always_comb begin
    start_pulse_d = 1'b0;
    abort_pulse_d = 1'b0;
    done_w1c      = 1'b0;
    err_w1c       = 1'b0;
    if (write_fire) begin
      unique case (s_axil_awaddr)
        ADDR_CTRL: begin
          if (s_axil_wstrb[0]) begin
            start_pulse_d = s_axil_wdata[0];
            abort_pulse_d = s_axil_wdata[1];
          end
        end
        ADDR_STATUS: begin
          if (s_axil_wstrb[0]) begin
            done_w1c = s_axil_wdata[1];
            err_w1c  = s_axil_wdata[2];
          end
        end
        default: ;
      endcase
    end
  end

  assign o_start_pulse = start_pulse_d;
  assign o_abort_pulse = abort_pulse_d;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ctrl_irq_en_q     <= 1'b0;
      ctrl_emit_empty_q <= 1'b0;
      ctrl_in_zp_a_q    <= 8'd128;
      topk_q            <= 16'd300;
      score_thresh_q    <= 16'h0000;
      scratch_q         <= 32'h0;
      s_axil_bvalid     <= 1'b0;
      s_axil_bresp      <= 2'b00;
      done_sticky_q     <= 1'b0;
      err_sticky_q      <= 1'b0;
    end else begin
      // B response handshake
      if (write_fire) begin
        s_axil_bvalid <= 1'b1;
        s_axil_bresp  <= 2'b00;  // OKAY
      end else if (s_axil_bvalid & s_axil_bready) begin
        s_axil_bvalid <= 1'b0;
      end

      // Sticky status
      if (i_done) done_sticky_q <= 1'b1;
      else if (done_w1c) done_sticky_q <= 1'b0;

      if (i_err) err_sticky_q <= 1'b1;
      else if (err_w1c) err_sticky_q <= 1'b0;

      // Register writes
      if (write_fire) begin
        unique case (s_axil_awaddr)
          ADDR_CTRL: begin
            if (s_axil_wstrb[0]) begin
              ctrl_irq_en_q     <= s_axil_wdata[2];
              ctrl_emit_empty_q <= s_axil_wdata[3];
            end
            if (s_axil_wstrb[2]) ctrl_in_zp_a_q <= s_axil_wdata[23:16];
          end
          ADDR_TOPK: begin
            if (s_axil_wstrb[0]) topk_q[7:0]          <= s_axil_wdata[7:0];
            if (s_axil_wstrb[1]) topk_q[15:8]         <= s_axil_wdata[15:8];
            if (s_axil_wstrb[2]) score_thresh_q[7:0]  <= s_axil_wdata[23:16];
            if (s_axil_wstrb[3]) score_thresh_q[15:8] <= s_axil_wdata[31:24];
          end
          ADDR_SCRATCH: begin
            if (s_axil_wstrb[0]) scratch_q[7:0]   <= s_axil_wdata[7:0];
            if (s_axil_wstrb[1]) scratch_q[15:8]  <= s_axil_wdata[15:8];
            if (s_axil_wstrb[2]) scratch_q[23:16] <= s_axil_wdata[23:16];
            if (s_axil_wstrb[3]) scratch_q[31:24] <= s_axil_wdata[31:24];
          end
          default: ;
        endcase
      end
    end
  end

  assign o_irq_en       = ctrl_irq_en_q;
  assign o_emit_empty   = ctrl_emit_empty_q;
  assign o_in_zp_a      = ctrl_in_zp_a_q;
  assign o_topk         = topk_q;
  assign o_score_thresh = score_thresh_q;

  // --------------------------------------------------------------------------
  // Read channel
  // --------------------------------------------------------------------------
  assign s_axil_arready = ~s_axil_rvalid;

  logic [31:0] rdata_mux;
  always_comb begin
    unique case (s_axil_araddr)
      ADDR_CTRL:    rdata_mux = {8'h0,
                                 ctrl_in_zp_a_q,
                                 12'h0,
                                 ctrl_emit_empty_q,
                                 ctrl_irq_en_q,
                                 2'b00};
      ADDR_STATUS:  rdata_mux = {16'h0,
                                 i_frame_id,
                                 5'h0,
                                 err_sticky_q,
                                 done_sticky_q,
                                 i_busy};
      ADDR_TOPK:    rdata_mux = {score_thresh_q, topk_q};
      ADDR_SCRATCH: rdata_mux = scratch_q;
      ADDR_ID:      rdata_mux = i_id_magic;
      ADDR_VERSION: rdata_mux = i_version;
      default:      rdata_mux = 32'h0;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s_axil_rvalid <= 1'b0;
      s_axil_rdata  <= 32'h0;
      s_axil_rresp  <= 2'b00;
    end else begin
      if (s_axil_arvalid & s_axil_arready) begin
        s_axil_rvalid <= 1'b1;
        s_axil_rdata  <= rdata_mux;
        s_axil_rresp  <= 2'b00;
      end else if (s_axil_rvalid & s_axil_rready) begin
        s_axil_rvalid <= 1'b0;
      end
    end
  end

  // unused tie-offs
  logic _unused;
  assign _unused = ^{s_axil_awprot, s_axil_arprot, s_axil_wstrb};

endmodule
