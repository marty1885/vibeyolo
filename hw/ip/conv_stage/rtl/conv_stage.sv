// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// conv_stage — synthesizable streaming wrapper around the conv_layer compute
// core. This is the per-layer datapath the chip actually instantiates: it
// turns a raster int8 activation stream into the next layer's raster int8
// stream, hiding the per-tile dataflow that the DV C++ drivers used to do in
// software.
//
//   ivalid/idata[CIN]  ─▶ linebuf_kxk(K,W_IN,H_IN,CIN)  (zero same-padding)
//        │                     │ K*K*CIN patch per input position
//        │      stride decimate ▼ (keep positions where row%S==0 && col%S==0)
//        │                  patch_q ──┐
//   on-die ROMs ($readmemh):          │   tile-sequencer FSM
//     WROM  [COUT][K*K*CIN] i8        ├──▶ conv_layer (P_COUT × dotN, requant,
//     SROM  [COUT] fp16 scale         │      optional SiLU)
//     BROM  [COUT] fp16 bias          │   per out-pixel: outer ct, inner cit,
//                                     │   first/last_cin, cout_tile_idx
//   collect P_COUT lanes / valid_o beat, keyed by cout_tile_idx_o, into a
//   COUT-wide pixel; emit ovalid/odata[COUT] when the last cout-tile lands.
//   done_o pulses after the final output pixel of the frame.
//
// Weight delivery: weights are layer-constant, so the (ct,cit)-indexed tile is
// a combinational read of the constant WROM array — the dedicated weight
// memory / broadcast network of the real chip. Heavy for big layers, but that
// silicon is inherent to a fixed-function accelerator with no shared MAC array.
//
// v1 scope: group=1 (depthwise arrives pre-expanded to dense weights, handled
// transparently), RESIDUAL=0, P_PIX=1. Residual feed is a defined extension
// (conv_layer already supports it; the per-pixel residual stream buffer is the
// only missing plumbing).

// verilator lint_off UNUSEDPARAM
module conv_stage #(
  parameter int  CIN        = 128,
  parameter int  COUT       = 128,
  parameter int  K          = 3,
  parameter int  STRIDE     = 1,
  parameter int  PAD        = 1,
  parameter int  H_IN       = 40,
  parameter int  W_IN       = 40,
  parameter int  P_COUT     = 16,
  parameter int  P_CIN      = 8,
  parameter int  SILU       = 1,
  parameter real S_OUT_PRE  = 4.0 / 127.0,
  parameter real S_OUT_SILU = 4.0 / 127.0,
  parameter string WINIT    = "",   // COUT*K*K*CIN bytes, lane=(kh*K+kw)*CIN+kc
  parameter string SINIT    = "",   // COUT fp16 scale
  parameter string BINIT    = ""    // COUT fp16 bias
) (
  input  logic                                 clk_i,
  input  logic                                 rst_ni,

  // Frame control
  input  logic                                 start_i,   // pulse: begin a frame
  output logic                                 done_o,     // pulse: frame complete

  // Input activation stream: one input pixel (all CIN channels) per beat.
  input  logic                                 ivalid_i,
  output logic                                 iready_o,
  input  logic signed [CIN-1:0][7:0]           idata_i,

  // Output activation stream: one output pixel (all COUT channels) per beat.
  output logic                                 ovalid_o,
  input  logic                                 oready_i,
  output logic signed [COUT-1:0][7:0]          odata_o
);

  // ─── Derived geometry ──────────────────────────────────────────────
  localparam int N_LANE_FULL = K * K * CIN;            // full patch lanes
  localparam int N_CIN_TILE   = (CIN + P_CIN - 1) / P_CIN;
  localparam int N_COUT_TILE  = (COUT + P_COUT - 1) / P_COUT;
  localparam int H_OUT         = (H_IN + 2*PAD - K) / STRIDE + 1;
  localparam int W_OUT         = (W_IN + 2*PAD - K) / STRIDE + 1;
  localparam int N_OUT         = H_OUT * W_OUT;
  localparam int CT_W = $clog2((N_COUT_TILE < 2) ? 2 : N_COUT_TILE);
  localparam int CIT_W = $clog2((N_CIN_TILE < 2) ? 2 : N_CIN_TILE);

  // ─── Constant weight / scale / bias ROMs ───────────────────────────
  logic signed [7:0] wrom [COUT*N_LANE_FULL];
  logic       [15:0] srom [COUT];
  logic       [15:0] brom [COUT];
  initial begin
    if (WINIT != "") $readmemh(WINIT, wrom);
    if (SINIT != "") $readmemh(SINIT, srom);
    if (BINIT != "") $readmemh(BINIT, brom);
  end

  // ─── linebuf: raster pixels in, K*K*CIN patches out ────────────────
  logic                          lb_clr;
  logic                          lb_rvalid;
  logic                          lb_rready;
  logic signed [N_LANE_FULL-1:0][7:0] lb_patch;

  logic lb_wready;
  // Hold input off until the frame is running and the clear pulse has passed,
  // so no pixel is swallowed by the linebuf reset.
  assign iready_o = lb_wready & running_q & ~lb_clr;
  wire   lb_wvalid = ivalid_i & running_q & ~lb_clr;

  linebuf_kxk #(
    .K(K), .W(W_IN), .H(H_IN), .Channels(CIN)
  ) u_linebuf (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .clr_i   (lb_clr),
    .wvalid_i(lb_wvalid),
    .wready_o(lb_wready),
    .wdata_i (idata_i),
    .rvalid_o(lb_rvalid),
    .rready_i(lb_rready),
    .rdata_o (lb_patch)
  );

  // ─── Control state ──────────────────────────────────────────────────
  logic                running_q;       // inside a frame
  logic                sched_busy_q;    // patch in flight (latch → pixel_complete)
  logic                issuing_q;       // currently emitting beats to conv_layer
  logic [CT_W-1:0]     ct_q;            // outer cout-tile counter
  logic [CIT_W-1:0]    cit_q;           // inner cin-tile counter
  logic [$clog2(W_IN+1)-1:0] in_col_q;  // next-patch input column
  logic [$clog2(H_IN+1)-1:0] in_row_q;  // next-patch input row
  logic [$clog2(N_OUT+1)-1:0] out_emit_q; // output pixels emitted this frame
  logic signed [N_LANE_FULL-1:0][7:0]  patch_q;

  // Output: single holding register. A new patch is accepted only when the
  // hold is empty, so at most one pixel is in flight (in the conv_layer
  // pipeline) plus one in the hold — the hold can never be overwritten before
  // it is drained. Costs a little intra-stage overlap (linebuf can't prefetch
  // during a schedule), which only affects sim cycle count, not correctness;
  // timing sign-off uses the analytic per-stage cycles.
  logic signed [COUT-1:0][7:0] out_pix_q;     // pixel under assembly
  logic signed [COUT-1:0][7:0] out_hold_q;    // completed pixel awaiting drain
  logic                        out_hold_vld_q;

  wire kept_pos = (in_col_q % STRIDE == 0) && (in_row_q % STRIDE == 0);

  // Accept a new patch only when running, not mid-schedule, and the output
  // hold is free for the pixel this patch will produce.
  assign lb_rready = running_q & ~sched_busy_q & ~out_hold_vld_q;

  wire patch_take = lb_rvalid & lb_rready;
  wire last_col   = (in_col_q == W_IN-1);
  wire last_row   = (in_row_q == H_IN-1);

  // ─── Tile gather (combinational) ────────────────────────────────────
  logic signed [K*K*P_CIN-1:0][7:0]              x_tile;
  logic signed [P_COUT-1:0][K*K*P_CIN-1:0][7:0]  w_tile;
  logic        [P_COUT-1:0][15:0]                scale_tile;
  logic        [P_COUT-1:0][15:0]                bias_tile;

  always_comb begin
    automatic int spat, sfull, oc;
    x_tile     = '0;
    w_tile     = '0;
    scale_tile = '0;
    bias_tile  = '0;
    for (int kk = 0; kk < K*K; kk++) begin
      for (int p = 0; p < P_CIN; p++) begin
        // cin-tile slice of the patch: channel cit*P_CIN+p within window cell kk
        spat = kk*CIN + cit_q*P_CIN + p;
        if ((cit_q*P_CIN + p) < CIN)
          x_tile[kk*P_CIN + p] = patch_q[spat];
      end
    end
    for (int co = 0; co < P_COUT; co++) begin
      oc = ct_q*P_COUT + co;
      if (oc < COUT) begin
        scale_tile[co] = srom[oc];
        bias_tile [co] = brom[oc];
        for (int kk = 0; kk < K*K; kk++) begin
          for (int p = 0; p < P_CIN; p++) begin
            sfull = oc*N_LANE_FULL + kk*CIN + cit_q*P_CIN + p;
            if ((cit_q*P_CIN + p) < CIN)
              w_tile[co][kk*P_CIN + p] = wrom[sfull];
          end
        end
      end
    end
  end

  // ─── conv_layer instance ────────────────────────────────────────────
  wire                          cl_valid_i = issuing_q;
  wire                          cl_first   = (cit_q == 0);
  wire                          cl_last    = (cit_q == N_CIN_TILE-1);
  logic                         cl_ready_o;  // conv_layer is non-stalling (==1)
  logic                         cl_valid_o;
  logic [CT_W-1:0]              cl_ct_o;
  logic signed [P_COUT-1:0][7:0] cl_y_o;

  conv_layer #(
    .CIN(CIN), .COUT(COUT), .K(K), .STRIDE(STRIDE), .PAD(PAD),
    .H_OUT(H_OUT), .W_OUT(W_OUT), .P_COUT(P_COUT), .P_CIN(P_CIN),
    .RESIDUAL(0), .SILU(SILU),
    .S_OUT_PRE(S_OUT_PRE), .S_OUT_SILU(S_OUT_SILU)
  ) u_conv (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .valid_i        (cl_valid_i),
    .ready_o        (cl_ready_o),
    .first_cin_i    (cl_first),
    .last_cin_i     (cl_last),
    .cout_tile_idx_i(ct_q),
    .x_i            (x_tile),
    .w_i            (w_tile),
    .scale_i        (scale_tile),
    .bias_i         (bias_tile),
    .r_i            ('0),
    .r_scale_i      ('0),
    .r_bias_i       ('0),
    .valid_o        (cl_valid_o),
    .ready_i        (1'b1),
    .cout_tile_idx_o(cl_ct_o),
    .y_o            (cl_y_o)
  );

  // ─── Sequencer + output assembly ────────────────────────────────────
  wire pixel_complete = cl_valid_o & (cl_ct_o == CT_W'(N_COUT_TILE-1));

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      running_q    <= 1'b0;
      sched_busy_q <= 1'b0;
      issuing_q    <= 1'b0;
      ct_q         <= '0;
      cit_q        <= '0;
      in_col_q     <= '0;
      in_row_q     <= '0;
      out_emit_q   <= '0;
      patch_q      <= '0;
      out_pix_q      <= '0;
      out_hold_q     <= '0;
      out_hold_vld_q <= 1'b0;
      lb_clr         <= 1'b0;
      done_o         <= 1'b0;
    end else begin
      done_o <= 1'b0;
      lb_clr <= 1'b0;

      // Frame start
      if (start_i && !running_q) begin
        running_q  <= 1'b1;
        sched_busy_q <= 1'b0;
        issuing_q  <= 1'b0;
        ct_q       <= '0;
        cit_q      <= '0;
        in_col_q   <= '0;
        in_row_q   <= '0;
        out_emit_q <= '0;
        lb_clr     <= 1'b1;
      end

      // Consume a patch from the linebuf
      if (patch_take) begin
        // advance raster position of the next patch
        if (last_col) begin
          in_col_q <= '0;
          in_row_q <= last_row ? '0 : (in_row_q + 1'b1);
        end else begin
          in_col_q <= in_col_q + 1'b1;
        end
        // launch the tile schedule only at kept (strided) positions
        if (kept_pos) begin
          patch_q      <= lb_patch;
          sched_busy_q <= 1'b1;       // in flight until pixel_complete
          issuing_q    <= 1'b1;       // start emitting beats
          ct_q         <= '0;
          cit_q        <= '0;
        end
      end

      // Tile-schedule beat advance (only while issuing)
      if (issuing_q) begin
        if (cit_q == CIT_W'(N_CIN_TILE-1)) begin
          cit_q <= '0;
          if (ct_q == CT_W'(N_COUT_TILE-1)) begin
            issuing_q <= 1'b0;        // last beat issued; await completion
            ct_q      <= '0;
          end else begin
            ct_q <= ct_q + 1'b1;
          end
        end else begin
          cit_q <= cit_q + 1'b1;
        end
      end

      // Collect conv_layer outputs into the pixel under assembly. On the last
      // cout-tile the pixel is complete: merge this beat and latch the hold.
      if (cl_valid_o) begin
        automatic logic signed [COUT-1:0][7:0] nextpix;
        nextpix = out_pix_q;
        for (int co = 0; co < P_COUT; co++)
          if ((cl_ct_o*P_COUT + co) < COUT)
            nextpix[cl_ct_o*P_COUT + co] = cl_y_o[co];
        out_pix_q <= nextpix;

        if (pixel_complete) begin
          out_hold_q     <= nextpix;
          out_hold_vld_q <= 1'b1;
          sched_busy_q   <= 1'b0;     // pixel done; free to accept next patch
          out_emit_q     <= out_emit_q + 1'b1;
          if (out_emit_q == ($clog2(N_OUT+1))'(N_OUT-1)) begin
            running_q <= 1'b0;
            done_o    <= 1'b1;
          end
        end
      end

      // Drain the hold on a downstream beat (unless a new pixel lands the same
      // cycle, in which case the load above keeps it valid).
      if (ovalid_o & oready_i & ~pixel_complete)
        out_hold_vld_q <= 1'b0;
    end
  end

  assign ovalid_o = out_hold_vld_q;
  assign odata_o  = out_hold_q;

  // verilator lint_off UNUSEDSIGNAL
  wire _unused = ^{cl_ready_o};
  // verilator lint_on UNUSEDSIGNAL
endmodule
// verilator lint_on UNUSEDPARAM
