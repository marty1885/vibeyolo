// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// sppf — SPPF (Spatial Pyramid Pooling Fast) integration block.
//
// Pipeline (sits between /model.9/cv1 and /model.9/cv2 in YOLO26n):
//
//     in_pixel[C] ──┬──────────────────────────────────────────┐
//                   │                                           │ (cv1 stream)
//                   ▼                                           │
//             [linebuf K=5] -> [C× maxpool 5×5] = mp1_pixel ──┬─┤
//                   │                                          │ │
//                   ▼                                          │ │
//             [linebuf K=5] -> [C× maxpool 5×5] = mp2_pixel ─┬─┤ │
//                   │                                         │ │ │
//                   ▼                                         │ │ │
//             [linebuf K=5] -> [C× maxpool 5×5] = mp3_pixel ──┤ │ │
//                                                              ▼ ▼ ▼
//                                                  concat → out_pixel[4*C]
//
// Implementation strategy: simple sequential frame-store pipeline. We
// store the full H*W*C input frame in a local RAM, then run three
// streaming passes through (linebuf + maxpool) writing each
// intermediate frame to its own RAM. Once mp3 is complete, we drain
// all four frames (cv1=input, mp1, mp2, mp3) into the output by
// channel-concatenation per pixel.
//
// Quantization: pure int8 streaming, all four sub-streams share the
// same scale (the L31 SiLU output scale). Maxpool is scale-preserving.
//
// Padding caveat (FOOTGUN): linebuf_kxk uses ZERO out-of-frame padding,
// whereas ONNX MaxPool uses -INF (ignored) padding. For a 13×13 effective
// receptive field on a 20×20 frame the only fully-valid output ROI is
// positions (6..13, 6..13) = 8×8. Boundary outputs may differ from ORT
// because zero-padding can mask negative max values. The DV testbench
// compares only the inner valid ROI.

module sppf #(
  parameter int H = 20,
  parameter int W = 20,
  parameter int C = 128,
  parameter int K = 5
) (
  input  logic                              clk_i,
  input  logic                              rst_ni,

  // start_i: pulse high for one cycle to begin processing a new frame.
  input  logic                              start_i,

  // Input pixel stream (C×i8 per beat). Raster scan, H*W beats per frame.
  input  logic                              ivalid_i,
  output logic                              iready_o,
  input  logic signed [C-1:0][7:0]          idata_i,

  // Output stream: 4*C int8 channels per pixel, H*W beats per frame.
  // Channel layout: [cv1[0..C-1], mp1[0..C-1], mp2[0..C-1], mp3[0..C-1]].
  output logic                              ovalid_o,
  input  logic                              oready_i,
  output logic signed [4*C-1:0][7:0]        odata_o,

  // Asserts after the last output pixel of a frame has been emitted.
  output logic                              done_o
);

  localparam int Total = H * W;
  localparam int IdxW  = (Total <= 1) ? 1 : $clog2(Total);

  // ── Four frame storages ──────────────────────────────────────
  //   in_mem  : cv1 stream (input)
  //   mp1_mem : after first  maxpool
  //   mp2_mem : after second maxpool
  //   mp3_mem : after third  maxpool
  logic signed [C-1:0][7:0] in_mem  [Total];
  logic signed [C-1:0][7:0] mp1_mem [Total];
  logic signed [C-1:0][7:0] mp2_mem [Total];
  logic signed [C-1:0][7:0] mp3_mem [Total];

  // ── FSM ──────────────────────────────────────────────────────
  typedef enum logic [2:0] {
    S_IDLE,    // wait for start_i
    S_LOAD,    // accept H*W pixels into in_mem
    S_PASS1,   // stream in_mem  -> linebuf -> maxpool -> mp1_mem
    S_PASS2,   // stream mp1_mem -> linebuf -> maxpool -> mp2_mem
    S_PASS3,   // stream mp2_mem -> linebuf -> maxpool -> mp3_mem
    S_DRAIN,   // emit H*W concatenated output pixels
    S_DONE
  } state_e;
  state_e state_q, state_d;

  logic [IdxW:0] in_cnt_q,  in_cnt_d;   // load counter (0..Total)
  logic [IdxW:0] out_cnt_q, out_cnt_d;  // drain counter

  // ── Streaming pass control ───────────────────────────────────
  // For each pass, we drive a linebuf with the previous frame's data
  // and capture the maxpool output into the next frame buffer.
  logic [IdxW:0] pass_wr_q, pass_wr_d;   // read pointer into source mem (write-into-linebuf side)
  logic [IdxW:0] pass_rd_q, pass_rd_d;   // write pointer into dest mem (read-out-of-linebuf side)

  // Linebuf handshake
  logic                              lb_wvalid;
  logic                              lb_wready;
  logic signed [C-1:0][7:0]          lb_wdata;
  logic                              lb_rvalid;
  logic                              lb_rready;
  logic signed [K*K*C-1:0][7:0]      lb_rdata;
  logic                              lb_clr;

  // Choose source data for the current pass based on FSM state.
  always_comb begin
    lb_wdata = '0;
    case (state_q)
      S_PASS1: if (pass_wr_q < IdxW'(Total)) lb_wdata = in_mem [pass_wr_q[IdxW-1:0]];
      S_PASS2: if (pass_wr_q < IdxW'(Total)) lb_wdata = mp1_mem[pass_wr_q[IdxW-1:0]];
      S_PASS3: if (pass_wr_q < IdxW'(Total)) lb_wdata = mp2_mem[pass_wr_q[IdxW-1:0]];
      default: lb_wdata = '0;
    endcase
  end

  // Drive linebuf write side: source has data while pass_wr < Total.
  logic in_pass;
  assign in_pass = (state_q == S_PASS1) || (state_q == S_PASS2) || (state_q == S_PASS3);

  // One-cycle clr pulse on entry to every pass: the linebuf is shared
  // across PASS1/2/3 and otherwise carries stale row state and its
  // H*W frame-boundary stop from the previous pass — which would
  // wedge lb_wready low forever in PASS2/3. We register a "just
  // entered a pass" flag and OR it into lb_clr so each pass starts
  // with a fresh linebuf.
  logic pass_entry_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) pass_entry_q <= 1'b0;
    else pass_entry_q <= (state_d != state_q) &&
                         ((state_d == S_PASS1) ||
                          (state_d == S_PASS2) ||
                          (state_d == S_PASS3));
  end

  // Count reads (lb_r_fires) issued in the current pass. We must stop
  // issuing reads after Total of them — otherwise the linebuf keeps
  // emitting wrapped patches (out_row/out_col wrap modulo H/W with no
  // Total-gate), the maxpool keeps capturing them, and the resulting
  // in-flight mp result lands in the next pass's mem at position 0.
  logic [IdxW:0] lb_reads_q;

  assign lb_clr    = pass_entry_q ||
                     (state_q == S_IDLE) || (state_q == S_LOAD);
  assign lb_wvalid = in_pass && (pass_wr_q < IdxW'(Total)) && !lb_clr;
  assign lb_rready = in_pass && !lb_clr && (lb_reads_q < IdxW'(Total));

  // Linebuf instance shared across the three passes (resets via lb_clr
  // between passes).
  linebuf_kxk #(
    .K(K), .W(W), .H(H), .Channels(C)
  ) u_lb (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .clr_i   (lb_clr),
    .wvalid_i(lb_wvalid),
    .wready_o(lb_wready),
    .wdata_i (lb_wdata),
    .rvalid_o(lb_rvalid),
    .rready_i(lb_rready),
    .rdata_o (lb_rdata)
  );

  // ── Per-channel maxpool ──────────────────────────────────────
  // The linebuf emits K*K*C patch bytes. For channel c, the K*K
  // elements live at indices [(ky*K+kx)*C + c] for ky,kx in [0,K).
  // Re-pack into the maxpool's expected layout: a K*K-wide signed
  // int8 vector per channel.
  logic signed [C-1:0][K*K-1:0][7:0] mp_x;
  always_comb begin
    for (int c = 0; c < C; c++) begin
      for (int kk = 0; kk < K*K; kk++) begin
        mp_x[c][kk] = lb_rdata[kk*C + c];
      end
    end
  end


  // Maxpool is a registered output: one cycle of latency relative to
  // lb_rvalid&lb_rready. We track that with a 1-cycle "captured"
  // pipeline register that mirrors the maxpool's internal register.
  logic                       mp_en;
  logic                       mp_valid_q;
  logic [IdxW:0]              mp_pos_q;
  logic signed [C-1:0][7:0]   mp_y;

  // Mux maxpool destination: which mem to write into.
  logic mp_we;
  assign mp_we = mp_valid_q;

  // Per-channel maxpool array.
  for (genvar c = 0; c < C; c++) begin : gen_mp
    maxpool_kxk #(.K(K)) u_mp (
      .clk_i (clk_i),
      .rst_ni(rst_ni),
      .en_i  (mp_en),
      .x_i   (mp_x[c]),
      .y_o   (mp_y[c])
    );
  end

  // ── FSM next-state and counter logic ──────────────────────────
  logic load_fire;
  assign load_fire = (state_q == S_LOAD) && ivalid_i && iready_o;

  logic lb_w_fire;
  logic lb_r_fire;
  assign lb_w_fire = lb_wvalid & lb_wready;
  assign lb_r_fire = lb_rvalid & lb_rready;

  // mp_en pulses when the linebuf consumed a fresh patch this cycle —
  // that's the same cycle the maxpool sees new x_i. The maxpool
  // registers the output, so the *next* cycle mp_y is valid for
  // pass_rd position pass_rd_q.
  assign mp_en = lb_r_fire;

  // We must NOT issue more linebuf reads than the dest mem can store.
  // Stop reads once pass_rd has captured all Total outputs.
  // But the linebuf's `enough_inputs` predicate also gates rvalid, so
  // back-pressure naturally falls out. We also need to stop reading
  // when pass_rd would overflow.
  // To keep it simple, allow rready while pass_rd_q < Total.
  // (override the global assignment further up)

  always_comb begin
    state_d   = state_q;
    in_cnt_d  = in_cnt_q;
    out_cnt_d = out_cnt_q;
    pass_wr_d = pass_wr_q;
    pass_rd_d = pass_rd_q;

    case (state_q)
      S_IDLE: begin
        if (start_i) begin
          state_d   = S_LOAD;
          in_cnt_d  = '0;
          out_cnt_d = '0;
          pass_wr_d = '0;
          pass_rd_d = '0;
        end
      end
      S_LOAD: begin
        if (load_fire) begin
          in_cnt_d = in_cnt_q + 1'b1;
          if (in_cnt_q + 1'b1 == IdxW'(Total)) begin
            state_d   = S_PASS1;
            pass_wr_d = '0;
            pass_rd_d = '0;
          end
        end
      end
      S_PASS1, S_PASS2, S_PASS3: begin
        if (lb_w_fire) pass_wr_d = pass_wr_q + 1'b1;
        // pass_rd is incremented when mp output is valid (one cycle
        // after lb_r_fire). We mirror mp_valid_q below.
        if (mp_valid_q && (pass_rd_q < IdxW'(Total))) begin
          pass_rd_d = pass_rd_q + 1'b1;
          if (pass_rd_q + 1'b1 == IdxW'(Total)) begin
            // pass complete
            pass_wr_d = '0;
            pass_rd_d = '0;
            if      (state_q == S_PASS1) state_d = S_PASS2;
            else if (state_q == S_PASS2) state_d = S_PASS3;
            else /* PASS3 */             state_d = S_DRAIN;
          end
        end
      end
      S_DRAIN: begin
        if (ovalid_o && oready_i) begin
          out_cnt_d = out_cnt_q + 1'b1;
          if (out_cnt_q + 1'b1 == IdxW'(Total)) begin
            state_d = S_DONE;
          end
        end
      end
      S_DONE: begin
        // stay until next start_i
        if (start_i) begin
          state_d   = S_LOAD;
          in_cnt_d  = '0;
          out_cnt_d = '0;
          pass_wr_d = '0;
          pass_rd_d = '0;
        end
      end
      default: state_d = S_IDLE;
    endcase
  end

  // ── Sequential ───────────────────────────────────────────────
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q    <= S_IDLE;
      in_cnt_q   <= '0;
      out_cnt_q  <= '0;
      pass_wr_q  <= '0;
      pass_rd_q  <= '0;
      mp_valid_q <= 1'b0;
      mp_pos_q   <= '0;
      lb_reads_q <= '0;
    end else begin
      state_q    <= state_d;
      in_cnt_q   <= in_cnt_d;
      out_cnt_q  <= out_cnt_d;
      pass_wr_q  <= pass_wr_d;
      pass_rd_q  <= pass_rd_d;
      // Maxpool latency tracker: mp_y this cycle corresponds to
      // pass_rd_q at the time lb_r_fire was asserted last cycle.
      mp_valid_q <= mp_en;
      // mp_pos_q must hold the frame position of the patch captured
      // this cycle. pass_rd_q advances on mp_valid_q (= mp_en delayed
      // by 1), so for consecutive lb_r_fires the *current* pass_rd_q
      // hasn't absorbed the prior fire yet. Use its next-cycle value
      // (pass_rd_q + mp_valid_q) so back-to-back fires get distinct
      // mp_pos_q values 0,1,2,... — otherwise the first two writes
      // both target position 0 and every subsequent write is off by 1.
      if (mp_en) mp_pos_q <= pass_rd_q + (mp_valid_q ? IdxW'(1) : IdxW'(0));
      // lb_reads_q resets at every pass entry (when lb_clr is high).
      if (lb_clr)         lb_reads_q <= '0;
      else if (lb_r_fire) lb_reads_q <= lb_reads_q + 1'b1;
    end
  end

  // Capture loaded input into in_mem.
  always_ff @(posedge clk_i) begin
    if (load_fire) in_mem[in_cnt_q[IdxW-1:0]] <= idata_i;
  end

  // Capture maxpool output into the appropriate frame buffer.
  always_ff @(posedge clk_i) begin
    if (mp_we && (mp_pos_q < IdxW'(Total))) begin
      case (state_q)
        S_PASS1: mp1_mem[mp_pos_q[IdxW-1:0]] <= mp_y;
        S_PASS2: mp2_mem[mp_pos_q[IdxW-1:0]] <= mp_y;
        S_PASS3: mp3_mem[mp_pos_q[IdxW-1:0]] <= mp_y;
        default: ;
      endcase
    end
  end

  // ── External handshakes ──────────────────────────────────────
  assign iready_o = (state_q == S_LOAD);

  // Drain: present the concatenated pixel.
  logic [IdxW-1:0] drain_idx;
  assign drain_idx = out_cnt_q[IdxW-1:0];
  logic signed [C-1:0][7:0] cv1_pix, mp1_pix, mp2_pix, mp3_pix;
  assign cv1_pix = in_mem [drain_idx];
  assign mp1_pix = mp1_mem[drain_idx];
  assign mp2_pix = mp2_mem[drain_idx];
  assign mp3_pix = mp3_mem[drain_idx];

  // Channel layout: lower channels = cv1, then mp1, mp2, mp3.
  always_comb begin
    for (int c = 0; c < C; c++) begin
      odata_o[          c] = cv1_pix[c];
      odata_o[  C   +   c] = mp1_pix[c];
      odata_o[2*C   +   c] = mp2_pix[c];
      odata_o[3*C   +   c] = mp3_pix[c];
    end
  end

  assign ovalid_o = (state_q == S_DRAIN);
  assign done_o   = (state_q == S_DONE);

endmodule
