// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// detect_head — YOLO26n /model.23 post-processing, end-to-end, NO NMS.
//
// Ingests the six conv-tail int8 streams (box cv2.x.2 = 4ch, cls cv3.x.2 =
// 80ch) for all 8400 anchors in canonical order (scale0 80x80 = 6400, scale1
// 40x40 = 1600, scale2 20x20 = 400) and produces the model outputs:
//   pred_boxes[300][4]  (normalized cx,cy,w,h, fp16)
//   logits   [300][80]  (fp16)
// selected by TopK(reduce_max(cls), k=300) across all anchors.
//
// There is NO DFL and NO sigmoid in this export: the box branch regresses 4
// ltrb distances directly (box_affine), and ranking runs on raw logits
// (sigmoid is monotone, so argmax/topk are unchanged). anchor (col+.5,row+.5)
// and stride {8,16,32} are derived from a counter — no ROM.
//
// Three sequential phases (FSM):
//   P_DECODE  — for each anchor: box_affine -> boxes_mem; cls -> logits_mem;
//               reduce_max_n -> dequant(score) -> score_mem. ~8400 cyc.
//   P_TOPK    — feed score_mem[0..8399] into topk_fp16 at its in_ready rate
//               (random-access read decouples from the heap's multi-cycle
//               sift, so no backpressure FIFO is needed). ~<=76k cyc.
//   P_GATHER  — for each of 300 heap indices: read boxes_mem (direct) and
//               logits_mem -> dequant_n -> stream out. ~300 cyc.
//
// Cross-scale: the three scales carry DIFFERENT per-tensor cls S_OUT, so the
// score is dequantized to a common fp16 before TopK ranks across all anchors,
// and gathered logits are dequantized with their originating scale's S_OUT
// (selected from the index range). Box uses per-scale S_box likewise.

module detect_head #(
  parameter int N_ANCHOR = 8400,
  parameter int N_CLS    = 80,
  parameter int K        = 300,
  parameter int IDX_W    = 14   // clog2(8400)=14
) (
  input  logic                          clk_i,
  input  logic                          rst_ni,

  input  logic                          start_i,   // pulse to begin a frame

  // Per-scale fp16 dequant scales (sampled at start).
  input  logic        [15:0]            s_box_i [3],
  input  logic        [15:0]            s_cls_i [3],

  // Anchor input stream (canonical order). Present one anchor when in_ready_o.
  output logic                          in_ready_o,
  input  logic                          in_valid_i,
  input  logic signed [3:0][7:0]        box_i,        // l,t,r,b int8
  input  logic signed [N_CLS-1:0][7:0]  cls_i,        // 80 class logits int8

  // Detection output stream (300 beats).
  output logic                          out_valid_o,
  output logic        [IDX_W-1:0]       out_anchor_o, // source anchor index
  output logic        [3:0][15:0]       out_box_o,    // cx,cy,w,h fp16 (normalized)
  output logic        [N_CLS-1:0][15:0] out_logits_o, // 80 class logits fp16

  output logic                          done_o        // pulse when frame complete
);

  // ─────────────── scale geometry ───────────────
  localparam int S0_CNT = 6400, S1_CNT = 1600;
  localparam int S0_GW  = 80,   S1_GW  = 40,   S2_GW  = 20;
  localparam int KIDX_W = (K <= 1) ? 1 : $clog2(K);
  // stride 8/16/32

  // ─────────────── behavioral SRAMs (PD swaps for macros) ───────────────
  logic [3:0][15:0]        boxes_mem  [N_ANCHOR];   // 67 KB
  logic [N_CLS-1:0][7:0]   logits_mem [N_ANCHOR];   // 672 KB
  logic [15:0]             score_mem  [N_ANCHOR];   // 16.8 KB

  // ─────────────── frame-latched scales ───────────────
  logic [15:0] sbox_q [3], scls_q [3];

  // ─────────────── FSM ───────────────
  typedef enum logic [2:0] {S_IDLE, S_DECODE, S_DRAIN, S_TOPK, S_TWAIT, S_GATHER, S_DONE} st_e;
  st_e st_q, st_d;

  // anchor walk counters (P_DECODE)
  logic [IDX_W-1:0] aidx_q;          // anchor index being presented
  logic [IDX_W-1:0] col_q, row_q;
  logic [1:0]       scl_q;           // scale id 0/1/2
  logic [IDX_W-1:0] gw;              // grid width of current scale
  logic signed [31:0] stride_cur;
  always_comb begin
    unique case (scl_q)
      2'd0:    begin gw = IDX_W'(S0_GW); stride_cur = 32'sd8;  end
      2'd1:    begin gw = IDX_W'(S1_GW); stride_cur = 32'sd16; end
      default: begin gw = IDX_W'(S2_GW); stride_cur = 32'sd32; end
    endcase
  end

  // scale id from an arbitrary anchor index (for gather + in-flight pipes)
  function automatic logic [1:0] scale_of(input logic [IDX_W-1:0] idx);
    if (idx < IDX_W'(S0_CNT))                 return 2'd0;
    else if (idx < IDX_W'(S0_CNT + S1_CNT))   return 2'd1;
    else                                       return 2'd2;
  endfunction

  // accept a new anchor this cycle?
  logic dec_accept;
  assign dec_accept = (st_q == S_DECODE) && in_valid_i;
  assign in_ready_o = (st_q == S_DECODE);

  // ─────────────── box decode datapath ───────────────
  logic        ba_valid;
  logic [15:0] ba_cx, ba_cy, ba_w, ba_h;
  box_affine u_box (
    .clk_i, .rst_ni,
    .valid_i (dec_accept),
    .l_i     (box_i[0]), .t_i (box_i[1]), .r_i (box_i[2]), .b_i (box_i[3]),
    .s_box_i (sbox_q[scl_q]),
    .col_i   (32'(signed'({1'b0, col_q}))),
    .row_i   (32'(signed'({1'b0, row_q}))),
    .stride_i(stride_cur),
    .valid_o (ba_valid),
    .cx_o(ba_cx), .cy_o(ba_cy), .w_o(ba_w), .h_o(ba_h)
  );
  // anchor index carried through box_affine's 18-deep pipe for the write addr
  // (BA_LAT == box_affine valid_i→valid_o latency, exact, offset 0).
  localparam int BA_LAT = 18;
  logic [IDX_W-1:0] ba_idx_pipe [BA_LAT];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) for (int k=0;k<BA_LAT;k++) ba_idx_pipe[k] <= '0;
    else begin
      ba_idx_pipe[0] <= aidx_q;
      for (int k=1;k<BA_LAT;k++) ba_idx_pipe[k] <= ba_idx_pipe[k-1];
    end
  end

  // ─────────────── score datapath: reduce_max -> dequant ───────────────
  logic              rm_valid;
  logic signed [7:0] rm_max;
  reduce_max_n #(.N(N_CLS)) u_rmax (
    .clk_i, .rst_ni, .en_i(dec_accept), .x_i(cls_i),
    .valid_o(rm_valid), .y_o(rm_max)
  );
  // scale id aligned to reduce_max output (1-cycle): pick that anchor's S_cls
  logic [1:0] scl_d1;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) scl_d1 <= 2'd0; else scl_d1 <= scl_q;
  end
  logic              sc_valid;
  logic [0:0][15:0]  sc_y;
  logic signed [0:0][7:0] rm_max_vec;
  assign rm_max_vec[0] = rm_max;
  dequant_n #(.N(1)) u_score_dq (
    .clk_i, .rst_ni, .en_i(rm_valid), .x_i(rm_max_vec), .scale_i(scls_q[scl_d1]),
    .valid_o(sc_valid), .y_o(sc_y)
  );
  // anchor index carried through reduce_max(1) + dequant(5) = 6 deep
  // (SC_LAT == reduce_max_n latency + dequant_n latency; aligns sc_idx_pipe
  // with sc_y/sc_valid for the score_mem write address).
  localparam int SC_LAT = 6;
  logic [IDX_W-1:0] sc_idx_pipe [SC_LAT];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) for (int k=0;k<SC_LAT;k++) sc_idx_pipe[k] <= '0;
    else begin
      sc_idx_pipe[0] <= aidx_q;
      for (int k=1;k<SC_LAT;k++) sc_idx_pipe[k] <= sc_idx_pipe[k-1];
    end
  end

  // count of decoded scores written (frame completes P_DECODE drain when all
  // 8400 boxes AND scores are stored)
  logic [IDX_W:0] box_wr_cnt_q, score_wr_cnt_q;

  // ─────────────── TopK ───────────────
  logic              tk_in_ready;
  logic              tk_in_valid;
  logic [15:0]       tk_in_value;
  logic [IDX_W-1:0]  tk_in_index;
  logic              tk_done;
  // topk values aren't needed downstream (we gather by index only).
  // verilator lint_off UNUSEDSIGNAL
  logic [15:0]       tk_out_value [K];
  // verilator lint_on UNUSEDSIGNAL
  logic [IDX_W-1:0]  tk_out_index [K];
  logic              tk_start;
  topk_fp16 #(.N(N_ANCHOR), .K(K), .IDX_W(IDX_W)) u_topk (
    .clk_i, .rst_ni,
    .start_i   (tk_start),
    .in_valid_i(tk_in_valid),
    .in_value_i(tk_in_value),
    .in_index_i(tk_in_index),
    .in_ready_o(tk_in_ready),
    .done_o    (tk_done),
    .out_value_o(tk_out_value),
    .out_index_o(tk_out_index)
  );
  // TopK feed pointer (P_TOPK): present score_mem[tptr] while in_ready.
  logic [IDX_W:0] tptr_q;
  assign tk_in_valid = (st_q == S_TOPK) && (tptr_q < (IDX_W+1)'(N_ANCHOR));
  assign tk_in_value = score_mem[tptr_q[IDX_W-1:0]];
  assign tk_in_index = tptr_q[IDX_W-1:0];

  // ─────────────── gather datapath ───────────────
  logic [IDX_W:0]   gptr_q;            // 0..K
  logic [IDX_W-1:0] g_anchor;
  assign g_anchor = tk_out_index[gptr_q[KIDX_W-1:0]];
  // logit dequant of the gathered anchor
  logic               gdq_valid;
  logic [N_CLS-1:0][15:0] gdq_y;
  // g_rd_en is COMBINATIONAL so the read enable, the read address (g_anchor),
  // and the gptr advance all use the SAME gptr_q this cycle (footgun: a
  // registered enable would lag the combinational address by one).
  logic               g_rd_en;
  assign g_rd_en = (st_q == S_GATHER) && (gptr_q < (IDX_W+1)'(K));
  logic [1:0]         g_scl;
  assign g_scl = scale_of(g_anchor);
  dequant_n #(.N(N_CLS)) u_gather_dq (
    .clk_i, .rst_ni,
    .en_i   (g_rd_en),
    .x_i    (logits_mem[g_anchor]),
    .scale_i(scls_q[g_scl]),
    .valid_o(gdq_valid),
    .y_o    (gdq_y)
  );
  // carry box + anchor through dequant's 5-cycle latency for aligned output
  // (G_LAT == dequant_n en_i→valid_o latency, exact, offset 0).
  localparam int G_LAT = 5;
  logic [3:0][15:0]  g_box_pipe   [G_LAT];
  logic [IDX_W-1:0]  g_anch_pipe  [G_LAT];
  logic              g_vld_pipe   [G_LAT];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int k=0;k<G_LAT;k++) begin
        g_box_pipe[k]<='0; g_anch_pipe[k]<='0; g_vld_pipe[k]<=1'b0;
      end
    end else begin
      g_box_pipe[0]  <= boxes_mem[g_anchor];
      g_anch_pipe[0] <= g_anchor;
      g_vld_pipe[0]  <= g_rd_en;
      for (int k=1;k<G_LAT;k++) begin
        g_box_pipe[k]  <= g_box_pipe[k-1];
        g_anch_pipe[k] <= g_anch_pipe[k-1];
        g_vld_pipe[k]  <= g_vld_pipe[k-1];
      end
    end
  end
  assign out_valid_o  = gdq_valid;
  assign out_box_o    = g_box_pipe[G_LAT-1];
  assign out_anchor_o = g_anch_pipe[G_LAT-1];
  assign out_logits_o = gdq_y;

  // ─────────────── sequential control + writes ───────────────
  logic [IDX_W:0] gout_cnt_q;   // count of detections emitted

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st_q <= S_IDLE;
      aidx_q <= '0; col_q <= '0; row_q <= '0; scl_q <= 2'd0;
      box_wr_cnt_q <= '0; score_wr_cnt_q <= '0;
      tptr_q <= '0; gptr_q <= '0; gout_cnt_q <= '0;
      tk_start <= 1'b0; done_o <= 1'b0;
      for (int i=0;i<3;i++) begin sbox_q[i] <= 16'h0; scls_q[i] <= 16'h0; end
    end else begin
      tk_start <= 1'b0;
      done_o   <= 1'b0;
      st_q     <= st_d;

      // box / score writes (independent of phase; gated by leaf valid)
      if (ba_valid) begin
        // element order: [0]=cx [1]=cy [2]=w [3]=h (concat is MSB-first, so
        // list h,w,cy,cx to land cx in element 0).
        boxes_mem[ba_idx_pipe[BA_LAT-1]] <= {ba_h, ba_w, ba_cy, ba_cx};
        box_wr_cnt_q <= box_wr_cnt_q + 1'b1;
      end
      if (sc_valid) begin
        score_mem[sc_idx_pipe[SC_LAT-1]] <= sc_y[0];
        score_wr_cnt_q <= score_wr_cnt_q + 1'b1;
      end

      case (st_q)
        S_IDLE: if (start_i) begin
          for (int i=0;i<3;i++) begin sbox_q[i] <= s_box_i[i]; scls_q[i] <= s_cls_i[i]; end
          aidx_q<='0; col_q<='0; row_q<='0; scl_q<=2'd0;
          box_wr_cnt_q<='0; score_wr_cnt_q<='0; tptr_q<='0; gptr_q<='0; gout_cnt_q<='0;
        end

        S_DECODE: if (dec_accept) begin
          // logits stored immediately at the presented anchor
          logits_mem[aidx_q] <= cls_i;
          // advance anchor walk
          aidx_q <= aidx_q + 1'b1;
          if (col_q == gw - 1) begin
            col_q <= '0;
            row_q <= row_q + 1'b1;
          end else begin
            col_q <= col_q + 1'b1;
          end
          // scale boundary transitions
          if (aidx_q == IDX_W'(S0_CNT - 1) || aidx_q == IDX_W'(S0_CNT + S1_CNT - 1)) begin
            col_q <= '0; row_q <= '0; scl_q <= scl_q + 1'b1;
          end
        end

        S_DRAIN: begin
          // wait for box/score write counters to reach N_ANCHOR; pulse topk
          // start the cycle we hand off to S_TOPK.
          if (st_d == S_TOPK) tk_start <= 1'b1;
        end

        S_TOPK: begin
          if (tk_in_valid && tk_in_ready) tptr_q <= tptr_q + 1'b1;
        end

        S_TWAIT: ; // wait tk_done

        S_GATHER: begin
          if (g_rd_en) gptr_q <= gptr_q + 1'b1;
          if (gdq_valid) gout_cnt_q <= gout_cnt_q + 1'b1;
        end

        S_DONE: done_o <= 1'b1;
        default: ;
      endcase
    end
  end

  // next-state
  always_comb begin
    st_d = st_q;
    unique case (st_q)
      S_IDLE:   if (start_i) st_d = S_DECODE;
      S_DECODE: if (aidx_q == IDX_W'(N_ANCHOR-1) && dec_accept) st_d = S_DRAIN;
      S_DRAIN:  if (box_wr_cnt_q == (IDX_W+1)'(N_ANCHOR) &&
                    score_wr_cnt_q == (IDX_W+1)'(N_ANCHOR)) st_d = S_TOPK;
      S_TOPK:   if (tptr_q == (IDX_W+1)'(N_ANCHOR)) st_d = S_TWAIT;
      S_TWAIT:  if (tk_done) st_d = S_GATHER;
      S_GATHER: if (gout_cnt_q == (IDX_W+1)'(K)) st_d = S_DONE;
      S_DONE:   st_d = S_IDLE;
      default:  st_d = S_IDLE;
    endcase
  end

endmodule
