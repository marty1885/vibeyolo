// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// upsample_concat — nearest-neighbour 2x upsample + channel-concat
// integration block. Sits at the YOLO26n P4->P3 neck join:
//
//                /model.10/cv2 (256 ch, 20x20)
//                          │
//                          ▼
//             /model.11/Resize  (NN, scale=2)
//                          │
//                          ▼ (256 ch, 40x40)
//                          │
//                          │   /model.6/cv2 (128 ch, 40x40)
//                          │              │
//                          ▼              ▼
//                  /model.12/Concat (axis=1)
//                          │
//                          ▼ (384 ch, 40x40)
//
// Implementation strategy: frame-store + drain (mirrors integ/sppf_model9).
//
//   1. S_LOAD_A : accept H_A*W_A beats of C_A-wide int8 raster-scan A.
//   2. S_LOAD_B : accept H_B*W_B beats of C_B-wide int8 raster-scan B.
//   3. S_DRAIN  : emit H_O*W_O beats of C_O-wide int8 raster-scan, where
//                 lower C_A channels  = A[ho/2, wo/2]   (NN 2x)
//                 upper C_B channels  = B[ho,   wo  ]
//
// Quantization: both inputs and the output share the SAME int8 scale
// S_OUT. The upstream producers are responsible for putting both
// streams on that common grid (in HW that would be a per-stream fp16
// rescale before this block; in extract.py we just pick S_OUT to cover
// the joint range and quantize both inputs from float against it).
// Nearest-neighbour and concat are both scale-preserving, so the DUT
// performs no arithmetic at all — only addressing.
//
// Note: ONNX Concat order is [Resize_out, P3_skip]. We emit the
// upsampled A channels first (channel indices [0, C_A)), then B
// (channel indices [C_A, C_A+C_B)).

module upsample_concat #(
  parameter int H_A = 20,
  parameter int W_A = 20,
  parameter int C_A = 256,
  parameter int H_B = 40,
  parameter int W_B = 40,
  parameter int C_B = 128
) (
  input  logic                              clk_i,
  input  logic                              rst_ni,

  // start_i: pulse high for one cycle to begin a new frame.
  input  logic                              start_i,

  // Input stream A — upsample source (C_A int8 lanes per beat, raster).
  input  logic                              avalid_i,
  output logic                              aready_o,
  input  logic signed [C_A-1:0][7:0]        adata_i,

  // Input stream B — skip path (C_B int8 lanes per beat, raster).
  input  logic                              bvalid_i,
  output logic                              bready_o,
  input  logic signed [C_B-1:0][7:0]        bdata_i,

  // Output stream — concat(upsample2(A), B), C_O wide int8 per pixel.
  output logic                              ovalid_o,
  input  logic                              oready_i,
  output logic signed [C_A+C_B-1:0][7:0]    odata_o,

  // Asserts after the final output pixel of a frame has been emitted.
  output logic                              done_o
);

  localparam int Total_A = H_A * W_A;
  localparam int Total_O = H_B * W_B;       // == H_O * W_O

  // Counter widths sized so that ==Total_A/_O comparisons fit even when
  // the counter has just incremented past the last valid value (so we
  // need one extra bit beyond clog2 of the value).
  localparam int AIdxW = (Total_A <= 1) ? 1 : $clog2(Total_A);
  localparam int OIdxW = (Total_O <= 1) ? 1 : $clog2(Total_O);

  // ── Frame stores ─────────────────────────────────────────────
  logic signed [C_A-1:0][7:0] a_mem [Total_A];
  logic signed [C_B-1:0][7:0] b_mem [Total_O];

  // ── FSM ──────────────────────────────────────────────────────
  typedef enum logic [2:0] {
    S_IDLE,
    S_LOAD_A,
    S_LOAD_B,
    S_DRAIN,
    S_DONE
  } state_e;
  state_e state_q, state_d;

  logic [AIdxW:0] a_cnt_q, a_cnt_d;
  logic [OIdxW:0] b_cnt_q, b_cnt_d;
  logic [OIdxW:0] o_cnt_q, o_cnt_d;

  // ── External handshakes ──────────────────────────────────────
  assign aready_o = (state_q == S_LOAD_A);
  assign bready_o = (state_q == S_LOAD_B);
  assign ovalid_o = (state_q == S_DRAIN);
  assign done_o   = (state_q == S_DONE);

  logic a_fire, b_fire, o_fire;
  assign a_fire = avalid_i && aready_o;
  assign b_fire = bvalid_i && bready_o;
  assign o_fire = ovalid_o && oready_i;

  // ── FSM next-state ───────────────────────────────────────────
  always_comb begin
    state_d = state_q;
    a_cnt_d = a_cnt_q;
    b_cnt_d = b_cnt_q;
    o_cnt_d = o_cnt_q;

    unique case (state_q)
      S_IDLE: begin
        if (start_i) begin
          state_d = S_LOAD_A;
          a_cnt_d = '0;
          b_cnt_d = '0;
          o_cnt_d = '0;
        end
      end
      S_LOAD_A: begin
        if (a_fire) begin
          a_cnt_d = a_cnt_q + 1'b1;
          if (a_cnt_q + 1'b1 == AIdxW'(Total_A)) begin
            state_d = S_LOAD_B;
          end
        end
      end
      S_LOAD_B: begin
        if (b_fire) begin
          b_cnt_d = b_cnt_q + 1'b1;
          if (b_cnt_q + 1'b1 == OIdxW'(Total_O)) begin
            state_d = S_DRAIN;
          end
        end
      end
      S_DRAIN: begin
        if (o_fire) begin
          o_cnt_d = o_cnt_q + 1'b1;
          if (o_cnt_q + 1'b1 == OIdxW'(Total_O)) begin
            state_d = S_DONE;
          end
        end
      end
      S_DONE: begin
        if (start_i) begin
          state_d = S_LOAD_A;
          a_cnt_d = '0;
          b_cnt_d = '0;
          o_cnt_d = '0;
        end
      end
      default: state_d = S_IDLE;
    endcase
  end

  // ── Sequential ───────────────────────────────────────────────
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= S_IDLE;
      a_cnt_q <= '0;
      b_cnt_q <= '0;
      o_cnt_q <= '0;
    end else begin
      state_q <= state_d;
      a_cnt_q <= a_cnt_d;
      b_cnt_q <= b_cnt_d;
      o_cnt_q <= o_cnt_d;
    end
  end

  // ── Frame-store writes ───────────────────────────────────────
  always_ff @(posedge clk_i) begin
    if (a_fire) a_mem[a_cnt_q[AIdxW-1:0]] <= adata_i;
    if (b_fire) b_mem[b_cnt_q[OIdxW-1:0]] <= bdata_i;
  end

  // ── Drain addressing ─────────────────────────────────────────
  // Output index decomposes into (ho, wo); the A-side read index is
  // (ho/2, wo/2) in a 20x20 array, i.e. linear index
  //     (ho >> 1) * W_A + (wo >> 1).
  // The B-side read index is just o_cnt_q (raster).
  localparam int WoW   = (W_B <= 1) ? 1 : $clog2(W_B);
  localparam int AdrAW = (Total_A <= 1) ? 1 : $clog2(Total_A);

  logic [WoW-1:0] ho, wo;
  logic [AdrAW-1:0] a_rd_idx;
  logic [OIdxW-1:0] b_rd_idx;
  logic signed [C_A-1:0][7:0] a_pix;
  logic signed [C_B-1:0][7:0] b_pix;

  always_comb begin
    // Decompose o_cnt_q into (ho, wo) within the 40x40 output frame.
    ho      = WoW'(o_cnt_q[OIdxW-1:0] / OIdxW'(W_B));
    wo      = WoW'(o_cnt_q[OIdxW-1:0] % OIdxW'(W_B));
    // NN downsample of the output coordinate to the A-frame coordinate.
    a_rd_idx = AdrAW'((ho >> 1) * WoW'(W_A) + (wo >> 1));
    b_rd_idx = o_cnt_q[OIdxW-1:0];

    a_pix = a_mem[a_rd_idx];
    b_pix = b_mem[b_rd_idx];

    // Channel layout: A's C_A channels first, then B's C_B channels.
    for (int c = 0; c < C_A; c++) odata_o[          c] = a_pix[c];
    for (int c = 0; c < C_B; c++) odata_o[C_A   +   c] = b_pix[c];
  end

  // ── Lint keep-alive for unused locals ────────────────────────
  logic _unused;
  assign _unused = ^{a_cnt_q[AIdxW], b_cnt_q[OIdxW], o_cnt_q[OIdxW]};

endmodule
