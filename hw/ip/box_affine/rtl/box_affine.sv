// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// box_affine — per-anchor affine box decode for the YOLO26n detect head.
//
// This export's box branch (cv2.x.2) regresses the four ltrb distances
// DIRECTLY as int8 (no DFL / 16-bin distribution — so box_decode does not
// apply here). Per anchor:
//
//   d = dequant(ltrb_i8) = fp16(ltrb_i8) * S_box        (grid units)
//   anchor = (col + 0.5, row + 0.5)                      (grid units)
//   x1 = (col+0.5 - d_l) * stride = cx_center - d_l*stride
//   x2 = (col+0.5 + d_r) * stride = cx_center + d_r*stride
//   y1 = cy_center - d_t*stride ;  y2 = cy_center + d_b*stride
//   cx = (x1+x2)/2  cy = (y1+y2)/2  w = x2-x1  h = y2-y1   (pixel units)
//   {cx,cy,w,h} /= 640                                    (normalized)
//
// where cx_center = (col+0.5)*stride = col_fp16*stride + 0.5*stride.
// The /2 and /640 fold into a single ×(1/1280) for the centers; w,h take
// a single ×(1/640). All arithmetic is fp16 via fp16_fma (fused single-
// rounding multiply-add). grid col/row and stride arrive as small i32 and
// are converted with i32_to_fp16; the head's FSM derives them from the
// streaming anchor counter (no ROM).
//
// LATENCY = 7 cycles from valid_i to valid_o. Throughput 1 anchor/cycle.
// fp16 arithmetic ⇒ validated to a fp16-ULP tolerance (see DV), not bit-
// exact; the integer stages of the head (reduce_max/topk/gather) are exact.

module box_affine (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               valid_i,
  input  logic signed [7:0]  l_i,
  input  logic signed [7:0]  t_i,
  input  logic signed [7:0]  r_i,
  input  logic signed [7:0]  b_i,
  input  logic        [15:0] s_box_i,    // fp16 dequant scale
  input  logic signed [31:0] col_i,      // grid column (0..gridW-1)
  input  logic signed [31:0] row_i,      // grid row
  input  logic signed [31:0] stride_i,   // 8 / 16 / 32

  output logic               valid_o,
  output logic        [15:0] cx_o,        // normalized fp16
  output logic        [15:0] cy_o,
  output logic        [15:0] w_o,
  output logic        [15:0] h_o
);

  localparam logic [15:0] FP16_ONE     = 16'h3C00;  //  1.0
  localparam logic [15:0] FP16_NEG_ONE = 16'hBC00;  // -1.0
  localparam logic [15:0] FP16_HALF    = 16'h3800;  //  0.5
  localparam logic [15:0] FP16_ZERO    = 16'h0000;  //  0.0
  localparam logic [15:0] FP16_INV640  = 16'h1666;  //  1/640
  localparam logic [15:0] FP16_INV1280 = 16'h1266;  //  1/1280

  // ─────────────── valid pipeline (7 stages) ───────────────
  logic [6:0] vq;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) vq <= 7'b0;
    else         vq <= {vq[5:0], valid_i};
  end
  assign valid_o = vq[6];

  // ─────────────── S0: register inputs ───────────────
  logic signed [31:0] l0, t0, r0, b0, col0, row0, stride0;
  logic        [15:0] sbox0;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      l0<=0; t0<=0; r0<=0; b0<=0; col0<=0; row0<=0; stride0<=0; sbox0<=16'h0;
    end else begin
      l0<=32'(signed'(l_i)); t0<=32'(signed'(t_i));
      r0<=32'(signed'(r_i)); b0<=32'(signed'(b_i));
      col0<=col_i; row0<=row_i; stride0<=stride_i; sbox0<=s_box_i;
    end
  end

  // ─────────────── S1: i32→fp16 conversions ───────────────
  logic [15:0] l_f1, t_f1, r_f1, b_f1, col_f1, row_f1, stride_f1, sbox1;
  // i32_to_fp16's auto-shift output is unneeded here (all inputs are small).
  // verilator lint_off UNUSEDSIGNAL
  logic [4:0]  us0, us1, us2, us3, us4, us5, us6;
  // verilator lint_on UNUSEDSIGNAL
  i32_to_fp16 u_l   (.clk_i, .rst_ni, .x_i(l0),     .y_o(l_f1),     .shift_o(us0));
  i32_to_fp16 u_t   (.clk_i, .rst_ni, .x_i(t0),     .y_o(t_f1),     .shift_o(us1));
  i32_to_fp16 u_r   (.clk_i, .rst_ni, .x_i(r0),     .y_o(r_f1),     .shift_o(us2));
  i32_to_fp16 u_b   (.clk_i, .rst_ni, .x_i(b0),     .y_o(b_f1),     .shift_o(us3));
  i32_to_fp16 u_col (.clk_i, .rst_ni, .x_i(col0),   .y_o(col_f1),   .shift_o(us4));
  i32_to_fp16 u_row (.clk_i, .rst_ni, .x_i(row0),   .y_o(row_f1),   .shift_o(us5));
  i32_to_fp16 u_st  (.clk_i, .rst_ni, .x_i(stride0),.y_o(stride_f1),.shift_o(us6));
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) sbox1 <= 16'h0; else sbox1 <= sbox0;
  end

  // ─────────────── S2: dequant d = ltrb_f * S_box; half = 0.5*stride ───────────────
  logic [15:0] dl2, dt2, dr2, db2, half2;
  logic [15:0] col_f2, row_f2, stride_f2;
  fp16_fma u_dl   (.clk_i, .rst_ni, .a_i(l_f1), .b_i(sbox1),     .c_i(FP16_ZERO), .y_o(dl2));
  fp16_fma u_dt   (.clk_i, .rst_ni, .a_i(t_f1), .b_i(sbox1),     .c_i(FP16_ZERO), .y_o(dt2));
  fp16_fma u_dr   (.clk_i, .rst_ni, .a_i(r_f1), .b_i(sbox1),     .c_i(FP16_ZERO), .y_o(dr2));
  fp16_fma u_db   (.clk_i, .rst_ni, .a_i(b_f1), .b_i(sbox1),     .c_i(FP16_ZERO), .y_o(db2));
  fp16_fma u_half (.clk_i, .rst_ni, .a_i(stride_f1), .b_i(FP16_HALF), .c_i(FP16_ZERO), .y_o(half2));
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin col_f2<=16'h0; row_f2<=16'h0; stride_f2<=16'h0; end
    else begin col_f2<=col_f1; row_f2<=row_f1; stride_f2<=stride_f1; end
  end

  // ─────────────── S3: cx_center = col*stride + half ───────────────
  logic [15:0] cxc3, cyc3, stride_f3, dl3, dt3, dr3, db3;
  fp16_fma u_cxc (.clk_i, .rst_ni, .a_i(col_f2), .b_i(stride_f2), .c_i(half2), .y_o(cxc3));
  fp16_fma u_cyc (.clk_i, .rst_ni, .a_i(row_f2), .b_i(stride_f2), .c_i(half2), .y_o(cyc3));
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin stride_f3<=16'h0; dl3<=16'h0; dt3<=16'h0; dr3<=16'h0; db3<=16'h0; end
    else begin stride_f3<=stride_f2; dl3<=dl2; dt3<=dt2; dr3<=dr2; db3<=db2; end
  end

  // ─────────────── S4: xyxy = center ± d*stride ───────────────
  logic [15:0] stride_neg3;
  assign stride_neg3 = {~stride_f3[15], stride_f3[14:0]};  // fp16 sign flip
  logic [15:0] x1_4, x2_4, y1_4, y2_4;
  fp16_fma u_x1 (.clk_i, .rst_ni, .a_i(dl3), .b_i(stride_neg3), .c_i(cxc3), .y_o(x1_4));
  fp16_fma u_x2 (.clk_i, .rst_ni, .a_i(dr3), .b_i(stride_f3),   .c_i(cxc3), .y_o(x2_4));
  fp16_fma u_y1 (.clk_i, .rst_ni, .a_i(dt3), .b_i(stride_neg3), .c_i(cyc3), .y_o(y1_4));
  fp16_fma u_y2 (.clk_i, .rst_ni, .a_i(db3), .b_i(stride_f3),   .c_i(cyc3), .y_o(y2_4));

  // ─────────────── S5: sum = x1+x2, diff = x2-x1 ───────────────
  logic [15:0] sumx5, diffx5, sumy5, diffy5;
  fp16_fma u_sx (.clk_i, .rst_ni, .a_i(x1_4), .b_i(FP16_ONE),     .c_i(x2_4), .y_o(sumx5));
  fp16_fma u_dx (.clk_i, .rst_ni, .a_i(x1_4), .b_i(FP16_NEG_ONE), .c_i(x2_4), .y_o(diffx5));
  fp16_fma u_sy (.clk_i, .rst_ni, .a_i(y1_4), .b_i(FP16_ONE),     .c_i(y2_4), .y_o(sumy5));
  fp16_fma u_dy (.clk_i, .rst_ni, .a_i(y1_4), .b_i(FP16_NEG_ONE), .c_i(y2_4), .y_o(diffy5));

  // ─────────────── S6: normalize (cx,cy ×1/1280 ; w,h ×1/640) ───────────────
  fp16_fma u_cx (.clk_i, .rst_ni, .a_i(sumx5),  .b_i(FP16_INV1280), .c_i(FP16_ZERO), .y_o(cx_o));
  fp16_fma u_cy (.clk_i, .rst_ni, .a_i(sumy5),  .b_i(FP16_INV1280), .c_i(FP16_ZERO), .y_o(cy_o));
  fp16_fma u_w  (.clk_i, .rst_ni, .a_i(diffx5), .b_i(FP16_INV640),  .c_i(FP16_ZERO), .y_o(w_o));
  fp16_fma u_h  (.clk_i, .rst_ni, .a_i(diffy5), .b_i(FP16_INV640),  .c_i(FP16_ZERO), .y_o(h_o));

endmodule
