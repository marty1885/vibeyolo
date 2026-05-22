// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// box_decode — DFL detection-head distribution → xyxy box decode.
//
// For each grid cell, 4 sides s ∈ {l, t, r, b} each have a 16-bin
// softmaxed probability distribution. The DFL "distance" per side is
//   d[s] = Σ_{i=0..15} p[s][i] * i        (in bin units; multiply by
//                                          stride for pixel units)
// Box xyxy (pixel units) at grid (cx, cy) with stride `S`:
//   x1 = (cx + 0.5 - d_l) * S = (cx + 0.5)*S − d_l*S
//   y1 = (cy + 0.5 - d_t) * S
//   x2 = (cx + 0.5 + d_r) * S
//   y2 = (cy + 0.5 + d_b) * S
//
// Algebraic form used: precompute cx_center = (cx + 0.5)*S in fp16, then
//   x1 = cx_center − d_l*S    via fp16_fma(d_l, −S, cx_center)
//   x2 = cx_center + d_r*S    via fp16_fma(d_r, +S, cx_center)
// Likewise for y1, y2.
//
// Pipeline (each "S<n>" stage = 1 register-to-register hop):
//
//   cycle 0: inputs presented
//   cycle 1: S0 input register (p, cx, cy, stride captured)
//   cycle 2: S1  i32_to_fp16 outputs (cx_fp16, cy_fp16, stride_fp16)
//                64 fp16_fma multipliers (m[s][i] = p[s][i] * bin_fp16[i])
//   cycle 3: S2  add-tree L1 per side (8 adds via fp16_fma);
//                half_stride = stride_fp16 * 0.5
//   cycle 4: S3  add-tree L2 per side (4 adds);
//                cx_center = cx_fp16*stride + half_stride (fp16_fma)
//                cy_center = cy_fp16*stride + half_stride
//   cycle 5: S4  add-tree L3 per side (2 adds)
//   cycle 6: S5  add-tree L4 per side (1 add) → d[s]
//   cycle 7: S6  final fp16_fma per coord → x1, y1, x2, y2
//
// Total latency = 7 cycles. Throughput = 1 box per cycle.
//
// Bin constants 0..15 are baked as fp16 literals. The multiplier for i=0
// is wasted (m[s][0]=0) but the tree structure is uniform; the optimizer
// can prune.
//
// Design notes:
//   * Uses fp16_fma everywhere — both pure multiplies (c=0) and pure adds
//     (b=1.0). This keeps a single arithmetic building block and inherits
//     its RNE rounding semantics.
//   * Negation of stride is a sign-bit flip (fp16 is sign-magnitude). No
//     special handling is needed unless stride is zero — in which case
//     the boxes degenerate to all-cx_center / cy_center values (a sane
//     default for an invalid stride).

module box_decode (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               valid_i,
  input  logic        [15:0] p_l_i [16],
  input  logic        [15:0] p_t_i [16],
  input  logic        [15:0] p_r_i [16],
  input  logic        [15:0] p_b_i [16],
  input  logic signed [15:0] cx_i,
  input  logic signed [15:0] cy_i,
  input  logic signed [15:0] stride_i,

  output logic               valid_o,
  output logic        [15:0] x1_o,
  output logic        [15:0] y1_o,
  output logic        [15:0] x2_o,
  output logic        [15:0] y2_o
);

  // LATENCY = 7 cycles from valid_i high to valid_o high (documented above).

  // Baked fp16 constants for i = 0..15.
  // 0 = 0x0000, 1 = 0x3C00, 2 = 0x4000, 3 = 0x4200, 4 = 0x4400,
  // 5 = 0x4500, 6 = 0x4600, 7 = 0x4700, 8 = 0x4800, 9 = 0x4880,
  // 10 = 0x4900, 11 = 0x4980, 12 = 0x4A00, 13 = 0x4A80, 14 = 0x4B00,
  // 15 = 0x4B80
  /* verilator lint_off LITENDIAN */
  localparam logic [15:0] BIN_FP16 [0:15] = '{
    16'h0000, 16'h3C00, 16'h4000, 16'h4200,
    16'h4400, 16'h4500, 16'h4600, 16'h4700,
    16'h4800, 16'h4880, 16'h4900, 16'h4980,
    16'h4A00, 16'h4A80, 16'h4B00, 16'h4B80
  };
  /* verilator lint_on LITENDIAN */

  localparam logic [15:0] FP16_ONE      = 16'h3C00;  // 1.0
  localparam logic [15:0] FP16_ZERO     = 16'h0000;
  localparam logic [15:0] FP16_HALF     = 16'h3800;  // 0.5

  // ─────────────── S0: register inputs ───────────────
  logic        v0;
  logic [15:0] p0 [4][16];                // [side][bin]
  logic signed [31:0] cx0, cy0, stride0;  // sign-extended to i32

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v0      <= 1'b0;
      cx0     <= 32'sd0;
      cy0     <= 32'sd0;
      stride0 <= 32'sd0;
      for (int s = 0; s < 4; s++)
        for (int i = 0; i < 16; i++)
          p0[s][i] <= 16'h0;
    end else begin
      v0      <= valid_i;
      cx0     <= 32'(signed'(cx_i));
      cy0     <= 32'(signed'(cy_i));
      stride0 <= 32'(signed'(stride_i));
      for (int i = 0; i < 16; i++) begin
        p0[0][i] <= p_l_i[i];
        p0[1][i] <= p_t_i[i];
        p0[2][i] <= p_r_i[i];
        p0[3][i] <= p_b_i[i];
      end
    end
  end

  // ─────────────── i32 → fp16 conversions (1 cycle) ───────────────
  // Output appears at end of S1.
  logic [15:0] cx_fp16_s1, cy_fp16_s1, stride_fp16_s1;

  // Box-decode inputs (cx, cy, stride) are all small integers bounded by
  // the image dimensions (≤ a few thousand), so i32_to_fp16's auto-
  // prescale always reports shift_o == 0. Tie off.
  logic [4:0] unused_sh_cx, unused_sh_cy, unused_sh_st;

  i32_to_fp16 u_cvt_cx (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (cx0),
    .y_o    (cx_fp16_s1),
    .shift_o(unused_sh_cx)
  );
  i32_to_fp16 u_cvt_cy (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (cy0),
    .y_o    (cy_fp16_s1),
    .shift_o(unused_sh_cy)
  );
  i32_to_fp16 u_cvt_st (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (stride0),
    .y_o    (stride_fp16_s1),
    .shift_o(unused_sh_st)
  );

  // ─────────────── valid pipeline (one bit per stage) ───────────────
  logic v1, v2, v3, v4, v5, v6;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0;
      v4 <= 1'b0; v5 <= 1'b0; v6 <= 1'b0;
    end else begin
      v1 <= v0;
      v2 <= v1;
      v3 <= v2;
      v4 <= v3;
      v5 <= v4;
      v6 <= v5;
    end
  end

  // ─────────────── S1: 64 parallel multiplies p[s][i] * bin[i] ───────────────
  // fp16_fma is registered → outputs visible at end of S1.
  logic [15:0] m_s1 [4][16];
  generate
    for (genvar gs = 0; gs < 4; gs = gs + 1) begin : g_side_mul
      for (genvar gi = 0; gi < 16; gi = gi + 1) begin : g_bin_mul
        fp16_fma u_mul (
          .clk_i  (clk_i),
          .rst_ni (rst_ni),
          .a_i    (p0[gs][gi]),
          .b_i    (BIN_FP16[gi]),
          .c_i    (FP16_ZERO),
          .y_o    (m_s1[gs][gi])
        );
      end
    end
  endgenerate

  // ─────────────── S2: add tree level 1 (16 → 8) per side ───────────────
  logic [15:0] t8_s2 [4][8];
  generate
    for (genvar gs = 0; gs < 4; gs = gs + 1) begin : g_side_t1
      for (genvar gi = 0; gi < 8; gi = gi + 1) begin : g_pair_t1
        fp16_fma u_add (
          .clk_i  (clk_i),
          .rst_ni (rst_ni),
          .a_i    (m_s1[gs][2*gi]),
          .b_i    (FP16_ONE),
          .c_i    (m_s1[gs][2*gi+1]),
          .y_o    (t8_s2[gs][gi])
        );
      end
    end
  endgenerate

  // half_stride = stride_fp16 * 0.5. Computed in S2 from S1 stride.
  logic [15:0] half_stride_s2;
  fp16_fma u_half_stride (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (stride_fp16_s1),
    .b_i    (FP16_HALF),
    .c_i    (FP16_ZERO),
    .y_o    (half_stride_s2)
  );

  // Pipe stride_fp16 and cx_fp16, cy_fp16 from S1 through to S2.
  logic [15:0] stride_fp16_s2, cx_fp16_s2, cy_fp16_s2;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      stride_fp16_s2 <= 16'h0;
      cx_fp16_s2     <= 16'h0;
      cy_fp16_s2     <= 16'h0;
    end else begin
      stride_fp16_s2 <= stride_fp16_s1;
      cx_fp16_s2     <= cx_fp16_s1;
      cy_fp16_s2     <= cy_fp16_s1;
    end
  end

  // ─────────────── S3: add tree level 2 (8 → 4) per side ───────────────
  logic [15:0] t4_s3 [4][4];
  generate
    for (genvar gs = 0; gs < 4; gs = gs + 1) begin : g_side_t2
      for (genvar gi = 0; gi < 4; gi = gi + 1) begin : g_pair_t2
        fp16_fma u_add (
          .clk_i  (clk_i),
          .rst_ni (rst_ni),
          .a_i    (t8_s2[gs][2*gi]),
          .b_i    (FP16_ONE),
          .c_i    (t8_s2[gs][2*gi+1]),
          .y_o    (t4_s3[gs][gi])
        );
      end
    end
  endgenerate

  // cx_center, cy_center at end of S3.
  // cx_center = cx_fp16 * stride_fp16 + half_stride
  logic [15:0] cx_center_s3, cy_center_s3;
  fp16_fma u_cxc (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (cx_fp16_s2),
    .b_i    (stride_fp16_s2),
    .c_i    (half_stride_s2),
    .y_o    (cx_center_s3)
  );
  fp16_fma u_cyc (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (cy_fp16_s2),
    .b_i    (stride_fp16_s2),
    .c_i    (half_stride_s2),
    .y_o    (cy_center_s3)
  );

  // Pipe stride_fp16 through S2→S3
  logic [15:0] stride_fp16_s3;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) stride_fp16_s3 <= 16'h0;
    else         stride_fp16_s3 <= stride_fp16_s2;
  end

  // ─────────────── S4: add tree level 3 (4 → 2) per side ───────────────
  logic [15:0] t2_s4 [4][2];
  generate
    for (genvar gs = 0; gs < 4; gs = gs + 1) begin : g_side_t3
      for (genvar gi = 0; gi < 2; gi = gi + 1) begin : g_pair_t3
        fp16_fma u_add (
          .clk_i  (clk_i),
          .rst_ni (rst_ni),
          .a_i    (t4_s3[gs][2*gi]),
          .b_i    (FP16_ONE),
          .c_i    (t4_s3[gs][2*gi+1]),
          .y_o    (t2_s4[gs][gi])
        );
      end
    end
  endgenerate

  // Pipe cx_center, cy_center, stride_fp16 S3→S4
  logic [15:0] cx_center_s4, cy_center_s4, stride_fp16_s4;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cx_center_s4 <= 16'h0; cy_center_s4 <= 16'h0; stride_fp16_s4 <= 16'h0;
    end else begin
      cx_center_s4   <= cx_center_s3;
      cy_center_s4   <= cy_center_s3;
      stride_fp16_s4 <= stride_fp16_s3;
    end
  end

  // ─────────────── S5: add tree level 4 (2 → 1) per side → d[s] ───────────────
  logic [15:0] d_s5 [4];
  generate
    for (genvar gs = 0; gs < 4; gs = gs + 1) begin : g_side_t4
      fp16_fma u_add (
        .clk_i  (clk_i),
        .rst_ni (rst_ni),
        .a_i    (t2_s4[gs][0]),
        .b_i    (FP16_ONE),
        .c_i    (t2_s4[gs][1]),
        .y_o    (d_s5[gs])
      );
    end
  endgenerate

  // Pipe cx_center, cy_center, stride_fp16 S4→S5
  logic [15:0] cx_center_s5, cy_center_s5, stride_fp16_s5;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cx_center_s5 <= 16'h0; cy_center_s5 <= 16'h0; stride_fp16_s5 <= 16'h0;
    end else begin
      cx_center_s5   <= cx_center_s4;
      cy_center_s5   <= cy_center_s4;
      stride_fp16_s5 <= stride_fp16_s4;
    end
  end

  // ─────────────── S6: final coordinates ───────────────
  // neg_stride = -stride_fp16 (sign-bit flip)
  logic [15:0] neg_stride_s5;
  assign neg_stride_s5 = {~stride_fp16_s5[15], stride_fp16_s5[14:0]};

  // x1 = d_l * (-stride) + cx_center
  // x2 = d_r * (+stride) + cx_center
  // y1 = d_t * (-stride) + cy_center
  // y2 = d_b * (+stride) + cy_center
  fp16_fma u_x1 (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (d_s5[0]),         // d_l
    .b_i    (neg_stride_s5),
    .c_i    (cx_center_s5),
    .y_o    (x1_o)
  );
  fp16_fma u_y1 (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (d_s5[1]),         // d_t
    .b_i    (neg_stride_s5),
    .c_i    (cy_center_s5),
    .y_o    (y1_o)
  );
  fp16_fma u_x2 (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (d_s5[2]),         // d_r
    .b_i    (stride_fp16_s5),
    .c_i    (cx_center_s5),
    .y_o    (x2_o)
  );
  fp16_fma u_y2 (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (d_s5[3]),         // d_b
    .b_i    (stride_fp16_s5),
    .c_i    (cy_center_s5),
    .y_o    (y2_o)
  );

  assign valid_o = v6;

  // Silence unused-bit warnings.
  logic _unused;
  assign _unused = ^{cx0[31:16], cy0[31:16], stride0[31:16], 1'b0};

endmodule
