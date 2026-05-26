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
// Pipeline. The two shared leaf IPs are pipelined: i32_to_fp16 takes
// I2F_LAT cycles and fp16_fma takes FMA_LAT cycles (mirrored from
// fp16_lat_pkg in the localparams below). The two parallel datapaths have
// different depths and must meet cycle-aligned at the final fmas, so every
// value carried *past* a compute stage is delayed by a shift-register whose
// depth is derived from I2F_LAT / FMA_LAT.
//
// Arrival cycles, measured relative to the S0 input register (cycle 0 = the
// cycle the S0 output is present):
//
//   Distance path (per side): 1 mul + 4 add-tree levels, all fp16_fma:
//     m[s][i]   ready at        FMA_LAT
//     add L1    ready at      2*FMA_LAT
//     add L2    ready at      3*FMA_LAT
//     add L3    ready at      4*FMA_LAT
//     d[s]      ready at      5*FMA_LAT          (long pole)
//
//   Center path:
//     cx/cy/stride_fp16  ready at  I2F_LAT
//     half_stride        ready at  I2F_LAT + FMA_LAT
//                          (cx_fp16/stride_fp16 are delayed FMA_LAT so all
//                           three u_cxc inputs land together at I2F_LAT+FMA_LAT)
//     cx_center/cy_center ready at I2F_LAT + 2*FMA_LAT
//
//   Final fma (per coord): inputs must all arrive at max(...) = 5*FMA_LAT:
//     cx_center/cy_center delayed by 3*FMA_LAT − I2F_LAT
//     neg_stride/stride_fp16 delayed from I2F_LAT to 5*FMA_LAT
//                          (extra delay 5*FMA_LAT − I2F_LAT)
//     output ready at      5*FMA_LAT + FMA_LAT = 6*FMA_LAT
//
// Total latency = 1 (S0 reg) + 6*FMA_LAT = 31 cycles. Throughput = 1/cycle.
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

  // Leaf-IP latencies — mirror of fp16_lat_pkg (see header). DV-validated;
  // do NOT import the package (this IP lives in ~many generated build lists).
  localparam int unsigned I2F_LAT = 2;   // == fp16_lat_pkg::I32_TO_FP16_LAT
  localparam int unsigned FMA_LAT = 5;   // == fp16_lat_pkg::FP16_FMA_LAT

  // Arrival cycles (relative to S0 output) of the values that merge at the
  // final fmas, and the resulting bypass-delay depths.
  localparam int unsigned D_READY        = 5*FMA_LAT;            // d[s]
  localparam int unsigned CXC_READY       = I2F_LAT + 2*FMA_LAT; // cx_center
  localparam int unsigned FINAL_IN        = D_READY;             // long pole
  // u_cxc fires at I2F_LAT+FMA_LAT: delay cx_fp16/stride_fp16 by FMA_LAT
  // so they align with half_stride at the u_cxc inputs.
  localparam int unsigned CXC_ALIGN_DLY   = FMA_LAT;
  // cx_center/cy_center → final fma input.
  localparam int unsigned CXC_FINAL_DLY   = FINAL_IN - CXC_READY; // 3*FMA-I2F
  // stride_fp16 (from i2f at I2F_LAT) → final fma input.
  localparam int unsigned STRIDE_FINAL_DLY = FINAL_IN - I2F_LAT;  // 5*FMA-I2F
  // valid_i → valid_o total latency: 1 (S0 reg) + 6*FMA_LAT.
  localparam int unsigned TOTAL_LAT = 1 + 6*FMA_LAT;

  // LATENCY = 31 cycles from valid_i high to valid_o high (documented above).

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

  // ─────────────── i32 → fp16 conversions (I2F_LAT cycles) ───────────────
  // Outputs (cx/cy/stride_fp16_s1) appear at cycle I2F_LAT.
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

  // ─────────────── valid pipeline (TOTAL_LAT-deep shift register) ───────────────
  // v0 is the S0 register valid (1 cycle of latency already). The remaining
  // TOTAL_LAT-1 stages track the rest of the pipeline to the final fma out.
  logic [TOTAL_LAT-2:0] valid_sr;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) valid_sr <= '0;
    else         valid_sr <= {valid_sr[TOTAL_LAT-3:0], v0};
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

  // half_stride = stride_fp16 * 0.5, via fp16_fma. The i2f outputs land at
  // cycle I2F_LAT; this fma's output (half_stride) lands at I2F_LAT+FMA_LAT.
  logic [15:0] half_stride;
  fp16_fma u_half_stride (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (stride_fp16_s1),
    .b_i    (FP16_HALF),
    .c_i    (FP16_ZERO),
    .y_o    (half_stride)
  );

  // Align cx_fp16 / cy_fp16 / stride_fp16 (available at I2F_LAT) to the
  // u_cxc/u_cyc fmas, which fire when half_stride is present (I2F_LAT+FMA_LAT).
  // Delay each by CXC_ALIGN_DLY (= FMA_LAT) so all three inputs land together.
  logic [15:0] cx_fp16_dl     [CXC_ALIGN_DLY];
  logic [15:0] cy_fp16_dl     [CXC_ALIGN_DLY];
  logic [15:0] stride_fp16_dl [CXC_ALIGN_DLY];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < CXC_ALIGN_DLY; i++) begin
        cx_fp16_dl[i]     <= 16'h0;
        cy_fp16_dl[i]     <= 16'h0;
        stride_fp16_dl[i] <= 16'h0;
      end
    end else begin
      cx_fp16_dl[0]     <= cx_fp16_s1;
      cy_fp16_dl[0]     <= cy_fp16_s1;
      stride_fp16_dl[0] <= stride_fp16_s1;
      for (int i = 1; i < CXC_ALIGN_DLY; i++) begin
        cx_fp16_dl[i]     <= cx_fp16_dl[i-1];
        cy_fp16_dl[i]     <= cy_fp16_dl[i-1];
        stride_fp16_dl[i] <= stride_fp16_dl[i-1];
      end
    end
  end
  logic [15:0] cx_fp16_a, cy_fp16_a, stride_fp16_a;
  assign cx_fp16_a     = cx_fp16_dl[CXC_ALIGN_DLY-1];
  assign cy_fp16_a     = cy_fp16_dl[CXC_ALIGN_DLY-1];
  assign stride_fp16_a = stride_fp16_dl[CXC_ALIGN_DLY-1];

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

  // cx_center = cx_fp16 * stride_fp16 + half_stride.
  // All three inputs land at I2F_LAT+FMA_LAT; output at I2F_LAT+2*FMA_LAT.
  logic [15:0] cx_center, cy_center;
  fp16_fma u_cxc (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (cx_fp16_a),
    .b_i    (stride_fp16_a),
    .c_i    (half_stride),
    .y_o    (cx_center)
  );
  fp16_fma u_cyc (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (cy_fp16_a),
    .b_i    (stride_fp16_a),
    .c_i    (half_stride),
    .y_o    (cy_center)
  );

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

  // Skew cx_center / cy_center (ready at I2F_LAT+2*FMA_LAT) to the final
  // fma input cycle (5*FMA_LAT) via a CXC_FINAL_DLY-deep delay line.
  logic [15:0] cx_center_dl [CXC_FINAL_DLY];
  logic [15:0] cy_center_dl [CXC_FINAL_DLY];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < CXC_FINAL_DLY; i++) begin
        cx_center_dl[i] <= 16'h0;
        cy_center_dl[i] <= 16'h0;
      end
    end else begin
      cx_center_dl[0] <= cx_center;
      cy_center_dl[0] <= cy_center;
      for (int i = 1; i < CXC_FINAL_DLY; i++) begin
        cx_center_dl[i] <= cx_center_dl[i-1];
        cy_center_dl[i] <= cy_center_dl[i-1];
      end
    end
  end
  logic [15:0] cx_center_f, cy_center_f;
  assign cx_center_f = cx_center_dl[CXC_FINAL_DLY-1];
  assign cy_center_f = cy_center_dl[CXC_FINAL_DLY-1];

  // Skew stride_fp16 (ready at I2F_LAT) to the final fma input cycle
  // (5*FMA_LAT) via a STRIDE_FINAL_DLY-deep delay line.
  logic [15:0] stride_fp16_dl2 [STRIDE_FINAL_DLY];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < STRIDE_FINAL_DLY; i++) stride_fp16_dl2[i] <= 16'h0;
    end else begin
      stride_fp16_dl2[0] <= stride_fp16_s1;
      for (int i = 1; i < STRIDE_FINAL_DLY; i++)
        stride_fp16_dl2[i] <= stride_fp16_dl2[i-1];
    end
  end
  logic [15:0] stride_fp16_f;
  assign stride_fp16_f = stride_fp16_dl2[STRIDE_FINAL_DLY-1];

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

  // ─────────────── Final stage: final coordinates ───────────────
  // d[s], cx_center_f/cy_center_f and stride_fp16_f all land at 5*FMA_LAT.
  // neg_stride = -stride_fp16 (sign-bit flip)
  logic [15:0] neg_stride_f;
  assign neg_stride_f = {~stride_fp16_f[15], stride_fp16_f[14:0]};

  // x1 = d_l * (-stride) + cx_center
  // x2 = d_r * (+stride) + cx_center
  // y1 = d_t * (-stride) + cy_center
  // y2 = d_b * (+stride) + cy_center
  fp16_fma u_x1 (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (d_s5[0]),         // d_l
    .b_i    (neg_stride_f),
    .c_i    (cx_center_f),
    .y_o    (x1_o)
  );
  fp16_fma u_y1 (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (d_s5[1]),         // d_t
    .b_i    (neg_stride_f),
    .c_i    (cy_center_f),
    .y_o    (y1_o)
  );
  fp16_fma u_x2 (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (d_s5[2]),         // d_r
    .b_i    (stride_fp16_f),
    .c_i    (cx_center_f),
    .y_o    (x2_o)
  );
  fp16_fma u_y2 (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (d_s5[3]),         // d_b
    .b_i    (stride_fp16_f),
    .c_i    (cy_center_f),
    .y_o    (y2_o)
  );

  assign valid_o = valid_sr[TOTAL_LAT-2];

  // Silence unused-bit warnings.
  logic _unused;
  assign _unused = ^{cx0[31:16], cy0[31:16], stride0[31:16], 1'b0};

endmodule
