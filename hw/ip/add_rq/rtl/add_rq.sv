// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// add_rq — residual-add requantizer.
//
// Adds two int8 streams that carry independent per-channel fp16 scales
// and emits a single requantized int8.
//
// Math (single-rounding per stage; see README):
//   fp_a   = i32_to_fp16(sign_extend(a_i8))
//   fp_b   = i32_to_fp16(sign_extend(b_i8))
//   ta     = fp_a * scale_a + 0
//   tb     = fp_b * scale_b + 0
//   sum    = ta * 1.0 + tb            // FMA reused as fp16 add
//   y_fp16 = sum * inv_out_scale + bias
//   y_o    = sat_i8( round_rne( y_fp16 ) )
//
// Composition (leaf-IP latencies after pipelining for the clock target):
//   stage 0: i32_to_fp16(a), i32_to_fp16(b)            (latency I2F_LAT=2)
//   stage 1: fp16_fma(fp_a, scale_a, 0)  → ta          (latency FMA_LAT=5)
//            fp16_fma(fp_b, scale_b, 0)  → tb
//   stage 2: fp16_fma(ta, 1.0, tb)       → sum         (latency FMA_LAT=5)
//   stage 3: fp16_fma(sum, inv_out_scale, bias) → fp_y (latency FMA_LAT=5)
//   stage 4: fp16_to_i8_sat(fp_y)        → y_o         (latency SAT_LAT=1)
//
// Total latency from valid_i to (valid_o, y_o) is
//   I2F_LAT + FMA_LAT + FMA_LAT + FMA_LAT + SAT_LAT = 2+5+5+5+1 = 18 cycles.
//
// The i32_to_fp16 / fp16_fma latencies live in fp16_lat_pkg (the canonical
// single source of truth). add_rq is instantiated in many generated layer
// build lists, so to avoid forcing the package into all of them it mirrors
// the values in local localparams below; the lockstep DV against the
// independent add_rq_ref golden fails if these ever drift.
//
// The scales/bias inputs are pipelined alongside the data — each side-channel
// is delayed by the latency of the leaf it bypasses at that stage — so the
// caller only needs to present them on the same cycle as a_i/b_i.

module add_rq (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               valid_i,
  input  logic signed  [7:0] a_i8_i,
  input  logic signed  [7:0] b_i8_i,
  input  logic        [15:0] scale_a_fp16_i,
  input  logic        [15:0] scale_b_fp16_i,
  input  logic        [15:0] inv_out_scale_fp16_i,
  input  logic        [15:0] bias_fp16_i,

  output logic               valid_o,
  output logic signed  [7:0] y_o
);

  localparam logic [15:0] FP16_ZERO = 16'h0000;
  localparam logic [15:0] FP16_ONE  = 16'h3C00;

  // Leaf-IP latencies — mirror of fp16_lat_pkg (see header). DV-validated.
  localparam int unsigned I2F_LAT = 2;   // == fp16_lat_pkg::I32_TO_FP16_LAT
  localparam int unsigned FMA_LAT = 5;   // == fp16_lat_pkg::FP16_FMA_LAT
  localparam int unsigned SAT_LAT = 1;   // fp16_to_i8_sat (unchanged)

  // ─── stage 0: int8 → fp16 via i32_to_fp16 ─────────────────────
  // Sign-extend i8 to i32 combinationally and feed to i32_to_fp16
  // (which registers its output).
  logic signed [31:0] a_i32, b_i32;
  assign a_i32 = 32'(a_i8_i);
  assign b_i32 = 32'(b_i8_i);

  logic [15:0] fp_a_s0, fp_b_s0;
  // a_i8 / b_i8 are sign-extended int8s; |x| ≤ 128 ≪ 2^16, so the
  // auto-prescale in i32_to_fp16 always yields shift_o == 0. Tie off.
  logic [4:0]  unused_shift_a, unused_shift_b;

  i32_to_fp16 u_a_i2f (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (a_i32),
    .y_o    (fp_a_s0),
    .shift_o(unused_shift_a)
  );

  i32_to_fp16 u_b_i2f (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (b_i32),
    .y_o    (fp_b_s0),
    .shift_o(unused_shift_b)
  );

  // Pipeline registers for the scales/bias across stage 0 (i32_to_fp16).
  // Delay by I2F_LAT so they realign with fp_a_s0/fp_b_s0 feeding stage 1.
  logic [15:0] scale_a_dl [I2F_LAT];
  logic [15:0] scale_b_dl [I2F_LAT];
  logic [15:0] inv_out_dl [I2F_LAT];
  logic [15:0] bias_dl    [I2F_LAT];
  logic [I2F_LAT-1:0] valid_dl0;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < I2F_LAT; i++) begin
        scale_a_dl[i] <= 16'd0;
        scale_b_dl[i] <= 16'd0;
        inv_out_dl[i] <= 16'd0;
        bias_dl[i]    <= 16'd0;
      end
      valid_dl0 <= '0;
    end else begin
      scale_a_dl[0] <= scale_a_fp16_i;
      scale_b_dl[0] <= scale_b_fp16_i;
      inv_out_dl[0] <= inv_out_scale_fp16_i;
      bias_dl[0]    <= bias_fp16_i;
      for (int i = 1; i < I2F_LAT; i++) begin
        scale_a_dl[i] <= scale_a_dl[i-1];
        scale_b_dl[i] <= scale_b_dl[i-1];
        inv_out_dl[i] <= inv_out_dl[i-1];
        bias_dl[i]    <= bias_dl[i-1];
      end
      valid_dl0 <= {valid_dl0[I2F_LAT-2:0], valid_i};
    end
  end
  logic [15:0] scale_a_s0, scale_b_s0, inv_out_s0, bias_s0;
  logic        valid_s0;
  assign scale_a_s0 = scale_a_dl[I2F_LAT-1];
  assign scale_b_s0 = scale_b_dl[I2F_LAT-1];
  assign inv_out_s0 = inv_out_dl[I2F_LAT-1];
  assign bias_s0    = bias_dl[I2F_LAT-1];
  assign valid_s0   = valid_dl0[I2F_LAT-1];

  // ─── stage 1: ta = fp_a * scale_a, tb = fp_b * scale_b ────────
  logic [15:0] ta_s1, tb_s1;

  fp16_fma u_fma_a (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (fp_a_s0),
    .b_i    (scale_a_s0),
    .c_i    (FP16_ZERO),
    .y_o    (ta_s1)
  );

  fp16_fma u_fma_b (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (fp_b_s0),
    .b_i    (scale_b_s0),
    .c_i    (FP16_ZERO),
    .y_o    (tb_s1)
  );

  // Delay inv_out/bias/valid by FMA_LAT across stage 1's fmas so they
  // realign with ta_s1/tb_s1.
  logic [15:0] inv_out_dl1 [FMA_LAT];
  logic [15:0] bias_dl1    [FMA_LAT];
  logic [FMA_LAT-1:0] valid_dl1;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < FMA_LAT; i++) begin
        inv_out_dl1[i] <= 16'd0;
        bias_dl1[i]    <= 16'd0;
      end
      valid_dl1 <= '0;
    end else begin
      inv_out_dl1[0] <= inv_out_s0;
      bias_dl1[0]    <= bias_s0;
      for (int i = 1; i < FMA_LAT; i++) begin
        inv_out_dl1[i] <= inv_out_dl1[i-1];
        bias_dl1[i]    <= bias_dl1[i-1];
      end
      valid_dl1 <= {valid_dl1[FMA_LAT-2:0], valid_s0};
    end
  end
  logic [15:0] inv_out_s1, bias_s1;
  logic        valid_s1;
  assign inv_out_s1 = inv_out_dl1[FMA_LAT-1];
  assign bias_s1    = bias_dl1[FMA_LAT-1];
  assign valid_s1   = valid_dl1[FMA_LAT-1];

  // ─── stage 2: sum = ta * 1.0 + tb (fp16 add via FMA) ──────────
  logic [15:0] sum_s2;

  fp16_fma u_fma_add (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (ta_s1),
    .b_i    (FP16_ONE),
    .c_i    (tb_s1),
    .y_o    (sum_s2)
  );

  // Delay inv_out/bias/valid by FMA_LAT across stage 2's fma so they
  // realign with sum_s2.
  logic [15:0] inv_out_dl2 [FMA_LAT];
  logic [15:0] bias_dl2    [FMA_LAT];
  logic [FMA_LAT-1:0] valid_dl2;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < FMA_LAT; i++) begin
        inv_out_dl2[i] <= 16'd0;
        bias_dl2[i]    <= 16'd0;
      end
      valid_dl2 <= '0;
    end else begin
      inv_out_dl2[0] <= inv_out_s1;
      bias_dl2[0]    <= bias_s1;
      for (int i = 1; i < FMA_LAT; i++) begin
        inv_out_dl2[i] <= inv_out_dl2[i-1];
        bias_dl2[i]    <= bias_dl2[i-1];
      end
      valid_dl2 <= {valid_dl2[FMA_LAT-2:0], valid_s1};
    end
  end
  logic [15:0] inv_out_s2, bias_s2;
  logic        valid_s2;
  assign inv_out_s2 = inv_out_dl2[FMA_LAT-1];
  assign bias_s2    = bias_dl2[FMA_LAT-1];
  assign valid_s2   = valid_dl2[FMA_LAT-1];

  // ─── stage 3: fp_y = sum * inv_out_scale + bias ───────────────
  logic [15:0] fp_y_s3;

  fp16_fma u_fma_out (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .a_i    (sum_s2),
    .b_i    (inv_out_s2),
    .c_i    (bias_s2),
    .y_o    (fp_y_s3)
  );

  // Delay valid by FMA_LAT across stage 3's fma so it realigns with fp_y_s3.
  logic [FMA_LAT-1:0] valid_dl3;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) valid_dl3 <= '0;
    else         valid_dl3 <= {valid_dl3[FMA_LAT-2:0], valid_s2};
  end
  logic valid_s3;
  assign valid_s3 = valid_dl3[FMA_LAT-1];

  // ─── stage 4: i8 saturate ─────────────────────────────────────
  fp16_to_i8_sat u_sat (
    .clk_i  (clk_i),
    .rst_ni (rst_ni),
    .x_i    (fp_y_s3),
    .y_o    (y_o)
  );

  // Delay valid by SAT_LAT across stage 4's fp16_to_i8_sat.
  logic [SAT_LAT-1:0] valid_dl4;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      valid_dl4 <= '0;
    end else begin
      valid_dl4[0] <= valid_s3;
      for (int i = 1; i < SAT_LAT; i++) valid_dl4[i] <= valid_dl4[i-1];
    end
  end
  assign valid_o = valid_dl4[SAT_LAT-1];

endmodule
