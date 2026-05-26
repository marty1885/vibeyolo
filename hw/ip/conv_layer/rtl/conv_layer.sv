// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// conv_layer — fully-parameterized tiled Conv (+ optional residual + optional SiLU).
//
// This is the single reusable IP behind every YOLO26n conv layer. A real
// layer becomes a ~30-line shim that:
//   1. imports scale_pkg::LAYER_<i>_*,
//   2. forwards them as parameters to conv_layer,
//   3. exposes whatever flattened port shape the existing TB / next layer
//      expects.
//
// Architecture (identical to integ/_layer_template/layer_template.sv):
//   * P_COUT parallel dotN cells, each N=K*K*P_CIN lanes wide.
//   * Per output pixel, outer loop over cout_tile, inner over cin_tile.
//   * On last_cin the P_COUT-wide i32 accumulator is committed into
//     P_COUT parallel requant cells (i32→fp16→FMA→i8 sat).
//   * If SILU: P_COUT parallel act_silu LUTs.
//   * If RESIDUAL: P_COUT parallel add_rq cells fold an int8 residual
//     stream (r_i) with its own per-channel fp16 scale into the activation.
//
// Output handshake: valid_o + ready_i. Because every internal stage is a
// fixed-latency feed-forward pipeline (no internal storage), this IP is
// non-stalling: ready_o == 1 always. ready_i is accepted but currently
// unused — back-pressure across the network is gated at the system level,
// not inside the per-layer compute pipeline.
//
// Residual scale folding
// ──────────────────────
// add_rq takes two int8 streams plus three fp16 coefficients:
//   y = sat_i8( (a_i8 * scale_a + b_i8 * scale_b) * inv_out_scale + bias )
// We wire it as:
//   a_i8           = act/requant output of the convolution (= post-SiLU
//                    if SILU=1, else post-requant)
//   scale_a_fp16   = constant S_OUT_SILU (the activation scale of the
//                    conv stream)
//   b_i8           = r_i             (residual int8)
//   scale_b_fp16   = r_scale_i       (per-channel residual scale, fp16)
//   inv_out_scale  = constant 1.0 / S_OUT_SILU (we keep the same output
//                    quant grid as the non-residual path)
//   bias_fp16      = r_bias_i        (per-channel post-add bias)
// This convention means the caller folds any extra trim into r_bias_i.

// verilator lint_off UNUSEDPARAM
// verilator lint_off UNUSEDSIGNAL
module conv_layer
#(
  parameter int  CIN          = 128,
  parameter int  COUT         = 128,
  parameter int  K            = 3,
  parameter int  STRIDE       = 1,   // informational; layer uses tiles, not pixels
  parameter int  PAD          = 1,   // informational
  parameter int  H_OUT        = 40,  // informational
  parameter int  W_OUT        = 40,  // informational
  parameter int  P_COUT       = 16,
  parameter int  P_CIN        = 8,
  parameter int  RESIDUAL     = 0,
  parameter int  SILU         = 1,
  parameter real S_OUT_PRE    = 4.0 / 127.0,
  parameter real S_OUT_SILU   = 4.0 / 127.0
) (
  input  logic                                            clk_i,
  input  logic                                            rst_ni,

  // Input handshake
  input  logic                                            valid_i,
  output logic                                            ready_o,

  // Tile control
  input  logic                                            first_cin_i,
  input  logic                                            last_cin_i,
  input  logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_i,

  // K*K*P_CIN i8 lanes for the current cin_tile of the current patch
  input  logic signed [K*K*P_CIN-1:0][7:0]                x_i,
  // P_COUT × (K*K*P_CIN) i8 weights for (cout_tile, cin_tile)
  input  logic signed [P_COUT-1:0][K*K*P_CIN-1:0][7:0]    w_i,
  // P_COUT fp16 scale/bias for the current cout_tile
  input  logic        [P_COUT-1:0][15:0]                  scale_i,
  input  logic        [P_COUT-1:0][15:0]                  bias_i,

  // Residual side-band (only meaningful when RESIDUAL=1)
  input  logic signed [P_COUT-1:0][7:0]                   r_i,
  input  logic        [P_COUT-1:0][15:0]                  r_scale_i,
  input  logic        [P_COUT-1:0][15:0]                  r_bias_i,

  // Output handshake
  output logic                                            valid_o,
  input  logic                                            ready_i,
  output logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_o,
  output logic signed [P_COUT-1:0][7:0]                   y_o
);

  // ─── Derived constants ───────────────────────────────────
  localparam int N_LANE        = K * K * P_CIN;
  localparam int N_COUT_TILE   = (COUT + P_COUT - 1) / P_COUT;
  localparam int CT_W          = $clog2((N_COUT_TILE < 2) ? 2 : N_COUT_TILE);
  localparam int DOT_LAT       = 1 + $clog2(N_LANE);
  // requant / add_rq latencies grew when i32_to_fp16 (1→2) and fp16_fma
  // (3→5, deepened for 7nm 1 GHz timing) were pipelined. These mirror the
  // submodule totals:
  //   requant = I2F(2) + scale-bump(1) + FMA(5) + sat(1)            = 9
  //   add_rq  = I2F(2) + FMA(5)×3      + sat(1)                     = 18
  // (canonical leaf latencies live in fp16_lat_pkg; DV validates these).
  localparam int REQUANT_LAT   = 9;
  localparam int SILU_LAT      = 1;
  localparam int ADDRQ_LAT     = 18;

  // Feed-forward; no internal stall.
  assign ready_o = 1'b1;
  // verilator lint_off UNUSEDSIGNAL
  wire _unused_ready_i = ready_i;
  // verilator lint_on UNUSEDSIGNAL

  // ─── Stage A: P_COUT parallel dotN over the current cin tile ───
  logic [P_COUT-1:0]              dot_valid;
  logic signed [P_COUT-1:0][31:0] dot_acc;

  genvar gc;
  generate
    for (gc = 0; gc < P_COUT; gc++) begin : g_dot
      dotN #(.N(N_LANE)) u_dot (
        .clk_i  (clk_i),
        .rst_ni (rst_ni),
        .en_i   (valid_i),
        .clr_i  (1'b0),
        .a_i    (x_i),
        .b_i    (w_i[gc]),
        .y_o    (dot_acc[gc]),
        .valid_o(dot_valid[gc])
      );
    end
  endgenerate

  // ─── Pipeline tile control alongside dotN ──────────────────────
  logic                     first_sr [DOT_LAT];
  logic                     last_sr  [DOT_LAT];
  logic [CT_W-1:0]          ct_sr    [DOT_LAT];
  logic [P_COUT-1:0][15:0]  scale_sr [DOT_LAT];
  logic [P_COUT-1:0][15:0]  bias_sr  [DOT_LAT];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < DOT_LAT; i++) begin
        first_sr[i] <= 1'b0;
        last_sr [i] <= 1'b0;
        ct_sr   [i] <= '0;
        scale_sr[i] <= '0;
        bias_sr [i] <= '0;
      end
    end else begin
      first_sr[0] <= first_cin_i & valid_i;
      last_sr [0] <= last_cin_i  & valid_i;
      ct_sr   [0] <= cout_tile_idx_i;
      scale_sr[0] <= scale_i;
      bias_sr [0] <= bias_i;
      for (int i = 1; i < DOT_LAT; i++) begin
        first_sr[i] <= first_sr[i-1];
        last_sr [i] <= last_sr [i-1];
        ct_sr   [i] <= ct_sr   [i-1];
        scale_sr[i] <= scale_sr[i-1];
        bias_sr [i] <= bias_sr [i-1];
      end
    end
  end

  // ─── Stage B: i32 accumulator across cin tiles ─────────────────
  logic signed [P_COUT-1:0][31:0] acc_q;
  logic                           commit_q;
  logic [CT_W-1:0]                commit_ct_q;
  logic [P_COUT-1:0][15:0]        commit_scale_q;
  logic [P_COUT-1:0][15:0]        commit_bias_q;

  wire first_d = first_sr[DOT_LAT-1];
  wire last_d  = last_sr [DOT_LAT-1];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int c = 0; c < P_COUT; c++) acc_q[c] <= '0;
      commit_q       <= 1'b0;
      commit_ct_q    <= '0;
      commit_scale_q <= '0;
      commit_bias_q  <= '0;
    end else begin
      if (dot_valid[0]) begin
        for (int c = 0; c < P_COUT; c++) begin
          if (first_d) acc_q[c] <= dot_acc[c];
          else         acc_q[c] <= acc_q[c] + dot_acc[c];
        end
      end
      commit_q       <= dot_valid[0] & last_d;
      commit_ct_q    <= ct_sr   [DOT_LAT-1];
      commit_scale_q <= scale_sr[DOT_LAT-1];
      commit_bias_q  <= bias_sr [DOT_LAT-1];
    end
  end

  logic                           rq_valid_in;
  logic signed [P_COUT-1:0][31:0] rq_acc_in;
  logic [P_COUT-1:0][15:0]        rq_scale_in;
  logic [P_COUT-1:0][15:0]        rq_bias_in;
  logic [CT_W-1:0]                rq_ct_in;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rq_valid_in <= 1'b0;
      rq_acc_in   <= '0;
      rq_scale_in <= '0;
      rq_bias_in  <= '0;
      rq_ct_in    <= '0;
    end else begin
      rq_valid_in <= commit_q;
      rq_acc_in   <= acc_q;
      rq_scale_in <= commit_scale_q;
      rq_bias_in  <= commit_bias_q;
      rq_ct_in    <= commit_ct_q;
    end
  end

  // ─── Stage C: P_COUT parallel requant ──────────────────────────
  logic [P_COUT-1:0]             rq_valid;
  logic signed [P_COUT-1:0][7:0] rq_y;

  generate
    for (gc = 0; gc < P_COUT; gc++) begin : g_rq
      requant u_rq (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .valid_i      (rq_valid_in),
        .acc_i        (rq_acc_in[gc]),
        .scale_fp16_i (rq_scale_in[gc]),
        .bias_fp16_i  (rq_bias_in [gc]),
        .valid_o      (rq_valid[gc]),
        .y_o          (rq_y[gc])
      );
    end
  endgenerate

  // Pipeline cout_tile_idx alongside requant (+ optional silu, + optional add_rq).
  localparam int POST_LAT = REQUANT_LAT + ((SILU != 0) ? SILU_LAT : 0) + ((RESIDUAL != 0) ? ADDRQ_LAT : 0);
  logic [CT_W-1:0] ct_post [POST_LAT];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < POST_LAT; i++) ct_post[i] <= '0;
    end else begin
      ct_post[0] <= rq_ct_in;
      for (int i = 1; i < POST_LAT; i++) ct_post[i] <= ct_post[i-1];
    end
  end

  // ─── Stage D: optional SiLU ────────────────────────────────────
  logic signed [P_COUT-1:0][7:0] post_silu_y;
  logic                          post_silu_valid;

  generate
    if (SILU != 0) begin : g_silu_en
      for (gc = 0; gc < P_COUT; gc++) begin : g_silu
        act_silu #(.InScale(S_OUT_PRE), .OutScale(S_OUT_SILU)) u_silu (
          .clk_i (clk_i),
          .rst_ni(rst_ni),
          .x_i   (rq_y[gc]),
          .y_o   (post_silu_y[gc])
        );
      end
      logic silu_valid_q;
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) silu_valid_q <= 1'b0;
        else         silu_valid_q <= rq_valid[0];
      end
      assign post_silu_valid = silu_valid_q;
    end else begin : g_silu_bypass
      assign post_silu_y     = rq_y;
      assign post_silu_valid = rq_valid[0];
    end
  endgenerate

  // ─── Stage E: optional residual add_rq ─────────────────────────
  // S_OUT_SILU is a real parameter; convert to fp16 at elaboration.
  function automatic [15:0] real_to_fp16(input real v);
    // Minimal real→fp16 RNE converter for compile-time use.
    real    av;
    int     e;
    real    m;
    int     mant;
    logic   sign;
    int     biased_e;
    logic [15:0] r;
    begin
      if (v == 0.0) return 16'h0000;
      sign = (v < 0.0);
      av   = sign ? -v : v;
      e    = 0;
      m    = av;
      while (m >= 2.0) begin m = m / 2.0; e = e + 1; end
      while (m <  1.0) begin m = m * 2.0; e = e - 1; end
      // m in [1,2)
      biased_e = e + 15;
      if (biased_e <= 0) begin
        // subnormal: shift the mantissa down; for our use S is ~ 1/30 .. 1/0.03,
        // never subnormal, but cover the case.
        // verilator coverage_off
        r = 16'h0000;
        // verilator coverage_on
        return r;
      end
      if (biased_e >= 31) begin
        // saturate to +/- max normal
        // verilator coverage_off
        r = {sign, 5'd30, 10'h3ff};
        // verilator coverage_on
        return r;
      end
      // mantissa: (m - 1.0) * 2^10, RNE
      mant = $rtoi((m - 1.0) * 1024.0 + 0.5);
      if (mant >= 1024) begin
        mant = 0;
        biased_e = biased_e + 1;
        if (biased_e >= 31) begin
          // verilator coverage_off
          r = {sign, 5'd30, 10'h3ff};
          // verilator coverage_on
          return r;
        end
      end
      r = {sign, biased_e[4:0], mant[9:0]};
      return r;
    end
  endfunction

  localparam logic [15:0] FP16_S_OUT     = real_to_fp16(S_OUT_SILU);
  localparam logic [15:0] FP16_INV_S_OUT = real_to_fp16(1.0 / S_OUT_SILU);

  // Pipeline r_i / r_scale_i / r_bias_i alongside the conv pipeline so that
  // they arrive at add_rq at the same time as the matching post-SiLU sample.
  // The producer drives them aligned with valid_i (i.e. with the input
  // patch). The conv pipeline latency to post_silu_y is:
  //   DOT_LAT (dotN) + 1 (acc) + 1 (rq_in) + REQUANT_LAT [+ SILU_LAT].
  localparam int R_DELAY = DOT_LAT + 1 + 1 + REQUANT_LAT + ((SILU != 0) ? SILU_LAT : 0);

  generate
    if (RESIDUAL != 0) begin : g_res_en
      logic signed [P_COUT-1:0][7:0]  r_sr   [R_DELAY];
      logic        [P_COUT-1:0][15:0] rs_sr  [R_DELAY];
      logic        [P_COUT-1:0][15:0] rb_sr  [R_DELAY];
      // Gate the residual side-band by valid_i — we only want commits whose
      // beat was the last_cin to propagate (i.e. align with post_silu_valid).
      // Since R_DELAY tracks the same valid_i tick that triggered the
      // last_cin commit, we sample r_i/r_scale_i/r_bias_i on every valid_i
      // beat but only use them when last_cin_i was high. To make this simple
      // and exact, the driver should hold r_i/r_scale_i/r_bias_i steady
      // across the cin-tile sweep — they're per (pixel, cout_tile), not per
      // cin_tile. We sample on the last_cin beat.
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
          for (int i = 0; i < R_DELAY; i++) begin
            r_sr [i] <= '0;
            rs_sr[i] <= '0;
            rb_sr[i] <= '0;
          end
        end else begin
          if (last_cin_i & valid_i) begin
            r_sr [0] <= r_i;
            rs_sr[0] <= r_scale_i;
            rb_sr[0] <= r_bias_i;
          end
          for (int i = 1; i < R_DELAY; i++) begin
            r_sr [i] <= r_sr [i-1];
            rs_sr[i] <= rs_sr[i-1];
            rb_sr[i] <= rb_sr[i-1];
          end
        end
      end

      logic [P_COUT-1:0]             arq_valid;
      logic signed [P_COUT-1:0][7:0] arq_y;

      for (gc = 0; gc < P_COUT; gc++) begin : g_addrq
        add_rq u_arq (
          .clk_i                (clk_i),
          .rst_ni               (rst_ni),
          .valid_i              (post_silu_valid),
          .a_i8_i               (post_silu_y[gc]),
          .b_i8_i               (r_sr [R_DELAY-1][gc]),
          .scale_a_fp16_i       (FP16_S_OUT),
          .scale_b_fp16_i       (rs_sr[R_DELAY-1][gc]),
          .inv_out_scale_fp16_i (FP16_INV_S_OUT),
          .bias_fp16_i          (rb_sr[R_DELAY-1][gc]),
          .valid_o              (arq_valid[gc]),
          .y_o                  (arq_y[gc])
        );
      end

      assign y_o     = arq_y;
      assign valid_o = arq_valid[0];
      // verilator lint_off UNUSEDSIGNAL
      wire [P_COUT-2:0] _u_arq_v = arq_valid[P_COUT-1:1];
      // verilator lint_on UNUSEDSIGNAL
    end else begin : g_res_bypass
      // verilator lint_off UNUSEDSIGNAL
      wire _u_r     = |r_i;
      wire _u_rs    = |r_scale_i;
      wire _u_rb    = |r_bias_i;
      wire _u_const = |FP16_S_OUT | |FP16_INV_S_OUT;
      // verilator lint_on UNUSEDSIGNAL
      assign y_o     = post_silu_y;
      assign valid_o = post_silu_valid;
    end
  endgenerate

  assign cout_tile_idx_o = ct_post[POST_LAT-1];

  // verilator lint_off UNUSEDSIGNAL
  wire [P_COUT-2:0] _u_rq_valid_hi = rq_valid[P_COUT-1:1];
  wire [P_COUT-2:0] _u_dot_valid_hi = dot_valid[P_COUT-1:1];
  // verilator lint_on UNUSEDSIGNAL

endmodule
// verilator lint_on UNUSEDPARAM
// verilator lint_on UNUSEDSIGNAL
