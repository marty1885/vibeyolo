// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// layer_template.sv — Canonical structure for a tiled Conv-BN-SiLU layer.
//
// This is the reference pattern every YOLO26n conv layer should follow. It
// honours scale_pkg::P_COUT_<i> and scale_pkg::P_CIN_<i> by instantiating
// exactly P_COUT dotN cells of width (K*K*P_CIN) and time-multiplexing over
// N_COUT_TILE × N_CIN_TILE phases per output pixel.
//
// Replace LAYER_TPL_* with LAYER_<i>_* (scale_pkg constants) when retrofitting
// a real layer. Search-and-replace the module name and the constants — the
// surrounding logic is parameter-driven and should not need structural edits.
//
//
// Architecture
// ────────────
//   * P_COUT parallel dotN cells, each N=K*K*P_CIN lanes wide.
//   * Per-pixel iteration:
//
//        for cout_tile in 0 .. N_COUT_TILE-1:        (outer)
//          acc[0..P_COUT-1] = 0
//          for cin_tile in 0 .. N_CIN_TILE-1:        (inner)
//            acc += dotN(x_tile, w_tile[cout_tile][cin_tile])
//          y_tile = silu(requant(acc, scale[cout_tile], bias[cout_tile]))
//          push y_tile (alongside cout_tile_idx) downstream
//
//   * Cycles/pixel = N_COUT_TILE × N_CIN_TILE (steady-state throughput).
//   * Frame cycles = H_out × W_out × N_COUT_TILE × N_CIN_TILE — this matches
//     LAYER_<i>_CYCLES from scale_pkg.
//
//
// Driver protocol
// ───────────────
//   The driver advances cin_tile innermost, cout_tile outermost. For each
//   beat it presents:
//     - x_i      : K*K*P_CIN i8 lanes for the current cin_tile
//                  (same patch is replayed P_CIN_TILE times per cout_tile)
//     - w_i      : P_COUT * (K*K*P_CIN) i8 weights for
//                  (cout_tile, cin_tile)
//     - scale_i, bias_i : P_COUT fp16 values for cout_tile
//     - cin_tile_idx_i  : 0..N_CIN_TILE-1
//     - cout_tile_idx_i : 0..N_COUT_TILE-1
//     - first_cin_i     : 1 when cin_tile_idx == 0 (zeros the accumulator)
//     - last_cin_i      : 1 when cin_tile_idx == N_CIN_TILE-1 (commits to
//                         requant+silu DOT_LAT cycles later)
//     - valid_i         : 1
//
//
// Output protocol
// ───────────────
//   On valid_o the layer emits a P_COUT-wide i8 vector y_o together with
//   cout_tile_idx_o telling the consumer which slice of the full COUT
//   activation this is. The testbench / next layer reconstructs the full
//   COUT-wide output by stitching N_COUT_TILE tiles per pixel.

module layer_template
  import scale_pkg::*;
#(
  // --- Replace with scale_pkg::LAYER_<i>_* parameters in real layers ---
  parameter int  COUT     = 128,
  parameter int  CIN      = 128,
  parameter int  K        = 3,
  parameter int  P_COUT   = 16,
  parameter int  P_CIN    = 8,
  // SiLU LUT scales (pre / post). Same convention as legacy layers.
  parameter real S_OUT_PRE  = 4.0 / 127.0,
  parameter real S_OUT_SILU = 4.0 / 127.0
) (
  input  logic                                            clk_i,
  input  logic                                            rst_ni,

  // Tile control
  input  logic                                            valid_i,
  input  logic                                            first_cin_i,  // cin_tile==0
  input  logic                                            last_cin_i,   // cin_tile==N_CIN_TILE-1
  // verilator lint_off UNUSEDSIGNAL
  input  logic [$clog2((COUT+P_COUT-1)/P_COUT == 0 ? 1 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_i,
  // verilator lint_on UNUSEDSIGNAL

  // K*K*P_CIN i8 lanes for the current cin_tile of the current patch
  input  logic signed [K*K*P_CIN-1:0][7:0]                x_i,
  // P_COUT × (K*K*P_CIN) i8 weights for (cout_tile, cin_tile)
  input  logic signed [P_COUT-1:0][K*K*P_CIN-1:0][7:0]    w_i,
  // P_COUT fp16 scale/bias for the current cout_tile
  input  logic        [P_COUT-1:0][15:0]                  scale_i,
  input  logic        [P_COUT-1:0][15:0]                  bias_i,

  output logic                                            valid_o,
  output logic [$clog2((COUT+P_COUT-1)/P_COUT == 0 ? 1 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_o,
  output logic signed [P_COUT-1:0][7:0]                   y_o
);

  // ─── Derived constants ───────────────────────────────
  localparam int N_LANE        = K * K * P_CIN;
  localparam int N_COUT_TILE   = (COUT + P_COUT - 1) / P_COUT;
  localparam int N_CIN_TILE    = (CIN  + P_CIN  - 1) / P_CIN;
  localparam int CT_W          = (N_COUT_TILE <= 1) ? 1 : $clog2(N_COUT_TILE);
  localparam int DOT_LAT       = 1 + $clog2(N_LANE);
  localparam int REQUANT_LAT   = 4;
  localparam int SILU_LAT      = 1;

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

  // ─── Pipeline control alongside dotN ───────────────────
  // first_cin / last_cin / cout_tile_idx / scale / bias arrive aligned with
  // the input window; they must arrive at the accumulator stage in step with
  // dot_acc, i.e. delayed DOT_LAT cycles.
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

  // ─── Stage B: i32 accumulator across cin tiles ─────────
  // One P_COUT-wide accumulator. On first_cin_d we start fresh; otherwise we
  // add the new dotN output. When last_cin_d fires we push the accumulated
  // value into requant. Because the driver iterates cin_tile innermost we
  // know consecutive dot_valid beats belong to the same (pixel, cout_tile).
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
      commit_ct_q    <= ct_sr[DOT_LAT-1];
      commit_scale_q <= scale_sr[DOT_LAT-1];
      commit_bias_q  <= bias_sr [DOT_LAT-1];
    end
  end

  // One pipeline stage between commit and requant to keep timing clean and
  // to give acc_q a cycle to settle.
  logic                           rq_valid_in;
  logic signed [P_COUT-1:0][31:0] rq_acc_in;
  logic [P_COUT-1:0][15:0]        rq_scale_in;
  logic [P_COUT-1:0][15:0]        rq_bias_in;
  logic [CT_W-1:0]                rq_ct_in;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rq_valid_in <= 1'b0; rq_acc_in <= '0;
      rq_scale_in <= '0;   rq_bias_in <= '0; rq_ct_in <= '0;
    end else begin
      rq_valid_in <= commit_q;
      rq_acc_in   <= acc_q;
      rq_scale_in <= commit_scale_q;
      rq_bias_in  <= commit_bias_q;
      rq_ct_in    <= commit_ct_q;
    end
  end

  // ─── Stage C: P_COUT parallel requant ──────────────────
  // requant has built-in autoscale (i32→fp16 → scale-bump → FMA → i8 sat)
  // so no ACC_SHIFT workaround is required at the layer level.
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

  // Pipeline cout_tile_idx alongside requant + silu.
  localparam int POST_LAT = REQUANT_LAT + SILU_LAT;
  logic [CT_W-1:0] ct_post [POST_LAT];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < POST_LAT; i++) ct_post[i] <= '0;
    end else begin
      ct_post[0] <= commit_ct_q;
      for (int i = 1; i < POST_LAT; i++) ct_post[i] <= ct_post[i-1];
    end
  end

  // ─── Stage D: P_COUT parallel SiLU ─────────────────────
  logic signed [P_COUT-1:0][7:0] silu_y;
  generate
    for (gc = 0; gc < P_COUT; gc++) begin : g_silu
      act_silu #(.InScale(S_OUT_PRE), .OutScale(S_OUT_SILU)) u_silu (
        .clk_i (clk_i),
        .rst_ni(rst_ni),
        .x_i   (rq_y[gc]),
        .y_o   (silu_y[gc])
      );
    end
  endgenerate

  logic silu_valid_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) silu_valid_q <= 1'b0;
    else         silu_valid_q <= rq_valid[0];
  end

  // verilator lint_off UNUSEDSIGNAL
  logic [P_COUT-2:0] unused_rq_valid;
  assign unused_rq_valid = rq_valid[P_COUT-1:1];
  // verilator lint_on UNUSEDSIGNAL

  assign valid_o         = silu_valid_q;
  assign cout_tile_idx_o = ct_post[POST_LAT-1];
  assign y_o             = silu_y;

endmodule
