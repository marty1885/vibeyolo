// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// conv_layer_ref — behavioral golden for conv_layer.
//
// Same I/O contract as conv_layer.sv. Implementation differences vs the
// DUT (intentional, so this is an independent oracle):
//
//   • The dotN tree is replaced with an unrolled combinational sum done
//     in longint (mathematically equal to the i32 tree, but coded
//     differently). The output is registered with the same DOT_LAT.
//   • The accumulator / requant / silu / add_rq sub-blocks are the same
//     trusted IPs as the DUT. Re-instantiating them keeps the i32→i8
//     quantisation bit-exact with the DUT.
//   • The cout_tile_idx and valid pipelining is rewritten as a single
//     shift register array (DUT uses split SRs).
//
// As a result DUT and REF should match bit-exactly. The DV uses this for
// a cheap lockstep check; cosine vs an independent C++ shadow proves the
// algorithm itself is correct.

// verilator lint_off UNUSEDPARAM
// verilator lint_off UNUSEDSIGNAL
module conv_layer_ref
#(
  parameter int  CIN          = 128,
  parameter int  COUT         = 128,
  parameter int  K            = 3,
  parameter int  STRIDE       = 1,
  parameter int  PAD          = 1,
  parameter int  H_OUT        = 40,
  parameter int  W_OUT        = 40,
  parameter int  P_COUT       = 16,
  parameter int  P_CIN        = 8,
  parameter int  RESIDUAL     = 0,
  parameter int  SILU         = 1,
  parameter real S_OUT_PRE    = 4.0 / 127.0,
  parameter real S_OUT_SILU   = 4.0 / 127.0
) (
  input  logic                                            clk_i,
  input  logic                                            rst_ni,

  input  logic                                            valid_i,
  output logic                                            ready_o,

  input  logic                                            first_cin_i,
  input  logic                                            last_cin_i,
  input  logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_i,

  input  logic signed [K*K*P_CIN-1:0][7:0]                x_i,
  input  logic signed [P_COUT-1:0][K*K*P_CIN-1:0][7:0]    w_i,
  input  logic        [P_COUT-1:0][15:0]                  scale_i,
  input  logic        [P_COUT-1:0][15:0]                  bias_i,

  input  logic signed [P_COUT-1:0][7:0]                   r_i,
  input  logic        [P_COUT-1:0][15:0]                  r_scale_i,
  input  logic        [P_COUT-1:0][15:0]                  r_bias_i,

  output logic                                            valid_o,
  input  logic                                            ready_i,
  output logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_o,
  output logic signed [P_COUT-1:0][7:0]                   y_o
);

  localparam int N_LANE      = K * K * P_CIN;
  localparam int N_COUT_TILE = (COUT + P_COUT - 1) / P_COUT;
  localparam int CT_W        = $clog2((N_COUT_TILE < 2) ? 2 : N_COUT_TILE);
  localparam int DOT_LAT     = 1 + $clog2(N_LANE);
  localparam int REQUANT_LAT = 4;
  localparam int SILU_LAT    = 1;
  localparam int ADDRQ_LAT   = 5;

  assign ready_o = 1'b1;
  wire _u_ready = ready_i;

  // ─── Combinational dot product, registered (DOT_LAT-1 pad) ─────
  // First we compute the sum as a single longint, register it once. To
  // match the tree's DOT_LAT we then push the i32 through DOT_LAT-1 more
  // pipeline stages alongside the control signals.
  logic signed [P_COUT-1:0][31:0] dot_comb;
  always_comb begin
    for (int c = 0; c < P_COUT; c++) begin
      longint signed acc;
      acc = 0;
      for (int n = 0; n < N_LANE; n++) begin
        acc += longint'(signed'(x_i[n])) * longint'(signed'(w_i[c][n]));
      end
      dot_comb[c] = acc[31:0];
    end
  end

  // ─── Pipeline SR for dot + control to match DOT_LAT ────────────
  logic signed [P_COUT-1:0][31:0] dot_sr [DOT_LAT];
  logic                           dv_sr  [DOT_LAT];
  logic                           fst_sr [DOT_LAT];
  logic                           lst_sr [DOT_LAT];
  logic [CT_W-1:0]                ct_sr  [DOT_LAT];
  logic [P_COUT-1:0][15:0]        sc_sr  [DOT_LAT];
  logic [P_COUT-1:0][15:0]        bi_sr  [DOT_LAT];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < DOT_LAT; i++) begin
        dot_sr[i] <= '0;
        dv_sr [i] <= 1'b0;
        fst_sr[i] <= 1'b0;
        lst_sr[i] <= 1'b0;
        ct_sr [i] <= '0;
        sc_sr [i] <= '0;
        bi_sr [i] <= '0;
      end
    end else begin
      dot_sr[0] <= dot_comb;
      dv_sr [0] <= valid_i;
      fst_sr[0] <= first_cin_i & valid_i;
      lst_sr[0] <= last_cin_i  & valid_i;
      ct_sr [0] <= cout_tile_idx_i;
      sc_sr [0] <= scale_i;
      bi_sr [0] <= bias_i;
      for (int i = 1; i < DOT_LAT; i++) begin
        dot_sr[i] <= dot_sr[i-1];
        dv_sr [i] <= dv_sr [i-1];
        fst_sr[i] <= fst_sr[i-1];
        lst_sr[i] <= lst_sr[i-1];
        ct_sr [i] <= ct_sr [i-1];
        sc_sr [i] <= sc_sr [i-1];
        bi_sr [i] <= bi_sr [i-1];
      end
    end
  end

  // ─── Accumulator across cin tiles ──────────────────────────────
  logic signed [P_COUT-1:0][31:0] acc_q;
  logic                           commit_q;
  logic [CT_W-1:0]                commit_ct_q;
  logic [P_COUT-1:0][15:0]        commit_scale_q;
  logic [P_COUT-1:0][15:0]        commit_bias_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int c = 0; c < P_COUT; c++) acc_q[c] <= '0;
      commit_q       <= 1'b0;
      commit_ct_q    <= '0;
      commit_scale_q <= '0;
      commit_bias_q  <= '0;
    end else begin
      if (dv_sr[DOT_LAT-1]) begin
        for (int c = 0; c < P_COUT; c++) begin
          if (fst_sr[DOT_LAT-1]) acc_q[c] <= dot_sr[DOT_LAT-1][c];
          else                   acc_q[c] <= acc_q[c] + dot_sr[DOT_LAT-1][c];
        end
      end
      commit_q       <= dv_sr[DOT_LAT-1] & lst_sr[DOT_LAT-1];
      commit_ct_q    <= ct_sr[DOT_LAT-1];
      commit_scale_q <= sc_sr[DOT_LAT-1];
      commit_bias_q  <= bi_sr[DOT_LAT-1];
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

  logic [P_COUT-1:0]             rq_valid;
  logic signed [P_COUT-1:0][7:0] rq_y;

  genvar gc;
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

  // Same fp16 constants as DUT
  function automatic [15:0] real_to_fp16(input real v);
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
      biased_e = e + 15;
      if (biased_e <= 0)  return 16'h0000;
      if (biased_e >= 31) return {sign, 5'd30, 10'h3ff};
      mant = $rtoi((m - 1.0) * 1024.0 + 0.5);
      if (mant >= 1024) begin
        mant = 0; biased_e = biased_e + 1;
        if (biased_e >= 31) return {sign, 5'd30, 10'h3ff};
      end
      r = {sign, biased_e[4:0], mant[9:0]};
      return r;
    end
  endfunction

  localparam logic [15:0] FP16_S_OUT     = real_to_fp16(S_OUT_SILU);
  localparam logic [15:0] FP16_INV_S_OUT = real_to_fp16(1.0 / S_OUT_SILU);

  localparam int R_DELAY = DOT_LAT + 1 + 1 + REQUANT_LAT + ((SILU != 0) ? SILU_LAT : 0);

  generate
    if (RESIDUAL != 0) begin : g_res_en
      logic signed [P_COUT-1:0][7:0]  r_sr   [R_DELAY];
      logic        [P_COUT-1:0][15:0] rs_sr  [R_DELAY];
      logic        [P_COUT-1:0][15:0] rb_sr  [R_DELAY];
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
      wire _u_r  = |r_i;
      wire _u_rs = |r_scale_i;
      wire _u_rb = |r_bias_i;
      wire _u_c  = |FP16_S_OUT | |FP16_INV_S_OUT;
      // verilator lint_on UNUSEDSIGNAL
      assign y_o     = post_silu_y;
      assign valid_o = post_silu_valid;
    end
  endgenerate

  assign cout_tile_idx_o = ct_post[POST_LAT-1];

  // verilator lint_off UNUSEDSIGNAL
  wire [P_COUT-2:0] _u_rq_hi = rq_valid[P_COUT-1:1];
  // verilator lint_on UNUSEDSIGNAL

endmodule
// verilator lint_on UNUSEDPARAM
// verilator lint_on UNUSEDSIGNAL
