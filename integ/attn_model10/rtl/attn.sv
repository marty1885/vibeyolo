// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// attn — YOLO26n /model.10 PSA attention block.
//
// Computes the attention sub-graph that sits between the qkv conv and the
// proj conv, returning ATTN_OUT (= attention(qkv) + pe), int8 @ S_AOUT:
//
//   QKV[256,20,20] int8 ─▶ dequant(×S_QKV) ─▶ reshape/split Q,K,V
//                          ─▶ flash_attn ─▶ O[2,400,64] fp16
//   O ─▶ reshape[128,20,20] ─▶ (+ pe·S_PE) ─▶ ×INV_S_AOUT ─▶ sat_i8 ─▶ ATTN_OUT
//
// The proj conv, the proj+spl1 residual, the ffn convs, and the proj_res+ffn1
// residual all happen EXTERNAL to this block (proj/ffn are conv_stage layers;
// the two residual adds are add_rq glue in the core). So this block's contract
// is purely: two int8 input streams (qkv 256-ch, pe 128-ch) → one int8 output
// stream (attn_out 128-ch), all raster (token = h*W+w), channel-parallel.
//
// Built from leaf IPs: i32_to_fp16, fp16_fma, flash_attn, fp16_to_i8_sat.
// Per-tensor scales (S_QKV, S_PE, INV_S_AOUT) are pinned at elaboration via
// attn_scales_pkg (the chip's calibrated constants).

module attn #(
  parameter int H        = 20,
  parameter int W        = 20,
  parameter int C_QKV    = 256,
  parameter int C_FE     = 128,
  parameter int HEADS    = 2,
  parameter int DIM_Q    = 32,
  parameter int DIM_K    = 32,
  parameter int DIM_V    = 64,
  parameter int BR       = 16,
  parameter int BC       = 32,
  parameter logic [15:0] TEMP_FP16 = 16'h31A8,  // 1/sqrt(32) in fp16
  // flash_attn wide accumulator (see flash_attn.sv). Sweepable for accuracy.
  parameter int unsigned ACC_EXP  = 8,
  parameter int unsigned ACC_MANT = 21
) (
  input  logic                              clk_i,
  input  logic                              rst_ni,
  input  logic                              start_i,

  input  logic                              qkv_valid_i,
  output logic                              qkv_ready_o,
  input  logic signed [C_QKV-1:0][7:0]      qkv_data_i,

  input  logic                              pe_valid_i,
  output logic                              pe_ready_o,
  input  logic signed [C_FE-1:0][7:0]       pe_data_i,

  output logic                              ovalid_o,
  input  logic                              oready_i,
  output logic signed [C_FE-1:0][7:0]       odata_o,

  output logic                              done_o
);

  import attn_scales_pkg::*;

  localparam int N      = H * W;                 // 400
  localparam int CntW   = (N <= 1) ? 1 : $clog2(N) + 1;
  // Pipeline depths. The leaf IPs were pipelined for timing — keep these in
  // sync with fp16_lat_pkg (i32_to_fp16=2, fp16_fma=5); the attn DV (which
  // exercises the real leaf RTL end-to-end) fails if they drift.
  localparam int I2F_LAT = 2;                    // == fp16_lat_pkg::I32_TO_FP16_LAT
  localparam int FMA_LAT = 5;                    // == fp16_lat_pkg::FP16_FMA_LAT
  // QKV deq chain: i32_to_fp16 → fp16_fma(×S_QKV)
  localparam int QKV_LAT = I2F_LAT + FMA_LAT;          // 7
  // ATTN_OUT chain: i2f → fma(×S_PE) → fma(O+pe) → fma(×INV_S_AOUT) → sat_i8
  localparam int AO_LAT  = I2F_LAT + 3*FMA_LAT + 1;    // 18
  localparam int DRIVE_TAIL_Q = QKV_LAT;
  localparam int DRIVE_TAIL_A = AO_LAT;

  // ── Frame stores (int8) ──────────────────────────────────────
  logic signed [C_QKV-1:0][7:0] qkv_mem [N];
  logic signed [C_FE-1:0][7:0]  pe_mem  [N];

  // Output frame store (filled by the attn-out chain).
  logic signed [C_FE-1:0][7:0]  out_mem [N];

  // ── FSM ──────────────────────────────────────────────────────
  typedef enum logic [3:0] {
    S_IDLE,
    S_LOAD_QKV,
    S_LOAD_PE,
    S_QKV_DEQ,
    S_FA_PULSE,
    S_FA_WAIT,
    S_AO,
    S_DRAIN,
    S_DONE
  } state_e;
  state_e state_q, state_d;

  logic [CntW-1:0] qkv_cnt_q, qkv_cnt_d;
  logic [CntW-1:0] pe_cnt_q,  pe_cnt_d;
  logic [CntW-1:0] o_cnt_q,   o_cnt_d;

  // Phase counters for the two compute pipelines.
  logic [CntW:0]   qd_cyc_q, qd_cyc_d;
  logic [CntW:0]   ao_cyc_q, ao_cyc_d;

  assign qkv_ready_o  = (state_q == S_LOAD_QKV);
  assign pe_ready_o   = (state_q == S_LOAD_PE);
  assign ovalid_o     = (state_q == S_DRAIN);
  assign done_o       = (state_q == S_DONE);

  logic qkv_fire, pe_fire, o_fire;
  assign qkv_fire = qkv_valid_i && qkv_ready_o;
  assign pe_fire  = pe_valid_i  && pe_ready_o;
  assign o_fire   = ovalid_o    && oready_i;

  // flash_attn handshake
  logic fa_start;
  logic fa_done;

  always_comb begin
    state_d   = state_q;
    qkv_cnt_d = qkv_cnt_q;
    pe_cnt_d  = pe_cnt_q;
    o_cnt_d   = o_cnt_q;
    qd_cyc_d  = qd_cyc_q;
    ao_cyc_d  = ao_cyc_q;
    fa_start  = 1'b0;

    unique case (state_q)
      S_IDLE: begin
        if (start_i) begin
          state_d   = S_LOAD_QKV;
          qkv_cnt_d = '0; pe_cnt_d = '0; o_cnt_d = '0;
          qd_cyc_d  = '0; ao_cyc_d = '0;
        end
      end
      S_LOAD_QKV: begin
        if (qkv_fire) begin
          qkv_cnt_d = qkv_cnt_q + 1'b1;
          if (qkv_cnt_q + 1'b1 == CntW'(N)) state_d = S_LOAD_PE;
        end
      end
      S_LOAD_PE: begin
        if (pe_fire) begin
          pe_cnt_d = pe_cnt_q + 1'b1;
          if (pe_cnt_q + 1'b1 == CntW'(N)) begin
            state_d  = S_QKV_DEQ;
            qd_cyc_d = '0;
          end
        end
      end
      S_QKV_DEQ: begin
        qd_cyc_d = qd_cyc_q + 1'b1;
        if (qd_cyc_q == (CntW+1)'(N + DRIVE_TAIL_Q - 1)) state_d = S_FA_PULSE;
      end
      S_FA_PULSE: begin
        fa_start = 1'b1;
        state_d  = S_FA_WAIT;
      end
      S_FA_WAIT: begin
        if (fa_done) begin
          state_d  = S_AO;
          ao_cyc_d = '0;
        end
      end
      S_AO: begin
        ao_cyc_d = ao_cyc_q + 1'b1;
        if (ao_cyc_q == (CntW+1)'(N + DRIVE_TAIL_A - 1)) state_d = S_DRAIN;
      end
      S_DRAIN: begin
        if (o_fire) begin
          o_cnt_d = o_cnt_q + 1'b1;
          if (o_cnt_q + 1'b1 == CntW'(N)) state_d = S_DONE;
        end
      end
      S_DONE: begin
        if (start_i) begin
          state_d   = S_LOAD_QKV;
          qkv_cnt_d = '0; pe_cnt_d = '0; o_cnt_d = '0;
          qd_cyc_d  = '0; ao_cyc_d = '0;
        end
      end
      default: state_d = S_IDLE;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q   <= S_IDLE;
      qkv_cnt_q <= '0; pe_cnt_q <= '0; o_cnt_q <= '0;
      qd_cyc_q  <= '0; ao_cyc_q <= '0;
    end else begin
      state_q   <= state_d;
      qkv_cnt_q <= qkv_cnt_d;
      pe_cnt_q  <= pe_cnt_d;
      o_cnt_q   <= o_cnt_d;
      qd_cyc_q  <= qd_cyc_d;
      ao_cyc_q  <= ao_cyc_d;
    end
  end

  // ── Frame-store writes ───────────────────────────────────────
  always_ff @(posedge clk_i) begin
    if (qkv_fire) qkv_mem[qkv_cnt_q[CntW-2:0]] <= qkv_data_i;
    if (pe_fire)  pe_mem [pe_cnt_q [CntW-2:0]] <= pe_data_i;
  end

  // ─── QKV dequant pipeline ───────────────────────────────────
  // At qd_cyc_q == k drive qkv_mem[k] (if k<N) through 256 parallel
  // i32_to_fp16 → fp16_fma(×S_QKV) chains; capture QKV_LAT later.
  logic [CntW-1:0] qd_drive_idx;
  logic            qd_drive_valid;
  logic [CntW-1:0] qd_cap_idx;
  logic            qd_cap_valid;
  assign qd_drive_valid = (state_q == S_QKV_DEQ) && (qd_cyc_q < (CntW+1)'(N));
  assign qd_drive_idx   = qd_drive_valid ? qd_cyc_q[CntW-1:0] : '0;
  assign qd_cap_valid   = (state_q == S_QKV_DEQ) && (qd_cyc_q >= (CntW+1)'(QKV_LAT));
  assign qd_cap_idx     = qd_cap_valid ? (qd_cyc_q[CntW-1:0] - CntW'(QKV_LAT)) : '0;

  logic signed [C_QKV-1:0][7:0] qkv_drive_pix;
  assign qkv_drive_pix = qkv_mem[qd_drive_idx];

  logic [15:0] qkv_fp [C_QKV];   // 2-stage pipeline output

  for (genvar c = 0; c < C_QKV; c++) begin : g_qkv_deq
    logic signed [31:0] x32;
    logic [15:0]        s0;
    logic [4:0]         sh_u;
    assign x32 = 32'($signed(qkv_drive_pix[c]));
    i32_to_fp16 u_i2f (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .x_i    (x32),
      .y_o    (s0),
      .shift_o(sh_u)
    );
    fp16_fma u_fma (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .a_i    (s0),
      .b_i    (S_QKV_FP16),
      .c_i    (16'h0000),
      .y_o    (qkv_fp[c])
    );
    logic _u_sh; assign _u_sh = ^sh_u;
    logic _u_sh_keep; always_ff @(posedge clk_i) _u_sh_keep <= _u_sh;
  end

  // Capture into Q/K/V flat fp16 buffers (indexed [h][n][d]).
  logic [15:0] Q_arr [HEADS*N*DIM_Q];
  logic [15:0] K_arr [HEADS*N*DIM_Q];
  logic [15:0] V_arr [HEADS*N*DIM_V];

  always_ff @(posedge clk_i) begin
    if (qd_cap_valid) begin
      for (int h = 0; h < HEADS; h++) begin
        // NB: inline (no `int base_ch = h*...`) — a procedural var with an
        // initializer is static, so its init runs once with h=0 and head 1
        // would silently read head 0's channels.
        for (int d = 0; d < DIM_Q; d++)
          Q_arr[h*N*DIM_Q + qd_cap_idx*DIM_Q + d] <= qkv_fp[h*(DIM_Q+DIM_K+DIM_V) + d];
        for (int d = 0; d < DIM_K; d++)
          K_arr[h*N*DIM_Q + qd_cap_idx*DIM_Q + d] <= qkv_fp[h*(DIM_Q+DIM_K+DIM_V) + DIM_Q + d];
        for (int d = 0; d < DIM_V; d++)
          V_arr[h*N*DIM_V + qd_cap_idx*DIM_V + d] <= qkv_fp[h*(DIM_Q+DIM_K+DIM_V) + DIM_Q + DIM_K + d];
      end
    end
  end

  // ─── flash_attn instance ───────────────────────────────────
  logic [HEADS*N*DIM_Q*16-1:0] q_flat;
  logic [HEADS*N*DIM_Q*16-1:0] k_flat;
  logic [HEADS*N*DIM_V*16-1:0] v_flat;
  logic [HEADS*N*DIM_V*16-1:0] o_flat;

  always_comb begin
    for (int i = 0; i < HEADS*N*DIM_Q; i++) begin
      q_flat[16*i +: 16] = Q_arr[i];
      k_flat[16*i +: 16] = K_arr[i];
    end
    for (int i = 0; i < HEADS*N*DIM_V; i++)
      v_flat[16*i +: 16] = V_arr[i];
  end

  flash_attn #(
    .HEADS       (HEADS),
    .N           (N),
    .DIM_Q       (DIM_Q),
    .DIM_V       (DIM_V),
    .BR          (BR),
    .BC          (BC),
    .TEMP_FP16   (TEMP_FP16),
    .ACC_EXP     (ACC_EXP),
    .ACC_MANT    (ACC_MANT),
    .MAX_CYC_HINT(100000)
  ) u_fa (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .start_i (fa_start),
    .done_o  (fa_done),
    .q_flat_i(q_flat),
    .k_flat_i(k_flat),
    .v_flat_i(v_flat),
    .o_flat_o(o_flat)
  );

  // ─── ATTN_OUT chain ─────────────────────────────────────────
  // Per-channel pipeline (C_FE=128 parallel) over N tokens:
  //   drive : pe_pix[c] int8, O[c] fp16 (= flash_attn O for this token/chan)
  //   r1    : pe_s0 = i32_to_fp16(pe_pix[c])              ; o_d1 <= O[c]
  //   r2    : pe_fp = fp16_fma(pe_s0, S_PE, 0)            ; o_d2 <= o_d1
  //   r3    : sum   = fp16_fma(o_d2, 1.0, pe_fp)   (= O + pe)
  //   r4    : y_fp  = fp16_fma(sum, INV_S_AOUT, 0)
  //   r5    : i8    = fp16_to_i8_sat(y_fp)
  // AO_LAT = I2F_LAT + 3*FMA_LAT + 1 = 18 from driven token to captured
  // int8 (each fma is FMA_LAT=5 cycles, i2f is I2F_LAT=2, sat is 1).
  //
  // O channel layout: ATTN_OUT channel c == head (c/DIM_V), value-dim
  // (c%DIM_V). flash_attn packs O as o_flat[16*(h*N*DIM_V + n*DIM_V + d)].
  localparam logic [15:0] FP16_ONE  = 16'h3C00;
  localparam logic [15:0] FP16_ZERO = 16'h0000;

  logic [CntW-1:0] ao_drive_idx;
  logic            ao_drive_valid;
  logic [CntW-1:0] ao_cap_idx;
  logic            ao_cap_valid;
  assign ao_drive_valid = (state_q == S_AO) && (ao_cyc_q < (CntW+1)'(N));
  assign ao_drive_idx   = ao_drive_valid ? ao_cyc_q[CntW-1:0] : '0;
  assign ao_cap_valid   = (state_q == S_AO) && (ao_cyc_q >= (CntW+1)'(AO_LAT))
                                            && (ao_cyc_q <  (CntW+1)'(N + AO_LAT));
  assign ao_cap_idx     = ao_cap_valid ? (ao_cyc_q[CntW-1:0] - CntW'(AO_LAT)) : '0;

  logic signed [C_FE-1:0][7:0] pe_drive_pix;
  assign pe_drive_pix = pe_mem[ao_drive_idx];

  // Unpack flash_attn O into a per-token array O_arr[n][c], channel
  // c == head*(DIM_V) + value-dim. Static loops → constant part-selects of
  // o_flat (no runtime base), then a clean per-token array read below.
  logic [15:0] O_arr [N][C_FE];
  always_comb begin
    for (int n = 0; n < N; n++)
      for (int h = 0; h < HEADS; h++)
        for (int d = 0; d < DIM_V; d++)
          O_arr[n][h*DIM_V + d] = o_flat[16*(h*N*DIM_V + n*DIM_V + d) +: 16];
  end

  logic signed [C_FE-1:0][7:0] attn_i8;

  for (genvar c = 0; c < C_FE; c++) begin : g_ao
    // O for the driven token, this channel.
    logic [15:0] o_val;
    assign o_val = O_arr[ao_drive_idx][c];

    // pe dequant: int8 → fp16 → ×S_PE
    logic signed [31:0] pe_x32;
    logic [15:0]        pe_s0;
    logic [4:0]         pe_sh;
    assign pe_x32 = 32'($signed(pe_drive_pix[c]));
    i32_to_fp16 u_pi2f (.clk_i, .rst_ni, .x_i(pe_x32), .y_o(pe_s0), .shift_o(pe_sh));

    logic [15:0] pe_fp;
    fp16_fma u_pe (.clk_i, .rst_ni, .a_i(pe_s0), .b_i(S_PE_FP16), .c_i(FP16_ZERO), .y_o(pe_fp));

    // Delay O to align with pe_fp at the u_sum input. pe_fp is produced
    // I2F_LAT + FMA_LAT cycles after the token is driven (i2f then ×S_PE
    // fma), so O must be delayed by the same amount.
    localparam int O_ALIGN = I2F_LAT + FMA_LAT;
    logic [15:0] o_dl [O_ALIGN];
    always_ff @(posedge clk_i) begin
      o_dl[0] <= o_val;
      for (int i = 1; i < O_ALIGN; i++) o_dl[i] <= o_dl[i-1];
    end

    logic [15:0] sum, y_fp;
    fp16_fma u_sum (.clk_i, .rst_ni, .a_i(o_dl[O_ALIGN-1]), .b_i(FP16_ONE), .c_i(pe_fp), .y_o(sum));
    fp16_fma u_y   (.clk_i, .rst_ni, .a_i(sum),  .b_i(INV_S_AOUT_FP16), .c_i(FP16_ZERO), .y_o(y_fp));

    logic signed [7:0] yi8;
    fp16_to_i8_sat u_q (.clk_i, .rst_ni, .x_i(y_fp), .y_o(yi8));
    assign attn_i8[c] = yi8;

    logic _u; assign _u = ^pe_sh;
    logic _u_keep; always_ff @(posedge clk_i) _u_keep <= _u;
  end

  always_ff @(posedge clk_i) begin
    if (ao_cap_valid) out_mem[ao_cap_idx] <= attn_i8;
  end

  // ─── Drain ──────────────────────────────────────────────────
  always_comb begin
    odata_o = out_mem[o_cnt_q[CntW-2:0]];
  end

  // Lint keep-alive for top counter bits.
  logic _unused;
  assign _unused = ^{qkv_cnt_q[CntW-1], pe_cnt_q[CntW-1], o_cnt_q[CntW-1],
                     qd_cyc_q[CntW], ao_cyc_q[CntW]};

endmodule
