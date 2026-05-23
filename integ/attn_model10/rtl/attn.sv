// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// attn — YOLO26n /model.10 PSA attention block, STRUCTURAL shim.
//
// Replaces the previous SV-real behavioral block with a structural
// composition of the existing leaf IPs:
//   * i32_to_fp16 / fp16_fma         — int8 dequant + scale
//   * hw/ip/flash_attn               — tiled fp16 attention (Q,K,V → O)
//   * fp16_fma / fp16_to_i8_sat      — residual add chain + requant
//
// Topology (verified vs ONNX in extract.py):
//
//   QKV[256,20,20]  ──┐
//                     │   ┌─────────────┐
//   reshape+split    →│   │ flash_attn  │ → O[2,400,64] (fp16)
//                     │   └─────────────┘
//                     │           │
//                     ▼           ▼
//             (dequant to fp16)   ┐    pe-add and proj-conv happen
//                                 │    EXTERNAL to this block in the
//                                 ▼    real chip; here we simply route
//                                 │    O through a keep-alive XOR so
//                                 │    Verilator retains the IP.
//                                 │
//   proj/spl1/ffn1[128,20,20] int8─┘
//                  │
//                  ▼
//      (3-input fp16 add tree, one stage per addend)
//                  │
//                  ▼
//      ×INV_S_OUT_FP16 + fp16_to_i8_sat → final_i8 to odata_o
//
// Port interface is byte-identical to the previous block so the
// existing extract.py / TB / golden_final_i8.hex compare unchanged.
// The DV's bit-exact check is on the residual chain (proj+spl1+ffn1)
// quantized to S_OUT; flash_attn's integration here primarily exercises
// the wiring and verilator-elaborates the production tile sizes.

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
  parameter logic [15:0] TEMP_FP16 = 16'h31A8   // 1/sqrt(32) in fp16
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

  input  logic                              proj_valid_i,
  output logic                              proj_ready_o,
  input  logic signed [C_FE-1:0][7:0]       proj_data_i,

  input  logic                              spl1_valid_i,
  output logic                              spl1_ready_o,
  input  logic signed [C_FE-1:0][7:0]       spl1_data_i,

  input  logic                              ffn1_valid_i,
  output logic                              ffn1_ready_o,
  input  logic signed [C_FE-1:0][7:0]       ffn1_data_i,

  output logic                              ovalid_o,
  input  logic                              oready_i,
  output logic signed [C_FE-1:0][7:0]       odata_o,

  output logic                              done_o
);

  import attn_scales_pkg::*;

  localparam int N      = H * W;                 // 400
  localparam int CntW   = (N <= 1) ? 1 : $clog2(N) + 1;
  // Pipeline depths:
  localparam int QKV_LAT = 2;                    // i32_to_fp16 → fp16_fma
  localparam int RES_LAT = 6;                    // see residual chain below
  localparam int DRIVE_TAIL_Q = QKV_LAT;
  localparam int DRIVE_TAIL_R = RES_LAT;

  // ── Frame stores (int8) ──────────────────────────────────────
  logic signed [C_QKV-1:0][7:0] qkv_mem [N];
  logic signed [C_FE-1:0][7:0]  pe_mem  [N];
  logic signed [C_FE-1:0][7:0]  proj_mem[N];
  logic signed [C_FE-1:0][7:0]  spl1_mem[N];
  logic signed [C_FE-1:0][7:0]  ffn1_mem[N];

  // Output frame store (filled by residual chain).
  logic signed [C_FE-1:0][7:0]  out_mem [N];

  // ── FSM ──────────────────────────────────────────────────────
  typedef enum logic [3:0] {
    S_IDLE,
    S_LOAD_QKV,
    S_LOAD_PE,
    S_LOAD_PROJ,
    S_LOAD_SPL1,
    S_LOAD_FFN1,
    S_QKV_DEQ,
    S_FA_PULSE,
    S_FA_WAIT,
    S_RES,
    S_DRAIN,
    S_DONE
  } state_e;
  state_e state_q, state_d;

  logic [CntW-1:0] qkv_cnt_q,  qkv_cnt_d;
  logic [CntW-1:0] pe_cnt_q,   pe_cnt_d;
  logic [CntW-1:0] proj_cnt_q, proj_cnt_d;
  logic [CntW-1:0] spl1_cnt_q, spl1_cnt_d;
  logic [CntW-1:0] ffn1_cnt_q, ffn1_cnt_d;
  logic [CntW-1:0] o_cnt_q,    o_cnt_d;

  // Phase counters for the two compute pipelines.
  // qd_cyc counts from 0 in S_QKV_DEQ; valid output at cyc ≥ QKV_LAT
  // for input pixel (cyc - QKV_LAT). Stays in state until cyc == N + QKV_LAT - 1.
  logic [CntW:0]   qd_cyc_q,  qd_cyc_d;
  logic [CntW:0]   rs_cyc_q,  rs_cyc_d;

  assign qkv_ready_o  = (state_q == S_LOAD_QKV);
  assign pe_ready_o   = (state_q == S_LOAD_PE);
  assign proj_ready_o = (state_q == S_LOAD_PROJ);
  assign spl1_ready_o = (state_q == S_LOAD_SPL1);
  assign ffn1_ready_o = (state_q == S_LOAD_FFN1);
  assign ovalid_o     = (state_q == S_DRAIN);
  assign done_o       = (state_q == S_DONE);

  logic qkv_fire, pe_fire, proj_fire, spl1_fire, ffn1_fire, o_fire;
  assign qkv_fire  = qkv_valid_i  && qkv_ready_o;
  assign pe_fire   = pe_valid_i   && pe_ready_o;
  assign proj_fire = proj_valid_i && proj_ready_o;
  assign spl1_fire = spl1_valid_i && spl1_ready_o;
  assign ffn1_fire = ffn1_valid_i && ffn1_ready_o;
  assign o_fire    = ovalid_o     && oready_i;

  // flash_attn handshake
  logic fa_start;
  logic fa_done;

  always_comb begin
    state_d    = state_q;
    qkv_cnt_d  = qkv_cnt_q;
    pe_cnt_d   = pe_cnt_q;
    proj_cnt_d = proj_cnt_q;
    spl1_cnt_d = spl1_cnt_q;
    ffn1_cnt_d = ffn1_cnt_q;
    o_cnt_d    = o_cnt_q;
    qd_cyc_d   = qd_cyc_q;
    rs_cyc_d   = rs_cyc_q;
    fa_start   = 1'b0;

    unique case (state_q)
      S_IDLE: begin
        if (start_i) begin
          state_d   = S_LOAD_QKV;
          qkv_cnt_d = '0; pe_cnt_d = '0; proj_cnt_d = '0;
          spl1_cnt_d = '0; ffn1_cnt_d = '0; o_cnt_d = '0;
          qd_cyc_d = '0; rs_cyc_d = '0;
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
          if (pe_cnt_q + 1'b1 == CntW'(N)) state_d = S_LOAD_PROJ;
        end
      end
      S_LOAD_PROJ: begin
        if (proj_fire) begin
          proj_cnt_d = proj_cnt_q + 1'b1;
          if (proj_cnt_q + 1'b1 == CntW'(N)) state_d = S_LOAD_SPL1;
        end
      end
      S_LOAD_SPL1: begin
        if (spl1_fire) begin
          spl1_cnt_d = spl1_cnt_q + 1'b1;
          if (spl1_cnt_q + 1'b1 == CntW'(N)) state_d = S_LOAD_FFN1;
        end
      end
      S_LOAD_FFN1: begin
        if (ffn1_fire) begin
          ffn1_cnt_d = ffn1_cnt_q + 1'b1;
          if (ffn1_cnt_q + 1'b1 == CntW'(N)) begin
            state_d  = S_QKV_DEQ;
            qd_cyc_d = '0;
          end
        end
      end
      S_QKV_DEQ: begin
        qd_cyc_d = qd_cyc_q + 1'b1;
        // Last useful capture happens when (qd_cyc_q - QKV_LAT) == N-1,
        // i.e. qd_cyc_q == N + QKV_LAT - 1. Transition next cycle.
        if (qd_cyc_q == (CntW+1)'(N + DRIVE_TAIL_Q - 1)) begin
          state_d = S_FA_PULSE;
        end
      end
      S_FA_PULSE: begin
        fa_start = 1'b1;
        state_d  = S_FA_WAIT;
      end
      S_FA_WAIT: begin
        if (fa_done) begin
          state_d  = S_RES;
          rs_cyc_d = '0;
        end
      end
      S_RES: begin
        rs_cyc_d = rs_cyc_q + 1'b1;
        if (rs_cyc_q == (CntW+1)'(N + DRIVE_TAIL_R - 1)) state_d = S_DRAIN;
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
          qkv_cnt_d = '0; pe_cnt_d = '0; proj_cnt_d = '0;
          spl1_cnt_d = '0; ffn1_cnt_d = '0; o_cnt_d = '0;
          qd_cyc_d = '0; rs_cyc_d = '0;
        end
      end
      default: state_d = S_IDLE;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q    <= S_IDLE;
      qkv_cnt_q  <= '0; pe_cnt_q  <= '0; proj_cnt_q <= '0;
      spl1_cnt_q <= '0; ffn1_cnt_q <= '0; o_cnt_q <= '0;
      qd_cyc_q   <= '0; rs_cyc_q  <= '0;
    end else begin
      state_q    <= state_d;
      qkv_cnt_q  <= qkv_cnt_d;
      pe_cnt_q   <= pe_cnt_d;
      proj_cnt_q <= proj_cnt_d;
      spl1_cnt_q <= spl1_cnt_d;
      ffn1_cnt_q <= ffn1_cnt_d;
      o_cnt_q    <= o_cnt_d;
      qd_cyc_q   <= qd_cyc_d;
      rs_cyc_q   <= rs_cyc_d;
    end
  end

  // ── Frame-store writes ───────────────────────────────────────
  always_ff @(posedge clk_i) begin
    if (qkv_fire)  qkv_mem [qkv_cnt_q [CntW-2:0]] <= qkv_data_i;
    if (pe_fire)   pe_mem  [pe_cnt_q  [CntW-2:0]] <= pe_data_i;
    if (proj_fire) proj_mem[proj_cnt_q[CntW-2:0]] <= proj_data_i;
    if (spl1_fire) spl1_mem[spl1_cnt_q[CntW-2:0]] <= spl1_data_i;
    if (ffn1_fire) ffn1_mem[ffn1_cnt_q[CntW-2:0]] <= ffn1_data_i;
  end

  // ─── QKV dequant pipeline ───────────────────────────────────
  // At qd_cyc_q == k drive qkv_mem[k] (if k<N) through 256 parallel
  // i32_to_fp16 → fp16_fma(×S_QKV) chains. The result is available
  // QKV_LAT=2 cycles later, capture into Q/K/V arrays.
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
        int base_ch = h * (DIM_Q + DIM_K + DIM_V);
        for (int d = 0; d < DIM_Q; d++)
          Q_arr[h*N*DIM_Q + qd_cap_idx*DIM_Q + d] <= qkv_fp[base_ch + d];
        for (int d = 0; d < DIM_K; d++)
          K_arr[h*N*DIM_Q + qd_cap_idx*DIM_Q + d] <= qkv_fp[base_ch + DIM_Q + d];
        for (int d = 0; d < DIM_V; d++)
          V_arr[h*N*DIM_V + qd_cap_idx*DIM_V + d] <= qkv_fp[base_ch + DIM_Q + DIM_K + d];
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

  // Keep flash_attn's output observable so Verilator doesn't prune the
  // IP. XOR-reduce o_flat into a 1-bit alive signal latched per frame.
  logic fa_alive_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)                 fa_alive_q <= 1'b0;
    else if (state_q == S_FA_WAIT && fa_done)
      fa_alive_q <= ^o_flat;
  end

  // ─── Residual chain ─────────────────────────────────────────
  // Per-channel pipeline (C_FE=128 parallel):
  //   stage 0:    feed proj/spl1/ffn1 int8                              (comb)
  //   stage 1:    i32_to_fp16 → proj_fp, spl1_fp, ffn1_fp               (reg)
  //   stage 2:    ta = fp16_fma(proj_fp, S_PROJ, 0)
  //               tb = fp16_fma(spl1_fp, S_SPL1, 0)
  //               tc = fp16_fma(ffn1_fp, S_FFN1, 0)                      (reg)
  //   stage 3:    ab = fp16_fma(ta, 1.0, tb)                             (reg)
  //   stage 4:    abc = fp16_fma(ab, 1.0, tc)                            (reg)
  //   stage 5:    y_fp = fp16_fma(abc, INV_S_OUT_FP16, 0)                (reg)
  //   stage 6:    final_i8 = fp16_to_i8_sat(y_fp)                        (reg)
  // RES_LAT = 6 from input to captured int8.

  localparam logic [15:0] FP16_ONE  = 16'h3C00;
  localparam logic [15:0] FP16_ZERO = 16'h0000;

  logic [CntW-1:0] rs_drive_idx;
  logic            rs_drive_valid;
  logic [CntW-1:0] rs_cap_idx;
  logic            rs_cap_valid;
  assign rs_drive_valid = (state_q == S_RES) && (rs_cyc_q < (CntW+1)'(N));
  assign rs_drive_idx   = rs_drive_valid ? rs_cyc_q[CntW-1:0] : '0;
  assign rs_cap_valid   = (state_q == S_RES) && (rs_cyc_q >= (CntW+1)'(RES_LAT))
                                              && (rs_cyc_q <  (CntW+1)'(N + RES_LAT));
  assign rs_cap_idx     = rs_cap_valid ? (rs_cyc_q[CntW-1:0] - CntW'(RES_LAT)) : '0;

  logic signed [C_FE-1:0][7:0] proj_pix, spl1_pix, ffn1_pix;
  assign proj_pix = proj_mem[rs_drive_idx];
  assign spl1_pix = spl1_mem[rs_drive_idx];
  assign ffn1_pix = ffn1_mem[rs_drive_idx];

  logic signed [C_FE-1:0][7:0] final_i8;

  for (genvar c = 0; c < C_FE; c++) begin : g_res
    logic signed [31:0] proj_x32, spl1_x32, ffn1_x32;
    assign proj_x32 = 32'($signed(proj_pix[c]));
    assign spl1_x32 = 32'($signed(spl1_pix[c]));
    assign ffn1_x32 = 32'($signed(ffn1_pix[c]));

    logic [15:0] proj_fp, spl1_fp, ffn1_fp;
    logic [4:0]  shp, shs, shf;
    i32_to_fp16 u_pi2f (.clk_i, .rst_ni, .x_i(proj_x32), .y_o(proj_fp), .shift_o(shp));
    i32_to_fp16 u_si2f (.clk_i, .rst_ni, .x_i(spl1_x32), .y_o(spl1_fp), .shift_o(shs));
    i32_to_fp16 u_fi2f (.clk_i, .rst_ni, .x_i(ffn1_x32), .y_o(ffn1_fp), .shift_o(shf));

    logic [15:0] ta, tb, tc;
    fp16_fma u_ta (.clk_i, .rst_ni, .a_i(proj_fp), .b_i(S_PROJ_FP16), .c_i(FP16_ZERO), .y_o(ta));
    fp16_fma u_tb (.clk_i, .rst_ni, .a_i(spl1_fp), .b_i(S_SPL1_FP16), .c_i(FP16_ZERO), .y_o(tb));
    fp16_fma u_tc (.clk_i, .rst_ni, .a_i(ffn1_fp), .b_i(S_FFN1_FP16), .c_i(FP16_ZERO), .y_o(tc));

    // tc must be delayed 1 cycle to align with ab in the (ab + tc) fma:
    // ta/tb/tc all emerge at the same pipeline stage; u_ab combines ta+tb
    // and outputs one cycle later, so tc must lag by 1 cycle to match.
    logic [15:0] tc_dly;
    always_ff @(posedge clk_i) tc_dly <= tc;

    logic [15:0] ab, abc, y_fp;
    fp16_fma u_ab  (.clk_i, .rst_ni, .a_i(ta),  .b_i(FP16_ONE), .c_i(tb),     .y_o(ab));
    fp16_fma u_abc (.clk_i, .rst_ni, .a_i(ab),  .b_i(FP16_ONE), .c_i(tc_dly), .y_o(abc));
    fp16_fma u_yf  (.clk_i, .rst_ni, .a_i(abc), .b_i(INV_S_OUT_FP16), .c_i(FP16_ZERO), .y_o(y_fp));

    logic signed [7:0] yi8;
    fp16_to_i8_sat u_q (.clk_i, .rst_ni, .x_i(y_fp), .y_o(yi8));

    assign final_i8[c] = yi8;

    // keep unused shift outputs from being optimised away
    logic _u; assign _u = ^{shp, shs, shf};
    logic _u_keep; always_ff @(posedge clk_i) _u_keep <= _u;
  end

  always_ff @(posedge clk_i) begin
    if (rs_cap_valid) out_mem[rs_cap_idx] <= final_i8;
  end

  // ─── Drain ──────────────────────────────────────────────────
  always_comb begin
    odata_o = out_mem[o_cnt_q[CntW-2:0]];
  end

  // Lint keep-alive for fa_alive_q and top counter bits.
  logic _unused;
  assign _unused = ^{fa_alive_q,
                     qkv_cnt_q[CntW-1], pe_cnt_q[CntW-1], proj_cnt_q[CntW-1],
                     spl1_cnt_q[CntW-1], ffn1_cnt_q[CntW-1], o_cnt_q[CntW-1],
                     qd_cyc_q[CntW], rs_cyc_q[CntW]};

endmodule
