// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// conv_stage_tb — dual-instance bit-exact DV wrapper.
//
//   DUT : conv_stage  (raster frame in -> raster frame out, internal linebuf,
//                       weight/scale/bias ROMs, tile sequencer)
//   REF : conv_layer   (the already-ORT/ref-validated compute core), driven by
//                       the software tile schedule from the C++ testbench.
//
// Both share the identical conv_layer compute, so any output mismatch is a
// conv_stage sequencing/plumbing bug. Packed ports are flattened to 1-D
// vectors for the Verilator C++ interface (same idiom as the layer TBs).

module conv_stage_tb #(
  parameter int  CIN    = 8,
  parameter int  COUT   = 8,
  parameter int  K      = 3,
  parameter int  STRIDE = 1,
  parameter int  PAD    = 1,
  parameter int  H_IN   = 6,
  parameter int  W_IN   = 6,
  parameter int  P_COUT = 4,
  parameter int  P_CIN  = 4,
  parameter int  SILU   = 1,
  parameter real S_OUT_PRE  = 4.0 / 127.0,
  parameter real S_OUT_SILU = 4.0 / 127.0,
  parameter string WINIT = "cs_w.hex",
  parameter string SINIT = "cs_s.hex",
  parameter string BINIT = "cs_b.hex"
) (
  input  logic clk_i,
  input  logic rst_ni,

  // ── DUT (conv_stage) stream interface ──
  input  logic                  start_i,
  output logic                  done_o,
  input  logic                  ivalid_i,
  output logic                  iready_o,
  input  logic [CIN*8-1:0]      idata_flat_i,
  output logic                  ovalid_o,
  input  logic                  oready_i,
  output logic [COUT*8-1:0]     odata_flat_o,

  // ── REF (conv_layer) tile interface ──
  input  logic                            ref_valid_i,
  input  logic                            ref_first_i,
  input  logic                            ref_last_i,
  input  logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0] ref_ct_i,
  input  logic [K*K*P_CIN*8-1:0]          ref_x_flat_i,
  input  logic [P_COUT*K*K*P_CIN*8-1:0]   ref_w_flat_i,
  input  logic [P_COUT*16-1:0]            ref_scale_flat_i,
  input  logic [P_COUT*16-1:0]            ref_bias_flat_i,
  output logic                            ref_valid_o,
  output logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0] ref_ct_o,
  output logic [P_COUT*8-1:0]             ref_y_flat_o
);

  // ── Unflatten DUT idata / flatten odata ──
  logic signed [CIN-1:0][7:0]  dut_idata;
  logic signed [COUT-1:0][7:0] dut_odata;
  genvar gi;
  generate
    for (gi = 0; gi < CIN; gi++) assign dut_idata[gi] = idata_flat_i[gi*8 +: 8];
    for (gi = 0; gi < COUT; gi++) assign odata_flat_o[gi*8 +: 8] = dut_odata[gi];
  endgenerate

  conv_stage #(
    .CIN(CIN), .COUT(COUT), .K(K), .STRIDE(STRIDE), .PAD(PAD),
    .H_IN(H_IN), .W_IN(W_IN), .P_COUT(P_COUT), .P_CIN(P_CIN),
    .SILU(SILU), .S_OUT_PRE(S_OUT_PRE), .S_OUT_SILU(S_OUT_SILU),
    .WINIT(WINIT), .SINIT(SINIT), .BINIT(BINIT)
  ) u_dut (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .start_i(start_i), .done_o(done_o),
    .ivalid_i(ivalid_i), .iready_o(iready_o), .idata_i(dut_idata),
    .ovalid_o(ovalid_o), .oready_i(oready_i), .odata_o(dut_odata)
  );

  // ── Unflatten REF tile inputs / flatten outputs ──
  localparam int NL = K*K*P_CIN;
  logic signed [NL-1:0][7:0]              ref_x;
  logic signed [P_COUT-1:0][NL-1:0][7:0]  ref_w;
  logic        [P_COUT-1:0][15:0]         ref_scale;
  logic        [P_COUT-1:0][15:0]         ref_bias;
  logic signed [P_COUT-1:0][7:0]          ref_y;
  genvar gl, gc;
  generate
    for (gl = 0; gl < NL; gl++) assign ref_x[gl] = ref_x_flat_i[gl*8 +: 8];
    for (gc = 0; gc < P_COUT; gc++) begin : g_refw
      assign ref_scale[gc] = ref_scale_flat_i[gc*16 +: 16];
      assign ref_bias[gc]  = ref_bias_flat_i[gc*16 +: 16];
      assign ref_y_flat_o[gc*8 +: 8] = ref_y[gc];
      for (gl = 0; gl < NL; gl++)
        assign ref_w[gc][gl] = ref_w_flat_i[(gc*NL + gl)*8 +: 8];
    end
  endgenerate

  conv_layer #(
    .CIN(CIN), .COUT(COUT), .K(K), .STRIDE(STRIDE), .PAD(PAD),
    .H_OUT((H_IN+2*PAD-K)/STRIDE+1), .W_OUT((W_IN+2*PAD-K)/STRIDE+1),
    .P_COUT(P_COUT), .P_CIN(P_CIN), .RESIDUAL(0), .SILU(SILU),
    .S_OUT_PRE(S_OUT_PRE), .S_OUT_SILU(S_OUT_SILU)
  ) u_ref (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .valid_i(ref_valid_i), .ready_o(),
    .first_cin_i(ref_first_i), .last_cin_i(ref_last_i), .cout_tile_idx_i(ref_ct_i),
    .x_i(ref_x), .w_i(ref_w), .scale_i(ref_scale), .bias_i(ref_bias),
    .r_i('0), .r_scale_i('0), .r_bias_i('0),
    .valid_o(ref_valid_o), .ready_i(1'b1), .cout_tile_idx_o(ref_ct_o), .y_o(ref_y)
  );

endmodule
