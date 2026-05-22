// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// conv_layer_tb — parameterized Verilator wrapper that drives DUT and REF
// in lockstep on identical stimulus.
//
// The Makefile builds one binary per configuration by passing parameter
// overrides via -G<NAME>=<VALUE>. The C++ side sees its own copy of the
// geometry via -CFLAGS "-DCFG_*=...".

module conv_layer_tb
#(
  parameter int  CIN          = 32,
  parameter int  COUT         = 32,
  parameter int  K            = 1,
  parameter int  STRIDE       = 1,
  parameter int  PAD          = 0,
  parameter int  H_OUT        = 8,
  parameter int  W_OUT        = 8,
  parameter int  P_COUT       = 16,
  parameter int  P_CIN        = 8,
  parameter int  RESIDUAL     = 0,
  parameter int  SILU         = 1
) (
  input  logic                                    clk_i,
  input  logic                                    rst_ni,

  input  logic                                    valid_i,
  input  logic                                    first_cin_i,
  input  logic                                    last_cin_i,
  // verilator lint_off UNUSEDSIGNAL
  input  logic [7:0]                              cout_tile_idx_i,   // wide enough for all configs
  // verilator lint_on UNUSEDSIGNAL

  // Flattened ports
  input  logic [K*K*P_CIN*8-1:0]                  x_flat_i,
  input  logic [P_COUT*K*K*P_CIN*8-1:0]           w_flat_i,
  input  logic [P_COUT*16-1:0]                    scale_flat_i,
  input  logic [P_COUT*16-1:0]                    bias_flat_i,

  input  logic [P_COUT*8-1:0]                     r_flat_i,
  input  logic [P_COUT*16-1:0]                    r_scale_flat_i,
  input  logic [P_COUT*16-1:0]                    r_bias_flat_i,

  output logic                                    dut_valid_o,
  output logic [7:0]                              dut_ct_o,
  output logic [P_COUT*8-1:0]                     dut_y_flat_o,

  output logic                                    ref_valid_o,
  output logic [7:0]                              ref_ct_o,
  output logic [P_COUT*8-1:0]                     ref_y_flat_o
);

  localparam int N_COUT_TILE = (COUT + P_COUT - 1) / P_COUT;
  localparam int CT_W        = $clog2((N_COUT_TILE < 2) ? 2 : N_COUT_TILE);

  logic signed [K*K*P_CIN-1:0][7:0]                x_w;
  logic signed [P_COUT-1:0][K*K*P_CIN-1:0][7:0]    w_w;
  logic        [P_COUT-1:0][15:0]                  scale_w;
  logic        [P_COUT-1:0][15:0]                  bias_w;
  logic signed [P_COUT-1:0][7:0]                   r_w;
  logic        [P_COUT-1:0][15:0]                  rs_w;
  logic        [P_COUT-1:0][15:0]                  rb_w;

  assign x_w     = x_flat_i;
  assign w_w     = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign r_w     = r_flat_i;
  assign rs_w    = r_scale_flat_i;
  assign rb_w    = r_bias_flat_i;

  logic signed [P_COUT-1:0][7:0] dut_y_w, ref_y_w;
  assign dut_y_flat_o = dut_y_w;
  assign ref_y_flat_o = ref_y_w;

  logic [CT_W-1:0] dut_ct_w, ref_ct_w;
  always_comb begin
    dut_ct_o = '0; dut_ct_o[CT_W-1:0] = dut_ct_w;
    ref_ct_o = '0; ref_ct_o[CT_W-1:0] = ref_ct_w;
  end

  // verilator lint_off UNUSEDSIGNAL
  logic dut_ready, ref_ready;
  // verilator lint_on UNUSEDSIGNAL

  conv_layer #(
    .CIN(CIN), .COUT(COUT), .K(K), .STRIDE(STRIDE), .PAD(PAD),
    .H_OUT(H_OUT), .W_OUT(W_OUT),
    .P_COUT(P_COUT), .P_CIN(P_CIN),
    .RESIDUAL(RESIDUAL), .SILU(SILU)
  ) u_dut (
    .clk_i           (clk_i),
    .rst_ni          (rst_ni),
    .valid_i         (valid_i),
    .ready_o         (dut_ready),
    .first_cin_i     (first_cin_i),
    .last_cin_i      (last_cin_i),
    .cout_tile_idx_i (cout_tile_idx_i[CT_W-1:0]),
    .x_i             (x_w),
    .w_i             (w_w),
    .scale_i         (scale_w),
    .bias_i          (bias_w),
    .r_i             (r_w),
    .r_scale_i       (rs_w),
    .r_bias_i        (rb_w),
    .valid_o         (dut_valid_o),
    .ready_i         (1'b1),
    .cout_tile_idx_o (dut_ct_w),
    .y_o             (dut_y_w)
  );

  conv_layer_ref #(
    .CIN(CIN), .COUT(COUT), .K(K), .STRIDE(STRIDE), .PAD(PAD),
    .H_OUT(H_OUT), .W_OUT(W_OUT),
    .P_COUT(P_COUT), .P_CIN(P_CIN),
    .RESIDUAL(RESIDUAL), .SILU(SILU)
  ) u_ref (
    .clk_i           (clk_i),
    .rst_ni          (rst_ni),
    .valid_i         (valid_i),
    .ready_o         (ref_ready),
    .first_cin_i     (first_cin_i),
    .last_cin_i      (last_cin_i),
    .cout_tile_idx_i (cout_tile_idx_i[CT_W-1:0]),
    .x_i             (x_w),
    .w_i             (w_w),
    .scale_i         (scale_w),
    .bias_i          (bias_w),
    .r_i             (r_w),
    .r_scale_i       (rs_w),
    .r_bias_i        (rb_w),
    .valid_o         (ref_valid_o),
    .ready_i         (1'b1),
    .cout_tile_idx_o (ref_ct_w),
    .y_o             (ref_y_w)
  );

endmodule
