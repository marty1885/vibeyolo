// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// cs_frame_tb — single conv_stage instance, real-image full-frame driver.
// Ports flattened to 1-D vectors for the Verilator C++ interface. Weights /
// scale / bias come from per-layer $readmemh ROMs (WINIT/SINIT/BINIT) and the
// input frame is streamed by the C++ driver from input_i8.hex.

module cs_frame_tb #(
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
  parameter string WINIT = "weights.i8.hex",
  parameter string SINIT = "scale_fp16.hex",
  parameter string BINIT = "bias_fp16.hex"
) (
  input  logic              clk_i,
  input  logic              rst_ni,
  input  logic              start_i,
  output logic              done_o,
  input  logic              ivalid_i,
  output logic              iready_o,
  input  logic [CIN*8-1:0]  idata_flat_i,
  output logic              ovalid_o,
  input  logic              oready_i,
  output logic [COUT*8-1:0] odata_flat_o
);
  logic signed [CIN-1:0][7:0]  idata;
  logic signed [COUT-1:0][7:0] odata;
  genvar gi;
  generate
    for (gi = 0; gi < CIN;  gi++) assign idata[gi] = idata_flat_i[gi*8 +: 8];
    for (gi = 0; gi < COUT; gi++) assign odata_flat_o[gi*8 +: 8] = odata[gi];
  endgenerate

  conv_stage #(
    .CIN(CIN), .COUT(COUT), .K(K), .STRIDE(STRIDE), .PAD(PAD),
    .H_IN(H_IN), .W_IN(W_IN), .P_COUT(P_COUT), .P_CIN(P_CIN),
    .SILU(SILU), .S_OUT_PRE(S_OUT_PRE), .S_OUT_SILU(S_OUT_SILU),
    .WINIT(WINIT), .SINIT(SINIT), .BINIT(BINIT)
  ) u_dut (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .start_i(start_i), .done_o(done_o),
    .ivalid_i(ivalid_i), .iready_o(iready_o), .idata_i(idata),
    .ovalid_o(ovalid_o), .oready_i(oready_i), .odata_o(odata)
  );
endmodule
