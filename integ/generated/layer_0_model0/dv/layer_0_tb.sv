// Generated TB for L0 (SILU=1).
module layer_0_tb
  import s_out_params::*;
(
  input  logic                  clk_i,
  input  logic                  rst_ni,
  input  logic                  valid_i,
  input  logic                  first_cin_i,
  input  logic                  last_cin_i,
  input  logic [0:0]      cout_tile_idx_i,
  input  logic [431:0]     x_flat_i,
  input  logic [3455:0]     w_flat_i,
  input  logic [255:0]     scale_flat_i,
  input  logic [255:0]     bias_flat_i,
  output logic                  valid_o,
  output logic [0:0]      cout_tile_idx_o,
  output logic [255:0]     y_flat_o
);
  logic signed [1:0][26:0][7:0]     x_w;
  logic signed [15:0][26:0][7:0]    w_w;
  logic        [15:0][15:0]                 scale_w, bias_w;
  logic signed [1:0][15:0][7:0]     y_w;
  assign x_w = x_flat_i;
  assign w_w = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign y_flat_o = y_w;
  layer_0 #(.P_COUT(16), .P_CIN(3), .P_PIX(2),
                .S_OUT_PRE(S_OUT_PRE_VAL), .S_OUT_SILU(S_OUT_SILU_VAL)) u_dut (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .valid_i(valid_i), .first_cin_i(first_cin_i), .last_cin_i(last_cin_i),
    .cout_tile_idx_i(cout_tile_idx_i),
    .x_i(x_w), .w_i(w_w), .scale_i(scale_w), .bias_i(bias_w),
    .valid_o(valid_o), .cout_tile_idx_o(cout_tile_idx_o), .y_o(y_w)
  );
endmodule
