// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// dotN — N-wide signed int8 dot product, pipelined.
//
// Computes y = sum_{i=0..N-1} signed(a_i[i]) * signed(b_i[i]).
//
// Pipeline:
//   - Stage 0: N parallel mac8 instances each forced into "clr every cycle"
//     mode, so acc_o becomes simply the registered product a*b.
//   - Stages 1..L (L = ceil(log2(N))): balanced pairwise adder tree with
//     a register between each level. Odd-trailing element is carried up
//     unchanged (correct because + is associative and x + 0 == x; this
//     matches a pow-2 tree padded with 0 on unused leaves but without
//     the dummy adders).
//
// Total latency = 1 + ceil(log2(N)) cycles from en_i to valid_o.
//
// valid_o is the en_i token shift-registered through the same depth.
// clr_i (synchronous) flushes all pipeline registers and the valid
// shift register to 0. Reset (!rst_ni) does the same asynchronously.

module dotN #(
  parameter int N = 16
) (
  input  logic                       clk_i,
  input  logic                       rst_ni,

  input  logic                       en_i,
  input  logic                       clr_i,
  input  logic signed [N-1:0][7:0]   a_i,
  input  logic signed [N-1:0][7:0]   b_i,

  output logic signed [31:0]         y_o,
  output logic                       valid_o
);

  // ── Constants ────────────────────────────────────────
  // Number of adder-tree levels needed to fold N inputs to 1.
  // For N==1, no tree levels are needed (LEVELS == 0); for N>=2 use
  // $clog2(N). LATENCY counts the product stage plus all tree stages.
  localparam int LEVELS  = (N <= 1) ? 0 : $clog2(N);
  localparam int LATENCY = 1 + LEVELS;

  // ── Stage 0: N parallel registered products via mac8 ─
  // Each mac8 is held in clr-every-cycle mode so its acc_o is just the
  // registered product. en_i gates the load; clr_i forces 0.
  logic signed [31:0] prod_q [N];

  genvar gi;
  generate
    for (gi = 0; gi < N; gi++) begin : g_mul
      // Drive mac8 with clr=1 every cycle so acc_o == 32'(a*b) one cycle
      // after the inputs. When en_i is low or clr_i is high we want the
      // product register to load 0, which we get by zeroing the inputs.
      logic signed [7:0] a_lane;
      logic signed [7:0] b_lane;
      always_comb begin
        if (clr_i || !en_i) begin
          a_lane = 8'sd0;
          b_lane = 8'sd0;
        end else begin
          a_lane = a_i[gi];
          b_lane = b_i[gi];
        end
      end

      mac8 u_mul (
        .clk_i  (clk_i),
        .rst_ni (rst_ni),
        .clr_i  (1'b1),       // load product every cycle
        .en_i   (1'b1),       // unused when clr_i==1, but tie active
        .a_i    (a_lane),
        .b_i    (b_lane),
        .acc_o  (prod_q[gi])
      );
    end
  endgenerate

  // ── Stages 1..LEVELS: pipelined balanced adder tree ──
  // We use a flat 2D array sized to the max width (N) at all levels and
  // simply ignore unused entries past `len`. `len` halves each level
  // (rounding up to handle the odd-trailing carry).
  //
  // node[0][.] is the product layer (combinational view of prod_q).
  // node[s][.] for s in 1..LEVELS is the s-th adder-tree register layer.

  logic signed [31:0] node [LEVELS+1][N];

  // Level 0: feed in registered products.
  always_comb begin
    for (int i = 0; i < N; i++) begin
      node[0][i] = prod_q[i];
    end
  end

  // Compute the width of each level at elaboration time.
  function automatic int level_len(input int lvl);
    int len;
    begin
      len = N;
      for (int s = 0; s < lvl; s++) begin
        len = (len + 1) / 2;
      end
      return len;
    end
  endfunction

  // Pipelined adder tree.
  generate
    for (genvar s = 1; s <= LEVELS; s++) begin : g_tree
      localparam int PREV = level_len(s-1);
      localparam int CUR  = level_len(s);

      for (genvar k = 0; k < CUR; k++) begin : g_node
        // Pair (2k, 2k+1) from previous level; if 2k+1 is past PREV-1,
        // carry the lone element up. This is "+0" semantically.
        if (2*k + 1 < PREV) begin : g_pair
          always_ff @(posedge clk_i or negedge rst_ni) begin
            if (!rst_ni) begin
              node[s][k] <= 32'sd0;
            end else if (clr_i) begin
              node[s][k] <= 32'sd0;
            end else begin
              node[s][k] <= node[s-1][2*k] + node[s-1][2*k+1];
            end
          end
        end else begin : g_carry
          always_ff @(posedge clk_i or negedge rst_ni) begin
            if (!rst_ni) begin
              node[s][k] <= 32'sd0;
            end else if (clr_i) begin
              node[s][k] <= 32'sd0;
            end else begin
              node[s][k] <= node[s-1][2*k];
            end
          end
        end
      end

      // Zero out unused upper slots so they don't show up as X in
      // simulation or get inferred as latches. Synthesis tools will
      // optimize these tied-zero regs away.
      for (genvar k = CUR; k < N; k++) begin : g_pad
        assign node[s][k] = 32'sd0;
      end
    end
  endgenerate

  // ── Output: top of tree (or the single product when N==1) ───
  generate
    if (LEVELS == 0) begin : g_n1
      // N == 1: y_o is just the registered product.
      assign y_o = prod_q[0];
    end else begin : g_top
      assign y_o = node[LEVELS][0];
    end
  endgenerate

  // ── Valid pipeline: shift en_i through LATENCY stages ───
  logic [LATENCY-1:0] valid_sr;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      valid_sr <= '0;
    end else if (clr_i) begin
      valid_sr <= '0;
    end else begin
      if (LATENCY == 1) begin
        valid_sr[0] <= en_i;
      end else begin
        valid_sr <= {valid_sr[LATENCY-2:0], en_i};
      end
    end
  end

  assign valid_o = valid_sr[LATENCY-1];

endmodule
