// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// maxpool_kxk — signed int8 K*K max-reduction.
//
// Parameterized K-by-K max pool tap (or, equivalently, an N-wide signed
// max reducer with N = K*K). All K*K candidates arrive in parallel on
// `x_i`; on each rising edge with en_i high the registered output `y_o`
// is updated to the maximum element. Async-assert / sync-deassert
// active-low reset drives `y_o` to 0. When en_i is low the prior value
// is held.
//
// Implementation: balanced binary comparator tree built by a recursive
// `automatic` function. This gives O(log2(N)) gate depth and keeps the
// implementation parameter-clean for K in {2,3,5} (and any other K).
// For odd-sized levels the trailing odd element is carried up one
// level unchanged — this is correct because max is associative and
// max(a) == a, and it matches a power-of-two tree padded with -∞ on
// the unused leaves but without the extra dummy comparators.

module maxpool_kxk #(
  parameter int K = 5
) (
  input  logic                              clk_i,
  input  logic                              rst_ni,

  input  logic                              en_i,
  input  logic signed [K*K-1:0][7:0]        x_i,

  output logic signed [7:0]                 y_o
);

  localparam int N = K * K;

  // Recursive pairwise max over an unpacked array of N signed int8s.
  // Walks the input array, building the next level into a packed-half
  // array, until only one element remains. `automatic` so each call gets
  // its own locals — required for recursion in SV.
  function automatic logic signed [7:0] smax_reduce(
      input logic signed [7:0] in [N]
  );
    logic signed [7:0] cur [N];
    logic signed [7:0] nxt [N];
    int len;
    int j;
    begin
      for (int i = 0; i < N; i++) cur[i] = in[i];
      len = N;
      while (len > 1) begin
        j = 0;
        for (int i = 0; i + 1 < len; i += 2) begin
          nxt[j] = (cur[i] > cur[i+1]) ? cur[i] : cur[i+1];
          j++;
        end
        // Carry odd trailing element straight up the tree.
        if (len[0]) begin
          nxt[j] = cur[len-1];
          j++;
        end
        for (int i = 0; i < j; i++) cur[i] = nxt[i];
        len = j;
      end
      return cur[0];
    end
  endfunction

  // Unpack the packed input into an SV unpacked array for the function.
  logic signed [7:0] x_arr [N];
  always_comb begin
    for (int i = 0; i < N; i++) begin
      x_arr[i] = x_i[i];
    end
  end

  logic signed [7:0] max_comb;
  always_comb begin
    max_comb = smax_reduce(x_arr);
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      y_o <= '0;
    end else if (en_i) begin
      y_o <= max_comb;
    end
  end

endmodule
