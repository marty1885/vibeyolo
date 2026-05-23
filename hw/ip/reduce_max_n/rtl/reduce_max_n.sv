// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// reduce_max_n — signed int8 N-wide max reduction (1 element/cycle).
//
// All N candidates arrive in parallel on `x_i`; on each rising edge with
// `en_i` high the registered output `y_o` is updated to the maximum
// element and `valid_o` pulses for that cycle. Async-assert /
// sync-deassert active-low reset drives `y_o` to 0. When `en_i` is low the
// prior value is held and `valid_o` is low.
//
// Used by the detect head (/model.23) to reduce the 80 per-anchor class
// logits to a single ranking score (N=80). Because all N class channels
// share one per-tensor activation scale, taking the max in int8 preserves
// the fp16 ordering — the dequant to a common fp16 happens downstream on
// the single survivor, not on all N lanes.
//
// Implementation: balanced binary comparator tree built by a recursive
// `automatic` function — O(log2 N) gate depth, parameter-clean for any N.
// Odd-sized levels carry the trailing element up unchanged (max is
// associative and max(a)==a). Independently re-derived golden lives in
// reduce_max_n_ref.sv as a flat linear scan.

module reduce_max_n #(
  parameter int N = 80
) (
  input  logic                          clk_i,
  input  logic                          rst_ni,

  input  logic                          en_i,
  input  logic signed [N-1:0][7:0]      x_i,

  output logic                          valid_o,
  output logic signed [7:0]             y_o
);

  // Recursive pairwise max over an unpacked array of N signed int8s.
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
        if (len[0]) begin   // carry odd trailing element up the tree
          nxt[j] = cur[len-1];
          j++;
        end
        for (int i = 0; i < j; i++) cur[i] = nxt[i];
        len = j;
      end
      return cur[0];
    end
  endfunction

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
      y_o     <= '0;
      valid_o <= 1'b0;
    end else begin
      valid_o <= en_i;
      if (en_i) begin
        y_o <= max_comb;
      end
    end
  end

endmodule
