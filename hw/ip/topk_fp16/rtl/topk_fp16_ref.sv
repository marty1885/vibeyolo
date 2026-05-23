// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// topk_fp16_ref — behavioural golden for topk_fp16.
//
// Maintains a sorted-by-value array of up to K (value, index) pairs, the
// running top-K set. On each accepted input:
//   • if fill < K: insert into the sorted array
//   • else if value > min: drop the min, insert new
//
// After N inputs (count == N), emits a done pulse and presents the K
// elements on out_value_o / out_index_o in the same order the DUT happens
// to emit them — *which order is heap-order in the DUT*, not sorted. The
// TB compares the two as multisets (sorts both before equality check), so
// for ref-correctness this module emits values **sorted descending** —
// that's a deterministic canonical order useful for debugging too.
//
// Comparisons in `real` arithmetic, exact for fp16 magnitudes since fp16
// has only 11 significand bits.

module topk_fp16_ref #(
  parameter int N     = 8400,
  parameter int K     = 300,
  parameter int IDX_W = $clog2(N)
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,

  input  logic                 start_i,

  input  logic                 in_valid_i,
  input  logic [15:0]          in_value_i,
  input  logic [IDX_W-1:0]     in_index_i,
  output logic                 in_ready_o,

  output logic                 done_o,
  output logic [15:0]          out_value_o [K],
  output logic [IDX_W-1:0]     out_index_o [K]
);

  localparam int CNT_W = $clog2(N + 1);

  // fp16 → real (handles ±0, normals, subnormals; NaN → very small).
  function automatic real fp16_to_real(input logic [15:0] x);
    logic        s;
    logic [4:0]  eb;
    logic [9:0]  f;
    real         v;
    int          e;
    int          k;
    real         m;
    begin
      s  = x[15]; eb = x[14:10]; f = x[9:0];
      if (eb == 5'd31) begin
        if (f != 0) v = -1.0e30;                   // NaN treated as -inf
        else        v = 1.0e30;                    // Inf
      end else if (eb == 5'd0) begin
        v = real'(f) * (1.0 / 16777216.0);
      end else begin
        m = 1.0 + real'(f) / 1024.0;
        e = int'(eb) - 15;
        if (e >= 0) for (k = 0; k < e;  k = k + 1) m = m * 2.0;
        else        for (k = 0; k < -e; k = k + 1) m = m / 2.0;
        v = m;
      end
      return s ? -v : v;
    end
  endfunction

  // Sorted-descending arrays.
  logic [15:0]      buf_val [K];
  logic [IDX_W-1:0] buf_idx [K];
  int               buf_len;

  logic [CNT_W-1:0] count_q;
  logic             active_q;     // 1 while we've been started and not yet done
  logic             done_q;

  assign in_ready_o = active_q && (count_q < CNT_W'(N));
  assign done_o     = done_q;

  // Insert (val, idx) into the sorted-descending array. If full, drop the
  // smallest if val > smallest; otherwise drop incoming.
  task automatic insert_sorted(input logic [15:0] v, input logic [IDX_W-1:0] idx);
    real         vr, cmp;
    int          i, j;
    int          ins_pos;
    begin
      vr = fp16_to_real(v);
      if (buf_len < K) begin
        // find insertion position
        ins_pos = buf_len;
        for (i = 0; i < buf_len; i = i + 1) begin
          if (ins_pos == buf_len) begin
            cmp = fp16_to_real(buf_val[i]);
            if (vr > cmp) ins_pos = i;
          end
        end
        // shift right
        for (j = buf_len; j > ins_pos; j = j - 1) begin
          buf_val[j] = buf_val[j-1];
          buf_idx[j] = buf_idx[j-1];
        end
        buf_val[ins_pos] = v;
        buf_idx[ins_pos] = idx;
        buf_len = buf_len + 1;
      end else begin
        // full. compare to smallest (at K-1)
        cmp = fp16_to_real(buf_val[K-1]);
        if (vr > cmp) begin
          // find insertion position
          ins_pos = K - 1;
          for (i = 0; i < K - 1; i = i + 1) begin
            if (ins_pos == K - 1) begin
              cmp = fp16_to_real(buf_val[i]);
              if (vr > cmp) ins_pos = i;
            end
          end
          // shift right, dropping last
          for (j = K - 1; j > ins_pos; j = j - 1) begin
            buf_val[j] = buf_val[j-1];
            buf_idx[j] = buf_idx[j-1];
          end
          buf_val[ins_pos] = v;
          buf_idx[ins_pos] = idx;
        end
      end
    end
  endtask

  /* verilator lint_off BLKSEQ */
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      count_q  <= '0;
      active_q <= 1'b0;
      done_q   <= 1'b0;
      buf_len   = 0;
      for (int i = 0; i < K; i++) begin
        buf_val[i] <= 16'h0;
        buf_idx[i] <= '0;
      end
    end else begin
      done_q <= 1'b0;

      if (start_i && !active_q) begin
        count_q  <= '0;
        active_q <= 1'b1;
        buf_len   = 0;
      end else if (active_q) begin
        if (in_valid_i && count_q < CNT_W'(N)) begin
          insert_sorted(in_value_i, in_index_i);
          count_q <= count_q + CNT_W'(1);
          // if this was the last one, pulse done next cycle
          if (count_q + CNT_W'(1) == CNT_W'(N)) begin
            done_q   <= 1'b1;
            active_q <= 1'b0;
          end
        end
      end
    end
  end

  always_comb begin
    for (int i = 0; i < K; i++) begin
      out_value_o[i] = buf_val[i];
      out_index_o[i] = buf_idx[i];
    end
  end
  /* verilator lint_on BLKSEQ */

endmodule
