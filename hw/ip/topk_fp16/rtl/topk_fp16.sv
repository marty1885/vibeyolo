// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// topk_fp16 — streaming top-K selector over fp16 candidate values.
//
// Algorithm: small min-heap of K (value, index) pairs. Root holds the
// current minimum of the retained top-K set.
//
//   Fill phase (first K elements ingested):
//     For each new element:
//       1) push onto heap at the next free slot (tail)
//       2) sift up: while value < parent.value, swap with parent
//
//   Replace phase (remaining N-K elements):
//     For each new element:
//       1) compare against heap[0] (root, current min of top-K)
//       2) if greater: replace root, sift down (one level per cycle;
//          two parallel fp16 compares per cycle pick the smaller child
//          and check it against the current parent)
//       3) else: skip (one cycle)
//
// Output is the K (value, index) pairs in heap order. TopK semantics only
// require the unordered set of K largest, so heap order is fine for any
// downstream consumer that gathers by index (the usual case).
//
// Interface contract:
//   • Pulse start_i for one cycle (from IDLE) to begin a new frame. After
//     start_i the IP asserts in_ready_o. Caller drives in_valid_i +
//     in_value_i + in_index_i on cycles where in_ready_o is high; back-
//     to-back accepts are not guaranteed (in_ready_o drops during multi-
//     cycle sift work).
//   • After N elements have been accepted, the IP completes any in-flight
//     sift, then asserts done_o for one cycle. The K elements are valid
//     on out_value_o / out_index_o (registered).
//   • While done_o is low after start_i, the result is not valid.
//
// Parameters:
//   N      — total number of candidates per frame (default 8400)
//   K      — number of top elements to retain (default 300)
//   IDX_W  — index bit width (default $clog2(N))
//
// fp16 comparison:
//   The comparator implements IEEE-754 binary16 sign-magnitude semantics
//   (NaN treated as smallest). For sigmoid outputs (non-negative) it
//   collapses to unsigned 16-bit compare on the bit pattern, but the IP
//   is correct for arbitrary signed fp16 inputs.

module topk_fp16 #(
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

  // ───────────────────────── parameters / helpers ────────────────

  // Heap slot index width. We use $clog2 for K and clamp to 1 to keep
  // array slicing well-formed for K==1.
  localparam int HIDX_W = (K <= 1) ? 1 : $clog2(K);
  // Fill counter goes from 0..K, needs one extra bit.
  localparam int FILL_W = (K <= 1) ? 1 : $clog2(K + 1);
  // Input counter goes from 0..N.
  localparam int CNT_W  = $clog2(N + 1);

  // fp16 compare: returns 1 if a > b. NaN treated as smallest.
  function automatic logic fp16_gt(input logic [15:0] a, input logic [15:0] b);
    logic        sa, sb;
    logic [4:0]  ea, eb;
    logic [9:0]  fa, fb;
    logic [14:0] ma, mb;
    logic        a_nan, b_nan;
    logic        a_zero, b_zero;
    begin
      sa = a[15]; ea = a[14:10]; fa = a[9:0];
      sb = b[15]; eb = b[14:10]; fb = b[9:0];
      a_nan  = (ea == 5'd31) && (fa != 0);
      b_nan  = (eb == 5'd31) && (fb != 0);
      a_zero = (ea == 0) && (fa == 0);
      b_zero = (eb == 0) && (fb == 0);
      ma = a[14:0];
      mb = b[14:0];
      if (a_nan && b_nan)        fp16_gt = 1'b0;
      else if (a_nan)            fp16_gt = 1'b0;
      else if (b_nan)            fp16_gt = 1'b1;
      else if (a_zero && b_zero) fp16_gt = 1'b0;
      else if (sa != sb)         fp16_gt = !sa;
      else if (sa == 1'b0)       fp16_gt = (ma > mb);
      else                       fp16_gt = (ma < mb);
    end
  endfunction

  function automatic logic fp16_lt(input logic [15:0] a, input logic [15:0] b);
    fp16_lt = fp16_gt(b, a);
  endfunction

  // ───────────────────────── heap storage ────────────────────────

  logic [15:0]      heap_val_q [K];
  logic [IDX_W-1:0] heap_idx_q [K];

  // ── heap-navigation helpers (read the registered heap) ──
  // Used by the two-levels-per-cycle sift-down. `has_child` is true when the
  // left child of slot p exists (the heap is complete, so no left ⇒ no right).
  // `min_child` returns the smaller-valued child slot.
  function automatic logic has_child(input logic [HIDX_W-1:0] p);
    logic [HIDX_W:0] l;
    l = {1'b0, p} + {1'b0, p} + (HIDX_W+1)'(1);
    return (l < (HIDX_W+1)'(K));
  endfunction

  function automatic logic [HIDX_W-1:0] min_child(input logic [HIDX_W-1:0] p);
    logic [HIDX_W:0] l, r;
    logic            has_r;
    l     = {1'b0, p} + {1'b0, p} + (HIDX_W+1)'(1);
    r     = l + (HIDX_W+1)'(1);
    has_r = (r < (HIDX_W+1)'(K));
    if (has_r && fp16_lt(heap_val_q[r[HIDX_W-1:0]], heap_val_q[l[HIDX_W-1:0]]))
      min_child = r[HIDX_W-1:0];
    else
      min_child = l[HIDX_W-1:0];
  endfunction

  // ───────────────────────── FSM ─────────────────────────────────

  typedef enum logic [2:0] {
    S_IDLE,
    S_ACCEPT,
    S_SIFT_UP,
    S_SIFT_DOWN,
    S_DONE
  } state_e;

  state_e state_q, state_d;

  logic [CNT_W-1:0]  count_q,  count_d;
  logic [FILL_W-1:0] fill_q,   fill_d;
  // cur_q indexes heap slots, width HIDX_W.
  logic [HIDX_W-1:0] cur_q,    cur_d;

  logic [15:0]       pend_val_q, pend_val_d;
  logic [IDX_W-1:0]  pend_idx_q, pend_idx_d;

  // Convenience constants
  // verilator lint_off WIDTHTRUNC
  // (Local constants below are wider than needed at the K bound — we mask
  //  back to HIDX_W when indexing the heap arrays.)
  // verilator lint_on WIDTHTRUNC

  assign in_ready_o = (state_q == S_ACCEPT) && (count_q != CNT_W'(N));
  assign done_o     = (state_q == S_DONE);

  // ── next-state logic ──
  always_comb begin
    automatic logic [HIDX_W-1:0] par_idx;
    automatic logic [HIDX_W-1:0] sc1, sc2;   // first/second-level smaller child

    state_d    = state_q;
    count_d    = count_q;
    fill_d     = fill_q;
    cur_d      = cur_q;
    pend_val_d = pend_val_q;
    pend_idx_d = pend_idx_q;

    par_idx = '0;
    sc1     = '0;
    sc2     = '0;

    case (state_q)
      S_IDLE: begin
        if (start_i) state_d = S_ACCEPT;
      end

      S_ACCEPT: begin
        if (count_q == CNT_W'(N)) begin
          state_d = S_DONE;
        end else if (in_valid_i) begin
          pend_val_d = in_value_i;
          pend_idx_d = in_index_i;
          count_d    = count_q + CNT_W'(1);
          if (fill_q != FILL_W'(K)) begin
            // Fill phase: slot to write = fill_q (truncated to HIDX_W bits).
            cur_d   = fill_q[HIDX_W-1:0];
            fill_d  = fill_q + FILL_W'(1);
            state_d = S_SIFT_UP;
          end else begin
            if (fp16_gt(in_value_i, heap_val_q[0])) begin
              cur_d   = '0;
              state_d = S_SIFT_DOWN;
            end
            // else stay in S_ACCEPT
          end
        end
      end

      S_SIFT_UP: begin
        if (cur_q == '0) begin
          state_d = (count_q == CNT_W'(N)) ? S_DONE : S_ACCEPT;
        end else begin
          par_idx = (cur_q - HIDX_W'(1)) >> 1;
          if (fp16_lt(heap_val_q[cur_q], heap_val_q[par_idx])) begin
            cur_d   = par_idx;
            state_d = S_SIFT_UP;
          end else begin
            state_d = (count_q == CNT_W'(N)) ? S_DONE : S_ACCEPT;
          end
        end
      end

      S_SIFT_DOWN: begin
        // Descend up to two heap levels this cycle. The value being sifted
        // is V = heap[cur_q]. Level 1: smaller child sc1; if V <= heap[sc1]
        // it has settled. Level 2: from sc1, smaller child sc2; if heap[sc2]
        // < V, V moves two levels (→ sc2) and we keep sifting; otherwise V
        // settles at sc1.
        if (!has_child(cur_q)) begin
          state_d = (count_q == CNT_W'(N)) ? S_DONE : S_ACCEPT;
        end else begin
          sc1 = min_child(cur_q);
          if (!fp16_lt(heap_val_q[sc1], heap_val_q[cur_q])) begin
            state_d = (count_q == CNT_W'(N)) ? S_DONE : S_ACCEPT;
          end else if (!has_child(sc1)) begin
            // sc1 is a leaf: one-level swap, then settled.
            cur_d   = sc1;
            state_d = (count_q == CNT_W'(N)) ? S_DONE : S_ACCEPT;
          end else begin
            sc2 = min_child(sc1);
            if (fp16_lt(heap_val_q[sc2], heap_val_q[cur_q])) begin
              cur_d   = sc2;
              state_d = S_SIFT_DOWN;
            end else begin
              cur_d   = sc1;
              state_d = (count_q == CNT_W'(N)) ? S_DONE : S_ACCEPT;
            end
          end
        end
      end

      S_DONE: begin
        state_d = S_IDLE;
      end

      default: state_d = S_IDLE;
    endcase
  end

  // ── sequential heap update ──
  always_ff @(posedge clk_i or negedge rst_ni) begin
    automatic logic [HIDX_W-1:0] par_idx;
    automatic logic [HIDX_W-1:0] sc1, sc2;   // first/second-level smaller child
    automatic logic [15:0]       tmp_v;
    automatic logic [IDX_W-1:0]  tmp_x;

    if (!rst_ni) begin
      state_q    <= S_IDLE;
      count_q    <= '0;
      fill_q     <= '0;
      cur_q      <= '0;
      pend_val_q <= '0;
      pend_idx_q <= '0;
      for (int i = 0; i < K; i++) begin
        heap_val_q[i] <= 16'h0;
        heap_idx_q[i] <= '0;
      end
    end else begin
      state_q    <= state_d;
      count_q    <= count_d;
      fill_q     <= fill_d;
      cur_q      <= cur_d;
      pend_val_q <= pend_val_d;
      pend_idx_q <= pend_idx_d;

      if (start_i && state_q == S_IDLE) begin
        count_q <= '0;
        fill_q  <= '0;
      end

      case (state_q)
        S_ACCEPT: begin
          if (in_valid_i && count_q != CNT_W'(N)) begin
            if (fill_q != FILL_W'(K)) begin
              heap_val_q[fill_q[HIDX_W-1:0]] <= in_value_i;
              heap_idx_q[fill_q[HIDX_W-1:0]] <= in_index_i;
            end else begin
              if (fp16_gt(in_value_i, heap_val_q[0])) begin
                heap_val_q[0] <= in_value_i;
                heap_idx_q[0] <= in_index_i;
              end
            end
          end
        end

        S_SIFT_UP: begin
          if (cur_q != '0) begin
            par_idx = (cur_q - HIDX_W'(1)) >> 1;
            if (fp16_lt(heap_val_q[cur_q], heap_val_q[par_idx])) begin
              tmp_v = heap_val_q[cur_q];
              tmp_x = heap_idx_q[cur_q];
              heap_val_q[cur_q]   <= heap_val_q[par_idx];
              heap_idx_q[cur_q]   <= heap_idx_q[par_idx];
              heap_val_q[par_idx] <= tmp_v;
              heap_idx_q[par_idx] <= tmp_x;
            end
          end
        end

        S_SIFT_DOWN: begin
          // Mirror the next-state logic: rotate V = heap[cur_q] down up to two
          // levels in one cycle. cur/sc1/sc2 are parent/child/grandchild, hence
          // distinct, so the multi-cell write never aliases. RHS reads are the
          // old (registered) values.
          if (has_child(cur_q)) begin
            sc1 = min_child(cur_q);
            if (fp16_lt(heap_val_q[sc1], heap_val_q[cur_q])) begin
              tmp_v = heap_val_q[cur_q];
              tmp_x = heap_idx_q[cur_q];
              if (!has_child(sc1)) begin
                // one-level swap (sc1 is a leaf)
                heap_val_q[cur_q] <= heap_val_q[sc1];
                heap_idx_q[cur_q] <= heap_idx_q[sc1];
                heap_val_q[sc1]   <= tmp_v;
                heap_idx_q[sc1]   <= tmp_x;
              end else begin
                sc2 = min_child(sc1);
                if (fp16_lt(heap_val_q[sc2], heap_val_q[cur_q])) begin
                  // two-level rotate: cur←sc1, sc1←sc2, sc2←V
                  heap_val_q[cur_q] <= heap_val_q[sc1];
                  heap_idx_q[cur_q] <= heap_idx_q[sc1];
                  heap_val_q[sc1]   <= heap_val_q[sc2];
                  heap_idx_q[sc1]   <= heap_idx_q[sc2];
                  heap_val_q[sc2]   <= tmp_v;
                  heap_idx_q[sc2]   <= tmp_x;
                end else begin
                  // one-level swap (V settles at sc1)
                  heap_val_q[cur_q] <= heap_val_q[sc1];
                  heap_idx_q[cur_q] <= heap_idx_q[sc1];
                  heap_val_q[sc1]   <= tmp_v;
                  heap_idx_q[sc1]   <= tmp_x;
                end
              end
            end
          end
        end

        default: ;
      endcase
    end
  end

  // ───────────────────────── output drive ────────────────────────
  always_comb begin
    for (int i = 0; i < K; i++) begin
      out_value_o[i] = heap_val_q[i];
      out_index_o[i] = heap_idx_q[i];
    end
  end

endmodule
