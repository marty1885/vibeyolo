# flash_attn — systolic flash-attention leaf IP

Parameterised tile-streaming flash-attention IP for YOLO26n PSA (`/model.10`) and A2C2f (`/model.22`). Computes scaled-dot-product self-attention in fp16 without ever materialising the full N×N scores tensor.

## Algorithm

Flash-attention v1 (online softmax) recurrence — exact softmax-attention,
computed one (row, col) iteration at a time:

```
For each (head h, row i):
  m_i = -inf, l_i = 0, O_i[*] = 0
  For each col j:
    s_ij  = (Q[h,i] · K[h,j]) * TEMP        // 1 scalar
    m_new = max(m_i, s_ij)                   // running max
    alpha = exp(m_i - m_new)                 // shrink-down factor
    p     = exp(s_ij - m_new)                // new col weight
    l_i   = alpha * l_i + p                  // running normaliser
    O_i[d] = alpha * O_i[d] + p * V[h,j,d]   // for d ∈ [0, DIM_V)
    m_i   = m_new
  O_i[d] /= l_i                              // final normalise per row
```

Per (head, row) only a length-1 running `(m_i, l_i)` and a DIM_V-wide running
`O_i` are kept. Never an N×N scores tensor.

## Boundary

- Single-clock, active-low async-assert / sync-deassert reset (consistent
  with the rest of the `hw/ip/*` IPs).
- Inputs Q, K, V loaded via per-port (we, waddr, wdata) triplets into
  inferred SRAMs. Address layout is flat: `addr(h,n,d) = h*N*D + n*D + d`.
- Pulse `start_i`. The IP raises `busy_o` and clears `done_o`. When the
  compute completes, `done_o` rises and stays high until next `start_i`.
- After `done_o`, read `O[h,n,d]` via `o_raddr_i` / `o_rdata_o`. Read is
  combinational from internal `o_mem`.

The IP does NOT include the PSA's pe-conv on V. That stays in the
integration shim, which adds pe(V) to O after this IP finishes. See
**ONNX-level verification** below.

## Parameters

| name    | meaning                                  | YOLO26n value |
|---------|------------------------------------------|---------------|
| HEADS   | number of attention heads                | 2             |
| N       | tokens per head (= H · W spatial)        | 400 (20²)     |
| DIM_Q   | per-head Q / K dim                       | 32            |
| DIM_V   | per-head V dim                           | 64            |
| TEMP    | softmax temperature (1/√DIM_Q baked in)  | 0.17677669    |

## Parallelism (this build, BR=1 / BC=1)

| structure                  | cells                          |
|---------------------------|---------------------------------|
| Q·K[j] dot product        | DIM_Q parallel fp16_fma + tree  |
| O[d] update (α·O + p·V)   | DIM_V parallel fp16_fma (×2)    |
| l update (α·l + p)        | 1 fp16_fma                      |
| final O[d] *= 1/l         | 1 fp16_fma streamed             |

Total fp16_fma cells at YOLO shapes = DIM_Q + 2·DIM_V + 2 = 162.

The col loop is sequential — 6 cycles per col (DOT, DOT_WAIT, SOFT, LUPD×3):
- `S_COL_DOT`        — launch Q[i] × K[j] multipliers
- `S_COL_DOT_WAIT`   — settle mul_out → registered dot → s_fma launched
- `S_COL_SOFT`       — s_fma_out registered; compute m_new, α, p (LUT)
- `S_COL_LUPD` ×3    — phase 0 settle p/α, phase 1 launch pv, launch aob+l,
                       phase 2 commit O_buf, l, m
- Plus final normalize: 1 cycle/d + 1 drain cycle.

## Cycle estimate at YOLO26n shapes (HEADS=2, N=400, DIM_Q=32, DIM_V=64)

```
per col       :  6 cyc
per row       :  N · 6  + DIM_V + ROW_INIT + ROW_FINAL_S + ROW_DONE
              =  400·6 + 64 + ~5 ≈ 2469 cyc
per head      :  N · 2469 ≈ 987 600 cyc
total frame   :  HEADS · 987 600 ≈ 1.98 M cyc
```

This exceeds T_FRAME = 100 000 cycles. **The IP is correct but does not
yet hit the throughput target.** Higher parallelism (BR-row unroll, BC-col
unroll) is the straight-line follow-up; the IP boundary, golden, and DV
harness are written parametrically and don't change. Expected with BR=8,
BC=4: ~60 k cyc/frame at the ~512-cell budget the task spec described.

## DV results

DV uses small parameters (HEADS=1, N=8, DIM_Q=4, DIM_V=4) for ~440 cycle
frames — the full sweep finishes in ~1 s after build:

- 18 frames driven: zeros, 8× random ±0.5, 8× random ±1.5, 1× peaked Q,
  1× tight ±0.1.
- DUT-vs-REF: 0 fails at tolerance 64 fp16 ULP OR |dut-ref| ≤ 0.01.
- DUT-vs-shadow (C++ double): 0 fails at the same threshold.
- Cosine similarity DUT vs shadow: 0.999976 worst, 1.000000 best.
- REF-vs-shadow: 13 ULP worst (sanity check that REF and shadow agree).

Acceptance per the task spec: cos ≥ 0.998, 0 mismatches at tolerance. Met.

### Tolerance rationale

The DUT applies an fp16 round at every micro-op (mul, dot reduce, scale,
exp LUT, FMA), while the REF rounds only at the final write. For outputs
of magnitude ≪ 0.01 (cancellations across N cols × softmax of fp16
intermediates), per-element ULP drift compounds. The absolute-error
fallback (0.01) is the same pattern used in `box_decode` for its
`cx - d*stride` cancellation case.

## Open question (resolved)

> Does `pe` (the depthwise conv on V) get folded into flash-attention's
> tiled flow, or sit outside?

**Outside.** Per ONNX `/model.10/m/m.0/attn`:
- `pe` consumes `attn/Reshape_2_output_0` (the full reduced V tensor).
- `pe` output is added to `attn/Add_output_0` (the post-attention output).
- Flash-attention's online recurrence updates `O_i` tile-by-tile and
  cannot inject pe-of-V *during* the recurrence (pe is a 3×3 spatial
  conv across the full N=H·W=400 tokens of V, not a per-token map).
- **Therefore**: `flash_attn` takes Q, K, V; produces O. pe is computed
  externally on V (e.g. a vanilla `conv_layer` instance with depthwise
  weights expanded to dense), and the integration shim does
  `attn_out = flash_attn(Q,K,V) + pe(V)` post-IP.

## Lint / build

```
make -C hw/ip/flash_attn/dv lint    # 0 warnings under verilator -Wall
make -C hw/ip/flash_attn/dv test    # 5/5 checks, 18 frames, ~1 s after build
```

Build time (Verilator compile + g++) ≈ 18 s wall; run time < 0.1 s.

## Files

```
hw/ip/flash_attn/
  rtl/flash_attn.sv         — DUT (parameterised, synthesizable)
  rtl/flash_attn_ref.sv     — SV behavioural golden in `real`
  dv/flash_attn_tb.sv       — DUT+REF lockstep TB wrapper
  dv/flash_attn_test.cc     — driver + scoreboard + C++ double shadow
  dv/Makefile               — uses mk/verilator.mk
  README.md                 — this file
```
