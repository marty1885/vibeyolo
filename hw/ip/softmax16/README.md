# softmax16

16-way fp16 softmax for the YOLO26n DFL (Distribution Focal Loss)
detection head. Reads 16 fp16 logits in parallel and emits 16 fp16
probabilities that sum to approximately 1.

```
m       = max(x[0..15])              // numerical stability
e[i]    = exp(x[i] - m)              // fp16 exp LUT
s       = sum(e[0..15])              // fp16 add tree
inv_s   = 1 / s                      // fp16 reciprocal LUT
y[i]    = e[i] * inv_s               // fp16 FMA (c = 0)
```

## Ports

| Direction | Name      | Width       | Description |
|-----------|-----------|-------------|-------------|
| input     | `clk_i`   | 1           | Clock |
| input     | `rst_ni`  | 1           | Active-low async reset |
| input     | `valid_i` | 1           | Vector valid (whole 16-lane vector) |
| input     | `x_i[16]` | 16 × 16 bit | fp16 logits, lane 0 .. lane 15 |
| output    | `valid_o` | 1           | Vector valid (latency-aligned with `y_o`) |
| output    | `y_o[16]` | 16 × 16 bit | fp16 probabilities |

The 16 lanes are processed together as one vector: there is a single
valid flowing along the pipeline.

## Latency

12 cycles, fully pipelined (1 vector per cycle throughput):

| Stage | Operation                                             |
|-------|-------------------------------------------------------|
| S0    | input register                                        |
| S1..S4| 4-stage `fp16_max` tree → `m`                         |
| S5    | per-lane `d[i] = m - x[i]` (fp16 sub) + exp LUT       |
| S6..S9| 4-stage `fp16_add` tree → `s = Σ e[i]`                |
| S10   | reciprocal LUT → `inv_s`                              |
| S11   | per-lane `e[i] * inv_s + 0` via `fp16_fma`            |

`fp16_max`, the `fp16 sub/add` helper, and the exp/recip LUTs are
implemented inside `softmax16.sv`. The per-lane multiplier reuses the
already-verified `fp16_fma` block.

## LUT design

### Exp LUT (`exp_lut[0..1023]`)

- 1024 entries, fp16.
- Index = `round(d * 64)`, where `d = m - x[i] ≥ 0`, clamped to
  `[0, 1023]`.  Step = 1/64; range covered = `[0, 16)`.
- `exp(-16) ≈ 1.13e-7`, below normal fp16 — values beyond the LUT
  saturate to zero by reading `exp_lut[1023]`.
- Worst-case relative error from quantization-only LUT is approximately
  `step/2 = 1/128 ≈ 0.78%`, i.e. roughly 8 fp16 ULPs.
- LUT contents are pre-baked at elaboration time using SystemVerilog
  `real` and `$exp`.

### Reciprocal LUT (`recip_mant_lut[0..1023]`)

- 1024 entries indexed by the 10-bit fp16 fraction of `s`.
- Entry `f` = `fp16(1.0 / (1.0 + f/1024.0))`, i.e. fp16 reciprocal of the
  mantissa value in `[1.0, 2.0)`.
- For `s` with biased exponent `eb` and fraction `f`:
  ```
  inv_s = recip_mant_lut[f]  with biased_exp ← r_biased + 15 - eb
  ```
- Pure-LUT design (no Newton-Raphson refinement). With a 1024-entry
  mantissa LUT the relative error of the reciprocal is at most
  `1/2048 ≈ 0.05%`, well inside fp16 ULP.
- `s` for a 16-way softmax is always in `[1, 16]`, so its biased
  exponent is in `[15, 19]` — extremely well-behaved input.

## Tolerance

- Per-output-lane fp16 sign-magnitude ULP difference.
- DUT vs SV `_ref` and DUT vs independent C++ double-precision shadow
  must both be `≤ 32 ULP` per lane on every vector.
- Sum of outputs must be within `0.05` of `1.0` for the general random
  pool and within `0.02` for tight-range (±2) DFL-typical inputs.
- Observed worst case in the current build: `25 ULP` over 5000 random
  ±8 vectors and 1000 tight-range ±2 vectors.

### Where the budget goes

| Source                           | Worst-case ULP contribution |
|----------------------------------|-----------------------------|
| exp LUT (step 1/64)              | ~8                          |
| reciprocal LUT (1024-entry mant) | ~1                          |
| `m - x` fp16 subtract            | ~1                          |
| sum tree (4 fp16-add stages)     | up to ~4 per term           |
| final fp16 FMA multiply          | ~1                          |
| compounded over per-lane × vec   | up to ~25 observed          |

If a customer needs tighter tolerance (≤8 ULP), the recommended path is
linear interpolation in the exp LUT — that drops exp error by a further
factor of ~step ≈ 1/64.

## Edge cases

- All-equal logits → all outputs are `≈ fp16(1/16) = 0x2C00`.
- Dominant logit (one lane far above the rest) → dominant lane `≈ 1.0`,
  others `≈ 0`.
- All-zero (or any constant) → same as all-equal.
- Subnormal input handling: subnormal `d` rounds to LUT index 0 → `e = 1`.
- NaN/Inf inputs are not expected on the DFL data path and are not
  required to produce meaningful softmax outputs. NaN propagates through
  the `fp16_add` helper; the final multiply via `fp16_fma` follows fp16
  rules (NaN → NaN canonical).

## Resource notes

- 1024 × 16 bit exp ROM + 1024 × 16 bit recip ROM = 4 KB total LUT.
- 16 instances of `fp16_fma` (one per lane) at the final stage.
- Several inline `fp16_add` and `fp16_max` instantiations inside the
  trees — each is a combinational function, no per-stage register cost
  beyond the explicit pipeline registers in `softmax16.sv`.

## DV

```
make -C hw/ip/softmax16/dv test     # 21/21 checks, lint clean
make -C hw/ip/softmax16/dv lint
```

- Tests directed edges (all-equal, one-hot dominant, all-very-negative,
  all-zero, gradient).
- 5000 random ±8 vectors and 1000 tight-range ±2 vectors.
- Each output lane is cross-checked against (a) the SV `_ref` (true
  softmax in `real` arithmetic, cast to fp16) and (b) an independent
  C++ double-precision shadow.
