# layer_5_cv2 — micro-integration: YOLO26n `/model.2/cv2` Conv-BN-SiLU

End-to-end test that proves our verified SV IPs correctly implement L5
of YOLO26n — the **exit conv of the first C3k2 block** — when compared
against onnxruntime on the same model
(`integ/yolo26n/model_int8.onnx`).

## Slice

```
/model.2/Concat_output_0                 (fp32, 48 channels)
 → DynamicQuantizeLinear  (u8, s_a, zp_a)
 → ConvInteger            (48 → 64, K=1, stride=1, pad=0)
 → Cast i32 → fp32
 → Mul (s_a · s_w)
 → Add bias               (fp32)
 → Sigmoid → Mul (SiLU)
 → /model.2/cv2/act/Mul_output_0
```

Conv params: `Cin=48 Cout=64 K=1 stride=1 pad=0`. Input/output spatial is
160×160. Per-tensor symmetric weight scale (`s_w ≈ 0.01118`, `zp_w=0`).

The 48-channel input is the external concat of
`[cv1_first_half(16), cv1_second_half(16), m.0_output(16)]`. The DUT just
accepts a 48-lane i8 vector per beat — the concat is wired upstream.

For DV we drive an 8×8×48 input ROI taken from ORT's
`/model.2/Concat_output_0_quantized` and produce an 8×8×64 output ROI.
K=1 means **no padding ring** is needed.

## Files

```
extract.py                  # ORT-driven stimulus + reference generator
rtl/layer_5_cv2.sv          # DUT: 64-way parallel dotN/requant/silu pipeline
dv/layer_5_cv2_tb.sv        # Verilator wrapper (flattened ports)
dv/layer_5_cv2_test.cc      # C++ test: loads stim, drives DUT, scores
dv/Makefile                 # build/run
stim/                       # generated stimulus + ORT/HW reference (.hex)
```

## Design choices

- **Parameters from `scale_pkg::LAYER_5_*`.** `NCH_OUT=LAYER_5_COUT=64`
  and `N_LANE=LAYER_5_CIN*K*K=48*1*1=48`. Full-frame parallelism target
  is `P_PIX=1, P_COUT=64, P_CIN=16 → 76800 cycles/frame`
  (T_FRAME=100k, `LAYER_5_CYCLES`).

- **u8 → i8 fold (dynamic-quant bridge).** ORT emits per-tensor u8
  (`s_a`, `zp_a`); the SV pipeline expects i8 symmetric. We convert
  `i8 = clip(u8 − 128, −128, 127)` and rewrite the bias so the i8·i8
  dot reproduces ORT's ConvInteger arithmetic exactly:

  ```
  bias_eff[c] = bias[c] + s_a · s_w · (128 − zp_a) · sum_w[c]
  scale_fp16  = fp16(s_a · s_w / S_OUT_PRE)
  bias_fp16   = fp16(bias_eff[c]    / S_OUT_PRE)
  ```

  HW-ref matches ORT to 0.0000 max-abs on all 6 samples.

- **Architecture: 64 × dotN(N=48) + 64 × requant + 64 × act_silu.** One
  48-lane input vector per cycle drives all 64 output channels in
  parallel. Reuses the same dataflow as `stem_l0` / `layer_1`, just
  narrower per-channel (N=48) and wider in output (NCH_OUT=64).

- **K=1, linebuf trivial.** No `linebuf_kxk` instantiation. The 48-ch
  vector for the current pixel is enough — system-level wiring just
  needs to gate the concat into a 48-lane per-pixel stream.

- **SiLU scales.** Empirical pre-SiLU range on normalised 0..1 inputs is
  `[-0.28, +11.5]`. We pick `S_OUT_PRE = S_OUT_SILU = 12/127 ≈ 0.0945`
  to span the range with margin — LUT step ≈ 0.094, the dominant
  DUT-vs-ORT error.

## Cycle accounting

| Stage              | Latency                                |
| ------------------ | -------------------------------------- |
| dotN               | `1 + ⌈log₂48⌉ = 7` cycles              |
| requant            | 3 cycles                               |
| act_silu           | 1 cycle                                |
| **Total pipeline** | **11 cycles**                          |

| Quantity                              | Value                              |
| ------------------------------------- | ---------------------------------- |
| MAC ops / output pixel                | `48 · 64 = 3072` MAC               |
| Output channels per cycle             | 64 (parallel)                      |
| Cycles / output pixel (throughput)    | 1                                  |
| Cycles for the 8×8 ROI (64 pixels)    | `64 + 11 − 1 = 74`                 |
| MAC units instantiated by the DUT     | `48 · 64 = 3072`                   |

### Full-frame accounting at scale_pkg parallelism

`scale_pkg.sv` chooses `P_PIX=1, P_COUT=64, P_CIN=16` for L5
(`T_FRAME=100k`):

| Quantity                      | Value                                        |
| ----------------------------- | -------------------------------------------- |
| Output pixels per frame       | `160 · 160 = 25,600`                         |
| Cin-tiles per pixel           | `Cin·K·K / P_CIN = 48/16 = 3`                |
| Cout-tiles per pixel          | `Cout / P_COUT = 64/64 = 1`                  |
| Pixel-tiles                   | `25,600 / P_PIX = 25,600`                    |
| **Cycles per frame**          | `25,600 · 3 · 1 = 76,800`                    |
| **MAC units (P_COUT·P_CIN)**  | `64 · 16 = 1,024`                            |
| MACs per frame                | `1,024 · 76,800 = 78,643,200` (matches `M`)  |

The unit DV instantiates the full `P_COUT=64, P_CIN=48` shape (3072
MAC), which is a 3× wider Cin tile than the scale_pkg target. The math
folded into the DUT is the same — only the time-multiplex factor of the
production layout changes.

## Comparison criterion

Per sample: **PASS** iff `cosine_sim > 0.998` AND `mean_abs_err < 5% ·
out_range`.

## Result (latest run)

All 6 samples PASS. Aggregate: avg cos = **0.999809**, avg mae = 0.024,
worst max-abs = 0.120 (≈ one LUT step at `S_OUT_SILU = 12/127`).

```
sample      max_abs   mae      cos        out_range   thresh   verdict
rand0       0.1073    0.0212   0.999911   10.891      0.545    PASS
rand1       0.1054    0.0210   0.999911   11.319      0.566    PASS
half        0.0903    0.0277   0.999697    5.168      0.258    PASS
gradient    0.1200    0.0290   0.999617    4.890      0.244    PASS
rand_low    0.1016    0.0235   0.999867   10.750      0.538    PASS
rand_high   0.0996    0.0241   0.999852   10.497      0.525    PASS
```

`DUT vs HW-ref` is identical to `DUT vs ORT` to 4 decimal places —
confirming the entire DUT-vs-ORT gap comes from the SiLU LUT
quantisation, not from int conv or fp16 rounding.

## Running

```
make -C integ/layer_5_cv2/dv test          # builds verilator binary, runs test
make -C integ/layer_5_cv2/dv lint          # lint-only
```

`make test` regenerates `stim/` by re-running `extract.py` (which loads
the ONNX model via onnxruntime).
