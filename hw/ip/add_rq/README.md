# add_rq

Residual-add requantizer. Adds two int8 streams that carry independent
per-channel fp16 scales and emits a single requantized int8.

Used wherever a YOLO26n graph performs a skip-connection / residual add
between two quantized tensors whose scales were chosen independently.

## Math

```
fp_a   = i32_to_fp16(sign_extend(a_i8))
fp_b   = i32_to_fp16(sign_extend(b_i8))
ta     = fp16(fp_a * scale_a)
tb     = fp16(fp_b * scale_b)
sum    = fp16(ta + tb)
y_fp16 = fp16(sum * inv_out_scale + bias)
y_o    = sat_i8( round_rne( y_fp16 ) )
```

Each fp16 result is the IEEE binary16 RNE rounding of the unrounded
real-number expression; this matches the semantics of the underlying
`fp16_fma` block.

## Ports

| Port                   | Dir | Width  | Description                                |
|------------------------|-----|--------|--------------------------------------------|
| `clk_i`                | in  | 1      | Clock, rising edge.                        |
| `rst_ni`               | in  | 1      | Async-assert, sync-deassert active-low.    |
| `valid_i`              | in  | 1      | Stimulus valid this cycle.                 |
| `a_i8_i`               | in  | 8 (s)  | int8 sample from stream A.                 |
| `b_i8_i`               | in  | 8 (s)  | int8 sample from stream B.                 |
| `scale_a_fp16_i`       | in  | 16     | fp16 per-channel scale for stream A.       |
| `scale_b_fp16_i`       | in  | 16     | fp16 per-channel scale for stream B.       |
| `inv_out_scale_fp16_i` | in  | 16     | fp16 reciprocal of output scale.           |
| `bias_fp16_i`          | in  | 16     | fp16 bias added before final i8 cast.      |
| `valid_o`              | out | 1      | Output valid (5 cycles after `valid_i`).   |
| `y_o`                  | out | 8 (s)  | Requantized int8 result.                   |

## Pipeline

5 cycles end-to-end. Each sub-block is a registered single-cycle stage:

| Cycle | Stage                              | Function                       |
|-------|------------------------------------|--------------------------------|
| 1     | `i32_to_fp16` × 2 (parallel)       | int8 → fp16 dequant            |
| 2     | `fp16_fma` × 2 (parallel)          | `ta = fp_a*sa`, `tb = fp_b*sb` |
| 3     | `fp16_fma(ta, 1.0, tb)`            | fp16 add                       |
| 4     | `fp16_fma(sum, inv_out, bias)`     | output requant FMA             |
| 5     | `fp16_to_i8_sat`                   | round + saturate to int8       |

The scales/bias inputs are pipelined alongside the data so the caller
only needs to present them on the same cycle as `a_i8_i` / `b_i8_i`.

## Composition

Instantiated sub-blocks (already DV-clean):

- `hw/ip/i32_to_fp16/`
- `hw/ip/fp16_fma/`
- `hw/ip/fp16_to_i8_sat/`

No dependency on `requant` (which is a sibling task).

## DV / cross-check

```
make -C hw/ip/add_rq/dv test     # build + lockstep DUT vs REF
make -C hw/ip/add_rq/dv lint
```

The TB drives DUT and REF (`add_rq_ref`) in lockstep on identical
stimulus and flags any cycle-level disagreement. The REF is an
independent algorithm (SystemVerilog `shortreal` arithmetic with its
own fp16 round-trip code) so a shared bug across DUT and REF is
implausible. The C++ test additionally compares against a `double`-
precision integer-mantissa shadow with a third, independent fp16
rounding implementation; that gives three independent paths.
