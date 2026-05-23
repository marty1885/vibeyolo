# dequant_n

N-lane **int8 → fp16 dequantization** with a shared fp16 scale:
`y[i] = fp16(int8(x[i]) * scale)`. One vector/cycle, latency 2.

## Why

The detect head gathers the 80 class **logits** of each of the 300 selected
anchors and dequantizes them to fp16 for the model output, using that anchor's
per-tensor cls `S_OUT` (which differs per scale — see the detect-head notes).
`N=80` here; parameterized for reuse.

Each lane = `i32_to_fp16` (the int8 widens losslessly) → `fp16_fma(·, scale, 0)`.
Because `fp16(int8)` is **exact**, the fused product equals
`fp16(exact_i8 × scale_fp16)` — so unlike the fp16-accumulating box path this is
**bit-exact**, and the DV asserts 0 ULP.

## Interface

`en_i` + `x_i` (N×int8) + `scale_i` (fp16) → after 2 cycles `valid_o` +
`y_o` (N×fp16). Parameter `N` (default 80).

## DV

`make -C dv test` builds N=80 and N=16. **Both pass 4/4** (~17 k random vectors
each across representative cls scales + random scales): 0 dut-vs-ref and 0
dut-vs-C++-shadow mismatches — fully bit-exact, as expected for exact int8
widening followed by a single fused multiply.
