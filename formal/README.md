# Formal verification of the leaf IPs

Every leaf IP ships a re-coded behavioral golden (`<ip>_ref.sv`) with an
identical port list to the synthesizable `<ip>.sv`. That is the setup for
**sequential equivalence checking (SEC)**: prove the RTL equals its golden for
*all* inputs and *all* cycles from reset — a far stronger claim than the
directed C++/Verilator DV the IPs ship with today.

## Toolchain (what's installed, and why it matters)

| Tool  | Role                                                                 |
|-------|----------------------------------------------------------------------|
| sv2v  | SV-2012 → Verilog-2005. yosys's built-in SV frontend rejects the SV the IPs use (unpacked function args, `void` fns, multi-packed ports). |
| yosys | `miter -equiv` builds the comparison and lowers it to a `$assert`.   |
| sby   | `prove` mode = BMC base case + unbounded temporal induction.         |
| z3    | Word-level SMT. Proves multiply/accumulate cones that bit-level SAT induction can't (see `mac8`). **No other SMT solver is installed.** |
| abc   | Bit-level IC3/PDR (sby `abc` engine), fallback for pure control.     |

## Usage

```sh
make -C formal sec-mac8         # SMT (z3) equivalence proof of one IP
make -C formal sat-i32_to_fp16  # bit-level SAT+induction proof of one IP
make -C formal -j4 equiv        # all SEC-able IPs (sby/z3) in parallel
make -C formal clean
```

Two engines, because no single one covers everything:

- **`sat-%`** — yosys bit-level SAT + temporal induction. Tolerant of the
  behavioral goldens' *unreset* pipeline registers, so it proves control /
  shift / saturate logic. Multiplier cones don't converge here.
- **`sec-%`** — sby + smtbmc + z3 (word-level). Proves datapath/multiply
  cones. Async-reset blocks need `INITZERO_<ip>` set (see Makefile) so the
  base case starts from the real post-reset state.

## Proven so far (unbounded equivalence, RTL ≡ golden ∀ inputs ∀ cycles)

| IP               | engine        | note |
|------------------|---------------|------|
| `i32_to_fp16`    | sat (k=2)     | int32→fp16, priority-encode + shift |
| `fp16_to_i8_sat` | sat           | fp16→int8 round/saturate |
| `mac8`           | sec / z3 (k=1)| int8 MAC; 16-bit product truncation proven lossless vs full-`int` golden |

### Numeric properties (`make -C formal props-mac8`)

Beyond equivalence, `mac8_props.sv` proves the accumulator's numeric safety by
k-induction (z3), carrying an exact 48-bit shadow accumulator + term counter:

- **`no_overflow`** — for up to **K=131071** multiply-accumulates since the
  last clr/reset, the int32 `acc_o` equals the *exact* integer sum
  (bit-for-bit) ⇒ never wraps. Unbounded proof.
- **tightness** — at **K=131072** the proof fails: the sum reaches 2³¹ and
  wraps. So 131071 is the exact safe depth (131072 × 16384 = 2³¹).
- **`clr_priority`** — synchronous clear overrides enable (loads the product).

mac8 is currently only instantiated inside `dotN` in clr-every-cycle mode
(K=1 in use), so this is headroom proof, not a live bug.

## Gates that bound the scope (discovered empirically)

1. **sv2v parse** — goldens using a `$`-system task fail sv2v 1f1b231
   (`upsample2_ref`, `skip_buf_ref`).
2. **yosys elaboration** — `while`/recursion-style functions can't be
   synthesized (`reduce_max_n` DUT's `smax_reduce`). Needs a static-bound
   rewrite to be formalizable.
3. **`real`-valued goldens are simulation refs, not formal models.** SMT
   cannot reason about IEEE `real`, so these IPs cannot be SEC'd against
   their golden — only property-checked: `add_rq`, `act_silu`, `act_sigmoid`,
   `dequant_n`, `box_affine`, `box_decode`, `softmax16`, `topk_fp16`.
4. **Reset modeling** — behavioral goldens leave pipeline regs unreset; SEC
   needs init pinning (`mac8`) or an explicit reset-sequence assumption.
5. **Protocol** — streaming/handshake blocks are equivalent only under a
   valid ready/valid protocol; the unconstrained miter yields spurious
   counterexamples (`concat_mux`, `linebuf_kxk`, `skip_buf`). Needs `assume`.
6. **Solver tractability** — fp16 mantissa-multiply miters (`fp16_fma`,
   `fp16_macw`, `dotN`) and large stateful blocks (`softmax16`, `topk_fp16`,
   `flash_attn`) are low-yield for unbounded SEC; use bounded BMC + targeted
   property checks instead.
7. **Submodule deps** — `requant` instantiates the already-proven
   `fp16_to_i8_sat`; gather the dep or blackbox it (compositional proof).

## Tier map (all 21 leaf IPs)

- **A — provable now:** `i32_to_fp16`✓ `fp16_to_i8_sat`✓ `mac8`✓
- **B — small refinement (dep-gather / fn-rewrite):** `requant`,
  `maxpool_kxk`, `reduce_max_n`
- **C — needs protocol assumptions:** `concat_mux`, `linebuf_kxk`,
  `skip_buf`, `upsample2`
- **D — property-only (real golden):** `add_rq`, `act_silu`, `act_sigmoid`,
  `dequant_n`, `box_affine`, `box_decode`, `softmax16`, `topk_fp16`
- **E — bounded/property only (fp-mult / big state):** `fp16_fma`,
  `fp16_macw`, `dotN`, `flash_attn`
