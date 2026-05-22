# yolo26n_top — frozen chip boundary

This is the **PD-facing top level**. The port list of `yolo26n_top.sv` is
frozen — once PD starts floorplanning we cannot change it.

All future integration (102 conv layers, SPPF, upsample, attention, detect
head, top-K, skip-FIFOs) lives inside `yolo26n_core.sv`. Its body grows;
its ports do not.

## Frozen ports

Single clock domain. `rst_ni` is async-assert / sync-deassert at the boundary
— PD inserts the `prim_rst_sync`.

### `s_axis_pix` — AXI4-Stream slave, image in
- 2 px/cycle, 3 ch u8 each (matches L0 `P_PIX=2`).
- `TDATA[47:0]` = `{pix1_b, pix1_g, pix1_r, pix0_b, pix0_g, pix0_r}`. `TDATA[63:48]` reserved.
- `TUSER[0]` = SOF, `TLAST` = EOL.
- Frame size implied by config (640×640 → 320 beats/row × 640 rows / 2 px/cycle … total 204 800 beats/frame).

### `m_axis_det` — AXI4-Stream master, detections out
- One detection per beat.
- `TDATA[6:0]` cls, `[23:8]` score (fp16), `[39:24]` x1, `[55:40]` y1, `[71:56]` x2, `[87:72]` y2 (all fp16).
- `TLAST` = last detection of the frame. With `CTRL.emit_empty=1`, an empty
  frame still emits one `TVALID & TLAST` beat with a zero payload so
  consumers can pipeline reliably.

### `s_axil` — AXI4-Lite slave, 12-bit addr, 32-bit data

| Offset | Reg     | Bits  | Field           | Notes                              |
| ------ | ------- | ----- | --------------- | ---------------------------------- |
| 0x000  | CTRL    | [0]   | start           | W1S pulse                          |
|        |         | [1]   | abort           | W1S pulse                          |
|        |         | [2]   | irq_en          |                                    |
|        |         | [3]   | emit_empty      |                                    |
|        |         | 23:16 | in_zp_a         | L0 input zero-point                |
| 0x004  | STATUS  | [0]   | busy            | RO                                 |
|        |         | [1]   | done            | W1C, also drives irq               |
|        |         | [2]   | err             | W1C                                |
|        |         | 15:8  | frame_id        | rolling 8-bit counter              |
| 0x008  | TOPK    | 15:0  | topk_slots      | default 300                        |
|        |         | 31:16 | score_thresh    | fp16                               |
| 0x00C  | SCRATCH | 31:0  | scratch         | RW debug                           |
| 0x010  | ID      | 31:0  | magic           | RO 0x594F4C4F ("YOLO")             |
| 0x014  | VERSION | 31:0  | {major, minor}  | RO, from `YOLO26N_VERSION_*` defs  |

### `irq_o`
Level-high, equal to `STATUS.done & CTRL.irq_en`.

## What's underneath today

`yolo26n_core.sv` is a skeletal pass-through: it accepts the pixel stream and
emits a single zero-detection terminator per frame. It exists today so PD can
synthesize the boundary, lint the AXI interfaces, and start floorplanning. It
is **not** an inference-correct datapath — that work tracks under
`TASKS.md` (SPPF, upsample, attn, detect head, top-level wiring of 102 layers).

## Conventions PD should rely on

- Port names match the AXI4 spec letter-for-letter; do not rename or repack.
- All AXIS payload fields are byte-aligned where possible and bit-stable.
- Reset is async-assert / sync-deassert; place the synchronizer at the pad.
- Single clock domain — no internal CDC at the boundary.
- No outstanding AXI-Lite transactions; one in-flight per channel.
