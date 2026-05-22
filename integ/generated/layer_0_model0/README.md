# layer_0_model0

Generated ordinary Conv+SiLU integration layer for `/model.0/conv/Conv_quant`.

- Shape: Cin=3 Cout=16 K=3 stride=2 pad=1
- Output: 320x320
- Generated DV wrapper factors: P_PIX=2, P_COUT=16, P_CIN=3, cycles~51200

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
