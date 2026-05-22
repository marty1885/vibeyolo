# layer_49_model16_cv1

Generated ordinary Conv+SiLU integration layer for `/model.16/cv1/conv/Conv_quant`.

- Shape: Cin=256 Cout=64 K=1 stride=1 pad=0
- Output: 80x80
- Generated DV wrapper factors: P_PIX=2, P_COUT=16, P_CIN=8, cycles~409600

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
