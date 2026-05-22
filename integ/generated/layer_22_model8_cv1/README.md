# layer_22_model8_cv1

Generated ordinary Conv+SiLU integration layer for `/model.8/cv1/conv/Conv_quant`.

- Shape: Cin=256 Cout=256 K=1 stride=1 pad=0
- Output: 20x20
- Generated DV wrapper factors: P_PIX=2, P_COUT=16, P_CIN=8, cycles~102400

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
