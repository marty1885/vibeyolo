# layer_23_model8_m0_cv1

Generated ordinary Conv+SiLU integration layer for `/model.8/m.0/cv1/conv/Conv_quant`.

- Shape: Cin=128 Cout=64 K=1 stride=1 pad=0
- Output: 20x20
- Generated DV wrapper factors: P_PIX=1, P_COUT=16, P_CIN=4, cycles~51200

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
