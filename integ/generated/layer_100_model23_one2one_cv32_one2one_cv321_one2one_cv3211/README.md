# layer_100_model23_one2one_cv32_one2one_cv321_one2one_cv3211

Generated ordinary Conv+SiLU integration layer for `/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv_quant`.

- Shape: Cin=80 Cout=80 K=1 stride=1 pad=0
- Output: 20x20
- Generated DV wrapper factors: P_PIX=1, P_COUT=16, P_CIN=2, cycles~80000

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
