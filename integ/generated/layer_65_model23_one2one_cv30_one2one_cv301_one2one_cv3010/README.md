# layer_65_model23_one2one_cv30_one2one_cv301_one2one_cv3010

Generated ordinary Conv+SiLU integration layer for `/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv_quant`.

- Shape: Cin=80 Cout=80 K=3 stride=1 pad=1
- Output: 80x80
- Generated DV wrapper factors: P_PIX=1, P_COUT=8, P_CIN=1, cycles~64000

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
