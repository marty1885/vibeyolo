# layer_68_model23_one2one_cv30_one2one_cv301_one2one_cv3011

Generated ordinary Conv+SiLU integration layer for `/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv_quant`.

- Shape: Cin=80 Cout=80 K=1 stride=1 pad=0
- Output: 80x80
- Generated DV wrapper factors: P_PIX=2, P_COUT=16, P_CIN=8, cycles~160000

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
