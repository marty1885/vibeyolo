# layer_78_model23_one2one_cv31_one2one_cv310_one2one_cv3100

Generated ordinary Conv+SiLU integration layer for `/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv_quant`.

- Shape: Cin=128 Cout=128 K=3 stride=1 pad=1
- Output: 40x40
- Generated DV wrapper factors: P_PIX=1, P_COUT=4, P_CIN=1, cycles~51200

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
