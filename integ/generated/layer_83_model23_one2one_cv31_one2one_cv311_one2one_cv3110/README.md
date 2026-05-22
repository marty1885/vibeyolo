# layer_83_model23_one2one_cv31_one2one_cv311_one2one_cv3110

Generated ordinary Conv+SiLU integration layer for `/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv_quant`.

- Shape: Cin=80 Cout=80 K=3 stride=1 pad=1
- Output: 40x40
- Generated DV wrapper factors: P_PIX=1, P_COUT=2, P_CIN=1, cycles~64000

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
