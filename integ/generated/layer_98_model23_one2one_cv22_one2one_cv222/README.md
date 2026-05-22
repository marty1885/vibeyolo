# layer_98_model23_one2one_cv22_one2one_cv222

Generated ordinary Conv+SiLU integration layer for `/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_quant`.

- Shape: Cin=16 Cout=4 K=1 stride=1 pad=0
- Output: 20x20
- Generated DV wrapper factors: P_PIX=1, P_COUT=1, P_CIN=1, cycles~25600

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
