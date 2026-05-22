# layer_94_model23_one2one_cv22_one2one_cv220

Generated ordinary Conv+SiLU integration layer for `/model.23/one2one_cv2.2/one2one_cv2.2.0/conv/Conv_quant`.

- Shape: Cin=256 Cout=16 K=3 stride=1 pad=1
- Output: 20x20
- Generated DV wrapper factors: P_PIX=1, P_COUT=16, P_CIN=2, cycles~51200

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
