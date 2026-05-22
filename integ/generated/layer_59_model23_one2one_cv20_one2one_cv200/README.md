# layer_59_model23_one2one_cv20_one2one_cv200

Generated ordinary Conv+SiLU integration layer for `/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv_quant`.

- Shape: Cin=64 Cout=16 K=3 stride=1 pad=1
- Output: 80x80
- Generated DV wrapper factors: P_PIX=1, P_COUT=16, P_CIN=8, cycles~51200

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
