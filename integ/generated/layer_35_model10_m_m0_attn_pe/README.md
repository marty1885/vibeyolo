# layer_35_model10_m_m0_attn_pe

Generated ordinary Conv+SiLU integration layer for `/model.10/m/m.0/attn/pe/conv/Conv_quant`.

- Shape: Cin=128 Cout=128 K=3 stride=1 pad=1
- Output: 20x20
- Generated DV wrapper factors: P_PIX=1, P_COUT=1, P_CIN=1, cycles~51200

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
