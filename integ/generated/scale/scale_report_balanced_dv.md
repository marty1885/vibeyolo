# Balanced scale report -- YOLO26n (T_FRAME = 100000)

The optimizer minimizes `P_PIX * P_COUT * P_CIN` subject to `cycles <= T_FRAME`.
Grouped/depthwise conv MACs use `Cin/group`. Ties prefer stages closer to the target.

- `max_p_pix`: 2
- `cout_cap`: 16
- `cin_cap`: 8

| idx | name | Cin | Cout | K | HxW | P_PIX | P_COUT | P_CIN | area | cycles | meets |
|-----|------|----:|-----:|--:|-----|------:|-------:|------:|-----:|-------:|:------|
| 0 | /model.0/conv/Conv_quant | 3 | 16 | 3 | 320x320 | 2 | 16 | 3 | 96 | 51200 | yes |
| 1 | /model.1/conv/Conv_quant | 16 | 32 | 3 | 160x160 | 2 | 16 | 8 | 256 | 51200 | yes |
| 2 | /model.2/cv1/conv/Conv_quant | 32 | 32 | 1 | 160x160 | 2 | 16 | 8 | 256 | 102400 | no |
| 3 | /model.2/m.0/cv1/conv/Conv_quant | 16 | 8 | 3 | 160x160 | 1 | 8 | 8 | 64 | 51200 | yes |
| 4 | /model.2/m.0/cv2/conv/Conv_quant | 8 | 16 | 3 | 160x160 | 1 | 16 | 4 | 64 | 51200 | yes |
| 5 | /model.2/cv2/conv/Conv_quant | 48 | 64 | 1 | 160x160 | 2 | 16 | 8 | 256 | 307200 | no |
| 6 | /model.3/conv/Conv_quant | 64 | 64 | 3 | 80x80 | 2 | 16 | 8 | 256 | 102400 | no |
| 7 | /model.4/cv1/conv/Conv_quant | 64 | 64 | 1 | 80x80 | 2 | 16 | 8 | 256 | 102400 | no |
| 8 | /model.4/m.0/cv1/conv/Conv_quant | 32 | 16 | 3 | 80x80 | 1 | 16 | 4 | 64 | 51200 | yes |
| 9 | /model.4/m.0/cv2/conv/Conv_quant | 16 | 32 | 3 | 80x80 | 1 | 16 | 4 | 64 | 51200 | yes |
| 10 | /model.4/cv2/conv/Conv_quant | 96 | 128 | 1 | 80x80 | 2 | 16 | 8 | 256 | 307200 | no |
| 11 | /model.5/conv/Conv_quant | 128 | 128 | 3 | 40x40 | 2 | 16 | 8 | 256 | 102400 | no |
| 12 | /model.6/cv1/conv/Conv_quant | 128 | 128 | 1 | 40x40 | 2 | 16 | 8 | 256 | 102400 | no |
| 13 | /model.6/m.0/cv1/conv/Conv_quant | 64 | 32 | 1 | 40x40 | 1 | 16 | 4 | 64 | 51200 | yes |
| 14 | /model.6/m.0/cv2/conv/Conv_quant | 64 | 32 | 1 | 40x40 | 1 | 16 | 4 | 64 | 51200 | yes |
| 15 | /model.6/m.0/m/m.0/cv1/conv/Conv_quant | 32 | 32 | 3 | 40x40 | 1 | 16 | 2 | 32 | 51200 | yes |
| 16 | /model.6/m.0/m/m.0/cv2/conv/Conv_quant | 32 | 32 | 3 | 40x40 | 1 | 16 | 2 | 32 | 51200 | yes |
| 17 | /model.6/m.0/m/m.1/cv1/conv/Conv_quant | 32 | 32 | 3 | 40x40 | 1 | 16 | 2 | 32 | 51200 | yes |
| 18 | /model.6/m.0/m/m.1/cv2/conv/Conv_quant | 32 | 32 | 3 | 40x40 | 1 | 16 | 2 | 32 | 51200 | yes |
| 19 | /model.6/m.0/cv3/conv/Conv_quant | 64 | 64 | 1 | 40x40 | 1 | 16 | 8 | 128 | 51200 | yes |
| 20 | /model.6/cv2/conv/Conv_quant | 192 | 128 | 1 | 40x40 | 2 | 16 | 8 | 256 | 153600 | no |
| 21 | /model.7/conv/Conv_quant | 128 | 256 | 3 | 20x20 | 2 | 16 | 8 | 256 | 51200 | yes |
| 22 | /model.8/cv1/conv/Conv_quant | 256 | 256 | 1 | 20x20 | 2 | 16 | 8 | 256 | 102400 | no |
| 23 | /model.8/m.0/cv1/conv/Conv_quant | 128 | 64 | 1 | 20x20 | 1 | 16 | 4 | 64 | 51200 | yes |
| 24 | /model.8/m.0/cv2/conv/Conv_quant | 128 | 64 | 1 | 20x20 | 1 | 16 | 4 | 64 | 51200 | yes |
| 25 | /model.8/m.0/m/m.0/cv1/conv/Conv_quant | 64 | 64 | 3 | 20x20 | 1 | 16 | 2 | 32 | 51200 | yes |
| 26 | /model.8/m.0/m/m.0/cv2/conv/Conv_quant | 64 | 64 | 3 | 20x20 | 1 | 16 | 2 | 32 | 51200 | yes |
| 27 | /model.8/m.0/m/m.1/cv1/conv/Conv_quant | 64 | 64 | 3 | 20x20 | 1 | 16 | 2 | 32 | 51200 | yes |
| 28 | /model.8/m.0/m/m.1/cv2/conv/Conv_quant | 64 | 64 | 3 | 20x20 | 1 | 16 | 2 | 32 | 51200 | yes |
| 29 | /model.8/m.0/cv3/conv/Conv_quant | 128 | 128 | 1 | 20x20 | 1 | 16 | 8 | 128 | 51200 | yes |
| 30 | /model.8/cv2/conv/Conv_quant | 384 | 256 | 1 | 20x20 | 2 | 16 | 8 | 256 | 153600 | no |
| 31 | /model.9/cv1/conv/Conv_quant | 256 | 128 | 1 | 20x20 | 2 | 16 | 8 | 256 | 51200 | yes |
| 32 | /model.9/cv2/conv/Conv_quant | 512 | 256 | 1 | 20x20 | 2 | 16 | 8 | 256 | 204800 | no |
| 33 | /model.10/cv1/conv/Conv_quant | 256 | 256 | 1 | 20x20 | 2 | 16 | 8 | 256 | 102400 | no |
| 34 | /model.10/m/m.0/attn/qkv/conv/Conv_quant | 128 | 256 | 1 | 20x20 | 2 | 16 | 8 | 256 | 51200 | yes |
| 35 | /model.10/m/m.0/attn/pe/conv/Conv_quant | 128 | 128 | 3 | 20x20 | 1 | 1 | 1 | 1 | 51200 | yes |
| 36 | /model.10/m/m.0/attn/proj/conv/Conv_quant | 128 | 128 | 1 | 20x20 | 1 | 16 | 8 | 128 | 51200 | yes |
| 37 | /model.10/m/m.0/ffn/ffn.0/conv/Conv_quant | 128 | 256 | 1 | 20x20 | 2 | 16 | 8 | 256 | 51200 | yes |
| 38 | /model.10/m/m.0/ffn/ffn.1/conv/Conv_quant | 256 | 128 | 1 | 20x20 | 2 | 16 | 8 | 256 | 51200 | yes |
| 39 | /model.10/cv2/conv/Conv_quant | 256 | 256 | 1 | 20x20 | 2 | 16 | 8 | 256 | 102400 | no |
| 40 | /model.13/cv1/conv/Conv_quant | 384 | 128 | 1 | 40x40 | 2 | 16 | 8 | 256 | 307200 | no |
| 41 | /model.13/m.0/cv1/conv/Conv_quant | 64 | 32 | 1 | 40x40 | 1 | 16 | 4 | 64 | 51200 | yes |
| 42 | /model.13/m.0/cv2/conv/Conv_quant | 64 | 32 | 1 | 40x40 | 1 | 16 | 4 | 64 | 51200 | yes |
| 43 | /model.13/m.0/m/m.0/cv1/conv/Conv_quant | 32 | 32 | 3 | 40x40 | 1 | 16 | 2 | 32 | 51200 | yes |
| 44 | /model.13/m.0/m/m.0/cv2/conv/Conv_quant | 32 | 32 | 3 | 40x40 | 1 | 16 | 2 | 32 | 51200 | yes |
| 45 | /model.13/m.0/m/m.1/cv1/conv/Conv_quant | 32 | 32 | 3 | 40x40 | 1 | 16 | 2 | 32 | 51200 | yes |
| 46 | /model.13/m.0/m/m.1/cv2/conv/Conv_quant | 32 | 32 | 3 | 40x40 | 1 | 16 | 2 | 32 | 51200 | yes |
| 47 | /model.13/m.0/cv3/conv/Conv_quant | 64 | 64 | 1 | 40x40 | 1 | 16 | 8 | 128 | 51200 | yes |
| 48 | /model.13/cv2/conv/Conv_quant | 192 | 128 | 1 | 40x40 | 2 | 16 | 8 | 256 | 153600 | no |
| 49 | /model.16/cv1/conv/Conv_quant | 256 | 64 | 1 | 80x80 | 2 | 16 | 8 | 256 | 409600 | no |
| 50 | /model.16/m.0/cv1/conv/Conv_quant | 32 | 16 | 1 | 80x80 | 1 | 16 | 4 | 64 | 51200 | yes |
| 51 | /model.16/m.0/cv2/conv/Conv_quant | 32 | 16 | 1 | 80x80 | 1 | 16 | 4 | 64 | 51200 | yes |
| 52 | /model.16/m.0/m/m.0/cv1/conv/Conv_quant | 16 | 16 | 3 | 80x80 | 1 | 16 | 2 | 32 | 51200 | yes |
| 53 | /model.16/m.0/m/m.0/cv2/conv/Conv_quant | 16 | 16 | 3 | 80x80 | 1 | 16 | 2 | 32 | 51200 | yes |
| 54 | /model.16/m.0/m/m.1/cv1/conv/Conv_quant | 16 | 16 | 3 | 80x80 | 1 | 16 | 2 | 32 | 51200 | yes |
| 55 | /model.16/m.0/m/m.1/cv2/conv/Conv_quant | 16 | 16 | 3 | 80x80 | 1 | 16 | 2 | 32 | 51200 | yes |
| 56 | /model.16/m.0/cv3/conv/Conv_quant | 32 | 32 | 1 | 80x80 | 1 | 16 | 8 | 128 | 51200 | yes |
| 57 | /model.16/cv2/conv/Conv_quant | 96 | 64 | 1 | 80x80 | 2 | 16 | 8 | 256 | 153600 | no |
| 58 | /model.17/conv/Conv_quant | 64 | 64 | 3 | 40x40 | 1 | 16 | 8 | 128 | 51200 | yes |
| 59 | /model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv_quant | 64 | 16 | 3 | 80x80 | 1 | 16 | 8 | 128 | 51200 | yes |
| 60 | /model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv_quant | 64 | 64 | 3 | 80x80 | 1 | 8 | 1 | 8 | 51200 | yes |
| 61 | /model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv_quant | 16 | 16 | 3 | 80x80 | 1 | 16 | 2 | 32 | 51200 | yes |
| 62 | /model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/conv/Conv_quant | 64 | 80 | 1 | 80x80 | 2 | 16 | 8 | 256 | 128000 | no |
| 63 | /model.19/cv1/conv/Conv_quant | 192 | 128 | 1 | 40x40 | 2 | 16 | 8 | 256 | 153600 | no |
| 64 | /model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_quant | 16 | 4 | 1 | 80x80 | 1 | 4 | 2 | 8 | 51200 | yes |
| 65 | /model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv_quant | 80 | 80 | 3 | 80x80 | 1 | 8 | 1 | 8 | 64000 | yes |
| 66 | /model.19/m.0/cv1/conv/Conv_quant | 64 | 32 | 1 | 40x40 | 1 | 16 | 4 | 64 | 51200 | yes |
| 67 | /model.19/m.0/cv2/conv/Conv_quant | 64 | 32 | 1 | 40x40 | 1 | 16 | 4 | 64 | 51200 | yes |
| 68 | /model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv_quant | 80 | 80 | 1 | 80x80 | 2 | 16 | 8 | 256 | 160000 | no |
| 69 | /model.19/m.0/m/m.0/cv1/conv/Conv_quant | 32 | 32 | 3 | 40x40 | 1 | 16 | 2 | 32 | 51200 | yes |
| 70 | /model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_quant | 80 | 80 | 1 | 80x80 | 2 | 16 | 8 | 256 | 160000 | no |
| 71 | /model.19/m.0/m/m.0/cv2/conv/Conv_quant | 32 | 32 | 3 | 40x40 | 1 | 16 | 2 | 32 | 51200 | yes |
| 72 | /model.19/m.0/m/m.1/cv1/conv/Conv_quant | 32 | 32 | 3 | 40x40 | 1 | 16 | 2 | 32 | 51200 | yes |
| 73 | /model.19/m.0/m/m.1/cv2/conv/Conv_quant | 32 | 32 | 3 | 40x40 | 1 | 16 | 2 | 32 | 51200 | yes |
| 74 | /model.19/m.0/cv3/conv/Conv_quant | 64 | 64 | 1 | 40x40 | 1 | 16 | 8 | 128 | 51200 | yes |
| 75 | /model.19/cv2/conv/Conv_quant | 192 | 128 | 1 | 40x40 | 2 | 16 | 8 | 256 | 153600 | no |
| 76 | /model.20/conv/Conv_quant | 128 | 128 | 3 | 20x20 | 1 | 16 | 8 | 128 | 51200 | yes |
| 77 | /model.23/one2one_cv2.1/one2one_cv2.1.0/conv/Conv_quant | 128 | 16 | 3 | 40x40 | 1 | 16 | 4 | 64 | 51200 | yes |
| 78 | /model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv_quant | 128 | 128 | 3 | 40x40 | 1 | 4 | 1 | 4 | 51200 | yes |
| 79 | /model.23/one2one_cv2.1/one2one_cv2.1.1/conv/Conv_quant | 16 | 16 | 3 | 40x40 | 1 | 8 | 1 | 8 | 51200 | yes |
| 80 | /model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/conv/Conv_quant | 128 | 80 | 1 | 40x40 | 2 | 16 | 8 | 256 | 64000 | yes |
| 81 | /model.22/cv1/conv/Conv_quant | 384 | 256 | 1 | 20x20 | 2 | 16 | 8 | 256 | 153600 | no |
| 82 | /model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_quant | 16 | 4 | 1 | 40x40 | 1 | 2 | 1 | 2 | 51200 | yes |
| 83 | /model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv_quant | 80 | 80 | 3 | 40x40 | 1 | 2 | 1 | 2 | 64000 | yes |
| 84 | /model.22/m.0/m.0.0/cv1/conv/Conv_quant | 128 | 64 | 3 | 20x20 | 1 | 16 | 4 | 64 | 51200 | yes |
| 85 | /model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv_quant | 80 | 80 | 1 | 40x40 | 1 | 16 | 8 | 128 | 80000 | yes |
| 86 | /model.22/m.0/m.0.0/cv2/conv/Conv_quant | 64 | 128 | 3 | 20x20 | 1 | 16 | 4 | 64 | 51200 | yes |
| 87 | /model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_quant | 80 | 80 | 1 | 40x40 | 1 | 16 | 8 | 128 | 80000 | yes |
| 88 | /model.22/m.0/m.0.1/attn/qkv/conv/Conv_quant | 128 | 256 | 1 | 20x20 | 2 | 16 | 8 | 256 | 51200 | yes |
| 89 | /model.22/m.0/m.0.1/attn/pe/conv/Conv_quant | 128 | 128 | 3 | 20x20 | 1 | 1 | 1 | 1 | 51200 | yes |
| 90 | /model.22/m.0/m.0.1/attn/proj/conv/Conv_quant | 128 | 128 | 1 | 20x20 | 1 | 16 | 8 | 128 | 51200 | yes |
| 91 | /model.22/m.0/m.0.1/ffn/ffn.0/conv/Conv_quant | 128 | 256 | 1 | 20x20 | 2 | 16 | 8 | 256 | 51200 | yes |
| 92 | /model.22/m.0/m.0.1/ffn/ffn.1/conv/Conv_quant | 256 | 128 | 1 | 20x20 | 2 | 16 | 8 | 256 | 51200 | yes |
| 93 | /model.22/cv2/conv/Conv_quant | 384 | 256 | 1 | 20x20 | 2 | 16 | 8 | 256 | 153600 | no |
| 94 | /model.23/one2one_cv2.2/one2one_cv2.2.0/conv/Conv_quant | 256 | 16 | 3 | 20x20 | 1 | 16 | 2 | 32 | 51200 | yes |
| 95 | /model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/conv/Conv_quant | 256 | 256 | 3 | 20x20 | 1 | 2 | 1 | 2 | 51200 | yes |
| 96 | /model.23/one2one_cv2.2/one2one_cv2.2.1/conv/Conv_quant | 16 | 16 | 3 | 20x20 | 1 | 2 | 1 | 2 | 51200 | yes |
| 97 | /model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/conv/Conv_quant | 256 | 80 | 1 | 20x20 | 1 | 16 | 8 | 128 | 64000 | yes |
| 98 | /model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_quant | 16 | 4 | 1 | 20x20 | 1 | 1 | 1 | 1 | 25600 | yes |
| 99 | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv_quant | 80 | 80 | 3 | 20x20 | 1 | 1 | 1 | 1 | 32000 | yes |
| 100 | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv_quant | 80 | 80 | 1 | 20x20 | 1 | 16 | 2 | 32 | 80000 | yes |
| 101 | /model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_quant | 80 | 80 | 1 | 20x20 | 1 | 16 | 2 | 32 | 80000 | yes |

**Total area proxy:** 12,368
**Worst cycles:** 409,600
**Target misses:** 24
