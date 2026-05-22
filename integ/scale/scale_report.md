# Scale report — YOLO26n  (T_FRAME = 100000 cyc/frame)

Layer count: **102** ConvInteger ops.

Heuristic: fully unroll P_COUT, then expand P_CIN inside the input window, then P_PIX.
"MAC cost" = P_PIX * P_COUT * P_CIN (total int8 multipliers instantiated).

| idx | name | Cin | Cout | K | s | H_out | W_out | M_i | P_PIX | P_COUT | P_CIN | MAC cost | cycles_i |
|-----|------|----:|-----:|--:|--:|------:|------:|----:|------:|-------:|------:|---------:|---------:|
| 0 | /model.0/conv/Conv_quant | 3 | 16 | 3 | 2 | 320 | 320 | 44236800 | 2 | 16 | 3 | 96 | 51200 |
| 1 | /model.1/conv/Conv_quant | 16 | 32 | 3 | 2 | 160 | 160 | 117964800 | 1 | 32 | 8 | 256 | 51200 |
| 2 | /model.2/cv1/conv/Conv_quant | 32 | 32 | 1 | 1 | 160 | 160 | 26214400 | 1 | 32 | 16 | 512 | 51200 |
| 3 | /model.2/m.0/cv1/conv/Conv_quant | 16 | 8 | 3 | 1 | 160 | 160 | 29491200 | 1 | 8 | 8 | 64 | 51200 |
| 4 | /model.2/m.0/cv2/conv/Conv_quant | 8 | 16 | 3 | 1 | 160 | 160 | 29491200 | 1 | 16 | 4 | 64 | 51200 |
| 5 | /model.2/cv2/conv/Conv_quant | 48 | 64 | 1 | 1 | 160 | 160 | 78643200 | 1 | 64 | 16 | 1024 | 76800 |
| 6 | /model.3/conv/Conv_quant | 64 | 64 | 3 | 2 | 80 | 80 | 235929600 | 1 | 64 | 8 | 512 | 51200 |
| 7 | /model.4/cv1/conv/Conv_quant | 64 | 64 | 1 | 1 | 80 | 80 | 26214400 | 1 | 64 | 8 | 512 | 51200 |
| 8 | /model.4/m.0/cv1/conv/Conv_quant | 32 | 16 | 3 | 1 | 80 | 80 | 29491200 | 1 | 16 | 4 | 64 | 51200 |
| 9 | /model.4/m.0/cv2/conv/Conv_quant | 16 | 32 | 3 | 1 | 80 | 80 | 29491200 | 1 | 32 | 2 | 64 | 51200 |
| 10 | /model.4/cv2/conv/Conv_quant | 96 | 128 | 1 | 1 | 80 | 80 | 78643200 | 1 | 128 | 8 | 1024 | 76800 |
| 11 | /model.5/conv/Conv_quant | 128 | 128 | 3 | 2 | 40 | 40 | 235929600 | 1 | 128 | 4 | 512 | 51200 |
| 12 | /model.6/cv1/conv/Conv_quant | 128 | 128 | 1 | 1 | 40 | 40 | 26214400 | 1 | 128 | 4 | 512 | 51200 |
| 13 | /model.6/m.0/cv1/conv/Conv_quant | 64 | 32 | 1 | 1 | 40 | 40 | 3276800 | 1 | 32 | 2 | 64 | 51200 |
| 14 | /model.6/m.0/cv2/conv/Conv_quant | 64 | 32 | 1 | 1 | 40 | 40 | 3276800 | 1 | 32 | 2 | 64 | 51200 |
| 15 | /model.6/m.0/m/m.0/cv1/conv/Conv_quant | 32 | 32 | 3 | 1 | 40 | 40 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 16 | /model.6/m.0/m/m.0/cv2/conv/Conv_quant | 32 | 32 | 3 | 1 | 40 | 40 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 17 | /model.6/m.0/m/m.1/cv1/conv/Conv_quant | 32 | 32 | 3 | 1 | 40 | 40 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 18 | /model.6/m.0/m/m.1/cv2/conv/Conv_quant | 32 | 32 | 3 | 1 | 40 | 40 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 19 | /model.6/m.0/cv3/conv/Conv_quant | 64 | 64 | 1 | 1 | 40 | 40 | 6553600 | 1 | 64 | 2 | 128 | 51200 |
| 20 | /model.6/cv2/conv/Conv_quant | 192 | 128 | 1 | 1 | 40 | 40 | 39321600 | 1 | 128 | 4 | 512 | 76800 |
| 21 | /model.7/conv/Conv_quant | 128 | 256 | 3 | 2 | 20 | 20 | 117964800 | 1 | 256 | 1 | 256 | 51200 |
| 22 | /model.8/cv1/conv/Conv_quant | 256 | 256 | 1 | 1 | 20 | 20 | 26214400 | 1 | 256 | 2 | 512 | 51200 |
| 23 | /model.8/m.0/cv1/conv/Conv_quant | 128 | 64 | 1 | 1 | 20 | 20 | 3276800 | 1 | 64 | 1 | 64 | 51200 |
| 24 | /model.8/m.0/cv2/conv/Conv_quant | 128 | 64 | 1 | 1 | 20 | 20 | 3276800 | 1 | 64 | 1 | 64 | 51200 |
| 25 | /model.8/m.0/m/m.0/cv1/conv/Conv_quant | 64 | 64 | 3 | 1 | 20 | 20 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 26 | /model.8/m.0/m/m.0/cv2/conv/Conv_quant | 64 | 64 | 3 | 1 | 20 | 20 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 27 | /model.8/m.0/m/m.1/cv1/conv/Conv_quant | 64 | 64 | 3 | 1 | 20 | 20 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 28 | /model.8/m.0/m/m.1/cv2/conv/Conv_quant | 64 | 64 | 3 | 1 | 20 | 20 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 29 | /model.8/m.0/cv3/conv/Conv_quant | 128 | 128 | 1 | 1 | 20 | 20 | 6553600 | 1 | 128 | 1 | 128 | 51200 |
| 30 | /model.8/cv2/conv/Conv_quant | 384 | 256 | 1 | 1 | 20 | 20 | 39321600 | 1 | 256 | 2 | 512 | 76800 |
| 31 | /model.9/cv1/conv/Conv_quant | 256 | 128 | 1 | 1 | 20 | 20 | 13107200 | 1 | 128 | 2 | 256 | 51200 |
| 32 | /model.9/cv2/conv/Conv_quant | 512 | 256 | 1 | 1 | 20 | 20 | 52428800 | 1 | 256 | 4 | 1024 | 51200 |
| 33 | /model.10/cv1/conv/Conv_quant | 256 | 256 | 1 | 1 | 20 | 20 | 26214400 | 1 | 256 | 2 | 512 | 51200 |
| 34 | /model.10/m/m.0/attn/qkv/conv/Conv_quant | 128 | 256 | 1 | 1 | 20 | 20 | 13107200 | 1 | 256 | 1 | 256 | 51200 |
| 35 | /model.10/m/m.0/attn/pe/conv/Conv_quant | 128 | 128 | 3 | 1 | 20 | 20 | 58982400 | 1 | 128 | 1 | 128 | 51200 |
| 36 | /model.10/m/m.0/attn/proj/conv/Conv_quant | 128 | 128 | 1 | 1 | 20 | 20 | 6553600 | 1 | 128 | 1 | 128 | 51200 |
| 37 | /model.10/m/m.0/ffn/ffn.0/conv/Conv_quant | 128 | 256 | 1 | 1 | 20 | 20 | 13107200 | 1 | 256 | 1 | 256 | 51200 |
| 38 | /model.10/m/m.0/ffn/ffn.1/conv/Conv_quant | 256 | 128 | 1 | 1 | 20 | 20 | 13107200 | 1 | 128 | 2 | 256 | 51200 |
| 39 | /model.10/cv2/conv/Conv_quant | 256 | 256 | 1 | 1 | 20 | 20 | 26214400 | 1 | 256 | 2 | 512 | 51200 |
| 40 | /model.13/cv1/conv/Conv_quant | 384 | 128 | 1 | 1 | 40 | 40 | 78643200 | 1 | 128 | 8 | 1024 | 76800 |
| 41 | /model.13/m.0/cv1/conv/Conv_quant | 64 | 32 | 1 | 1 | 40 | 40 | 3276800 | 1 | 32 | 2 | 64 | 51200 |
| 42 | /model.13/m.0/cv2/conv/Conv_quant | 64 | 32 | 1 | 1 | 40 | 40 | 3276800 | 1 | 32 | 2 | 64 | 51200 |
| 43 | /model.13/m.0/m/m.0/cv1/conv/Conv_quant | 32 | 32 | 3 | 1 | 40 | 40 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 44 | /model.13/m.0/m/m.0/cv2/conv/Conv_quant | 32 | 32 | 3 | 1 | 40 | 40 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 45 | /model.13/m.0/m/m.1/cv1/conv/Conv_quant | 32 | 32 | 3 | 1 | 40 | 40 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 46 | /model.13/m.0/m/m.1/cv2/conv/Conv_quant | 32 | 32 | 3 | 1 | 40 | 40 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 47 | /model.13/m.0/cv3/conv/Conv_quant | 64 | 64 | 1 | 1 | 40 | 40 | 6553600 | 1 | 64 | 2 | 128 | 51200 |
| 48 | /model.13/cv2/conv/Conv_quant | 192 | 128 | 1 | 1 | 40 | 40 | 39321600 | 1 | 128 | 4 | 512 | 76800 |
| 49 | /model.16/cv1/conv/Conv_quant | 256 | 64 | 1 | 1 | 80 | 80 | 104857600 | 1 | 64 | 32 | 2048 | 51200 |
| 50 | /model.16/m.0/cv1/conv/Conv_quant | 32 | 16 | 1 | 1 | 80 | 80 | 3276800 | 1 | 16 | 4 | 64 | 51200 |
| 51 | /model.16/m.0/cv2/conv/Conv_quant | 32 | 16 | 1 | 1 | 80 | 80 | 3276800 | 1 | 16 | 4 | 64 | 51200 |
| 52 | /model.16/m.0/m/m.0/cv1/conv/Conv_quant | 16 | 16 | 3 | 1 | 80 | 80 | 14745600 | 1 | 16 | 2 | 32 | 51200 |
| 53 | /model.16/m.0/m/m.0/cv2/conv/Conv_quant | 16 | 16 | 3 | 1 | 80 | 80 | 14745600 | 1 | 16 | 2 | 32 | 51200 |
| 54 | /model.16/m.0/m/m.1/cv1/conv/Conv_quant | 16 | 16 | 3 | 1 | 80 | 80 | 14745600 | 1 | 16 | 2 | 32 | 51200 |
| 55 | /model.16/m.0/m/m.1/cv2/conv/Conv_quant | 16 | 16 | 3 | 1 | 80 | 80 | 14745600 | 1 | 16 | 2 | 32 | 51200 |
| 56 | /model.16/m.0/cv3/conv/Conv_quant | 32 | 32 | 1 | 1 | 80 | 80 | 6553600 | 1 | 32 | 4 | 128 | 51200 |
| 57 | /model.16/cv2/conv/Conv_quant | 96 | 64 | 1 | 1 | 80 | 80 | 39321600 | 1 | 64 | 8 | 512 | 76800 |
| 58 | /model.17/conv/Conv_quant | 64 | 64 | 3 | 2 | 40 | 40 | 58982400 | 1 | 64 | 2 | 128 | 51200 |
| 59 | /model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv_quant | 64 | 16 | 3 | 1 | 80 | 80 | 58982400 | 1 | 16 | 8 | 128 | 51200 |
| 60 | /model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv_quant | 64 | 64 | 3 | 1 | 80 | 80 | 235929600 | 1 | 64 | 8 | 512 | 51200 |
| 61 | /model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv_quant | 16 | 16 | 3 | 1 | 80 | 80 | 14745600 | 1 | 16 | 2 | 32 | 51200 |
| 62 | /model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/conv/Conv_quant | 64 | 80 | 1 | 1 | 80 | 80 | 32768000 | 1 | 80 | 8 | 640 | 51200 |
| 63 | /model.19/cv1/conv/Conv_quant | 192 | 128 | 1 | 1 | 40 | 40 | 39321600 | 1 | 128 | 4 | 512 | 76800 |
| 64 | /model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_quant | 16 | 4 | 1 | 1 | 80 | 80 | 409600 | 1 | 4 | 2 | 8 | 51200 |
| 65 | /model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv_quant | 80 | 80 | 3 | 1 | 80 | 80 | 368640000 | 1 | 80 | 8 | 640 | 64000 |
| 66 | /model.19/m.0/cv1/conv/Conv_quant | 64 | 32 | 1 | 1 | 40 | 40 | 3276800 | 1 | 32 | 2 | 64 | 51200 |
| 67 | /model.19/m.0/cv2/conv/Conv_quant | 64 | 32 | 1 | 1 | 40 | 40 | 3276800 | 1 | 32 | 2 | 64 | 51200 |
| 68 | /model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv_quant | 80 | 80 | 1 | 1 | 80 | 80 | 40960000 | 1 | 80 | 8 | 640 | 64000 |
| 69 | /model.19/m.0/m/m.0/cv1/conv/Conv_quant | 32 | 32 | 3 | 1 | 40 | 40 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 70 | /model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_quant | 80 | 80 | 1 | 1 | 80 | 80 | 40960000 | 1 | 80 | 8 | 640 | 64000 |
| 71 | /model.19/m.0/m/m.0/cv2/conv/Conv_quant | 32 | 32 | 3 | 1 | 40 | 40 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 72 | /model.19/m.0/m/m.1/cv1/conv/Conv_quant | 32 | 32 | 3 | 1 | 40 | 40 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 73 | /model.19/m.0/m/m.1/cv2/conv/Conv_quant | 32 | 32 | 3 | 1 | 40 | 40 | 14745600 | 1 | 32 | 1 | 32 | 51200 |
| 74 | /model.19/m.0/cv3/conv/Conv_quant | 64 | 64 | 1 | 1 | 40 | 40 | 6553600 | 1 | 64 | 2 | 128 | 51200 |
| 75 | /model.19/cv2/conv/Conv_quant | 192 | 128 | 1 | 1 | 40 | 40 | 39321600 | 1 | 128 | 4 | 512 | 76800 |
| 76 | /model.20/conv/Conv_quant | 128 | 128 | 3 | 2 | 20 | 20 | 58982400 | 1 | 128 | 1 | 128 | 51200 |
| 77 | /model.23/one2one_cv2.1/one2one_cv2.1.0/conv/Conv_quant | 128 | 16 | 3 | 1 | 40 | 40 | 29491200 | 1 | 16 | 4 | 64 | 51200 |
| 78 | /model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv_quant | 128 | 128 | 3 | 1 | 40 | 40 | 235929600 | 1 | 128 | 4 | 512 | 51200 |
| 79 | /model.23/one2one_cv2.1/one2one_cv2.1.1/conv/Conv_quant | 16 | 16 | 3 | 1 | 40 | 40 | 3686400 | 1 | 8 | 1 | 8 | 51200 |
| 80 | /model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/conv/Conv_quant | 128 | 80 | 1 | 1 | 40 | 40 | 16384000 | 1 | 80 | 4 | 320 | 51200 |
| 81 | /model.22/cv1/conv/Conv_quant | 384 | 256 | 1 | 1 | 20 | 20 | 39321600 | 1 | 256 | 2 | 512 | 76800 |
| 82 | /model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_quant | 16 | 4 | 1 | 1 | 40 | 40 | 102400 | 1 | 2 | 1 | 2 | 51200 |
| 83 | /model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv_quant | 80 | 80 | 3 | 1 | 40 | 40 | 92160000 | 1 | 80 | 2 | 160 | 64000 |
| 84 | /model.22/m.0/m.0.0/cv1/conv/Conv_quant | 128 | 64 | 3 | 1 | 20 | 20 | 29491200 | 1 | 64 | 1 | 64 | 51200 |
| 85 | /model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv_quant | 80 | 80 | 1 | 1 | 40 | 40 | 10240000 | 1 | 80 | 2 | 160 | 64000 |
| 86 | /model.22/m.0/m.0.0/cv2/conv/Conv_quant | 64 | 128 | 3 | 1 | 20 | 20 | 29491200 | 1 | 64 | 1 | 64 | 51200 |
| 87 | /model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_quant | 80 | 80 | 1 | 1 | 40 | 40 | 10240000 | 1 | 80 | 2 | 160 | 64000 |
| 88 | /model.22/m.0/m.0.1/attn/qkv/conv/Conv_quant | 128 | 256 | 1 | 1 | 20 | 20 | 13107200 | 1 | 256 | 1 | 256 | 51200 |
| 89 | /model.22/m.0/m.0.1/attn/pe/conv/Conv_quant | 128 | 128 | 3 | 1 | 20 | 20 | 58982400 | 1 | 128 | 1 | 128 | 51200 |
| 90 | /model.22/m.0/m.0.1/attn/proj/conv/Conv_quant | 128 | 128 | 1 | 1 | 20 | 20 | 6553600 | 1 | 128 | 1 | 128 | 51200 |
| 91 | /model.22/m.0/m.0.1/ffn/ffn.0/conv/Conv_quant | 128 | 256 | 1 | 1 | 20 | 20 | 13107200 | 1 | 256 | 1 | 256 | 51200 |
| 92 | /model.22/m.0/m.0.1/ffn/ffn.1/conv/Conv_quant | 256 | 128 | 1 | 1 | 20 | 20 | 13107200 | 1 | 128 | 2 | 256 | 51200 |
| 93 | /model.22/cv2/conv/Conv_quant | 384 | 256 | 1 | 1 | 20 | 20 | 39321600 | 1 | 256 | 2 | 512 | 76800 |
| 94 | /model.23/one2one_cv2.2/one2one_cv2.2.0/conv/Conv_quant | 256 | 16 | 3 | 1 | 20 | 20 | 14745600 | 1 | 16 | 2 | 32 | 51200 |
| 95 | /model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/conv/Conv_quant | 256 | 256 | 3 | 1 | 20 | 20 | 235929600 | 1 | 256 | 2 | 512 | 51200 |
| 96 | /model.23/one2one_cv2.2/one2one_cv2.2.1/conv/Conv_quant | 16 | 16 | 3 | 1 | 20 | 20 | 921600 | 1 | 2 | 1 | 2 | 51200 |
| 97 | /model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/conv/Conv_quant | 256 | 80 | 1 | 1 | 20 | 20 | 8192000 | 1 | 80 | 2 | 160 | 51200 |
| 98 | /model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_quant | 16 | 4 | 1 | 1 | 20 | 20 | 25600 | 1 | 1 | 1 | 1 | 25600 |
| 99 | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv_quant | 80 | 80 | 3 | 1 | 20 | 20 | 23040000 | 1 | 40 | 1 | 40 | 64000 |
| 100 | /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv_quant | 80 | 80 | 1 | 1 | 20 | 20 | 2560000 | 1 | 40 | 1 | 40 | 64000 |
| 101 | /model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_quant | 80 | 80 | 1 | 1 | 20 | 20 | 2560000 | 1 | 40 | 1 | 40 | 64000 |

**Total MACs/frame:** 3,974,886,400  (= 3.9749 GMAC).
**Total MAC units instantiated:** 25,261.
**Worst-layer cycles_i:** 76800  (target T = 100000).
All layers meet T_FRAME.
