# YOLO26n batch generation + DV report (L27-L101)

Total DV wall time: 861s (14m 21s)

## Pass/fail table

| Layer | ONNX node | Pattern | Generation | Build | Test cos | Wall | Failure reason |
|------:|-----------|---------|:----------:|:-----:|---------:|-----:|----------------|
| L27 | `/model.8/m.0/m/m.1/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.999239 | 11s |  |
| L28 | `/model.8/m.0/m/m.1/cv2/conv/Conv_quant` | residual | OK | OK | 0.999905 | 25s |  |
| L29 | `/model.8/m.0/cv3/conv/Conv_quant` | ordinary | OK | OK | 0.999379 | 10s |  |
| L30 | `/model.8/cv2/conv/Conv_quant` | ordinary | OK | OK | 0.999352 | 25s |  |
| L31 | `/model.9/cv1/conv/Conv_quant` | no-silu | OK | OK | 0.999897 | 17s |  |
| L32 | `/model.9/cv2/conv/Conv_quant` | residual | OK | OK | 0.999350 | 78s |  |
| L33 | `/model.10/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.999690 | 21s |  |
| L34 | `/model.10/m/m.0/attn/qkv/conv/Conv_quant` | no-silu | OK | OK | 0.999874 | 17s |  |
| L35 | `/model.10/m/m.0/attn/pe/conv/Conv_quant` | grouped/depthwise+no-silu | OK | FAIL |  |  | GEN-partial: grouped/depthwise (no extract/test emitted) |
| L36 | `/model.10/m/m.0/attn/proj/conv/Conv_quant` | residual+no-silu | FAIL | - |  |  | AssertionError: residual+no-silu not supported by generator |
| L37 | `/model.10/m/m.0/ffn/ffn.0/conv/Conv_quant` | ordinary | OK | OK | 0.996903 |  | TEST FAILED: 4 error(s) |
| L38 | `/model.10/m/m.0/ffn/ffn.1/conv/Conv_quant` | residual+no-silu | FAIL | - |  |  | AssertionError: residual+no-silu not supported by generator |
| L39 | `/model.10/cv2/conv/Conv_quant` | ordinary | OK | OK | 0.998448 | 21s |  |
| L40 | `/model.13/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.999429 | 19s |  |
| L41 | `/model.13/m.0/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.997615 | 8s |  |
| L42 | `/model.13/m.0/cv2/conv/Conv_quant` | ordinary | OK | OK | 0.998541 | 8s |  |
| L43 | `/model.13/m.0/m/m.0/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.996998 |  | TEST FAILED: 2 error(s) |
| L44 | `/model.13/m.0/m/m.0/cv2/conv/Conv_quant` | residual | OK | OK | 0.999406 | 19s |  |
| L45 | `/model.13/m.0/m/m.1/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.997451 |  | TEST FAILED: 1 error(s) |
| L46 | `/model.13/m.0/m/m.1/cv2/conv/Conv_quant` | residual | OK | OK | 0.999923 | 19s |  |
| L47 | `/model.13/m.0/cv3/conv/Conv_quant` | ordinary | OK | OK | 0.998450 | 8s |  |
| L48 | `/model.13/cv2/conv/Conv_quant` | ordinary | OK | OK | 0.998503 | 16s |  |
| L49 | `/model.16/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.999660 | 13s |  |
| L50 | `/model.16/m.0/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.996221 |  | TEST FAILED: 6 error(s) |
| L51 | `/model.16/m.0/cv2/conv/Conv_quant` | ordinary | OK | OK | 0.999765 | 7s |  |
| L52 | `/model.16/m.0/m/m.0/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.998938 | 9s |  |
| L53 | `/model.16/m.0/m/m.0/cv2/conv/Conv_quant` | residual | OK | OK | 0.998505 | 17s |  |
| L54 | `/model.16/m.0/m/m.1/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.999532 | 8s |  |
| L55 | `/model.16/m.0/m/m.1/cv2/conv/Conv_quant` | residual | OK | OK | 0.998792 | 17s |  |
| L56 | `/model.16/m.0/cv3/conv/Conv_quant` | ordinary | OK | OK | 0.999808 | 8s |  |
| L57 | `/model.16/cv2/conv/Conv_quant` | ordinary | OK | OK | 0.999610 | 13s |  |
| L58 | `/model.17/conv/Conv_quant` | no-silu | OK | OK | 0.999956 | 13s |  |
| L59 | `/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv_quant` | no-silu | OK | OK | 0.999980 | 13s |  |
| L60 | `/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv_quant` | grouped/depthwise | OK | FAIL |  |  | GEN-partial: grouped/depthwise (no extract/test emitted) |
| L61 | `/model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv_quant` | no-silu | OK | OK | 0.999981 | 9s |  |
| L62 | `/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/conv/Conv_quant` | no-silu | OK | OK | 0.999895 | 12s |  |
| L63 | `/model.19/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.999141 | 15s |  |
| L64 | `/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_quant` | no-silu | OK | OK | 0.999988 | 5s |  |
| L65 | `/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv_quant` | grouped/depthwise | OK | FAIL |  |  | GEN-partial: grouped/depthwise (no extract/test emitted) |
| L66 | `/model.19/m.0/cv1/conv/Conv_quant` | no-silu | OK | OK | 0.999956 | 7s |  |
| L67 | `/model.19/m.0/cv2/conv/Conv_quant` | no-silu | OK | OK | 0.999934 | 8s |  |
| L68 | `/model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv_quant` | ordinary | OK | OK | 0.998915 |  | TEST FAILED: 1 error(s) |
| L69 | `/model.19/m.0/m/m.0/cv1/conv/Conv_quant` | no-silu | OK | OK | 0.999918 | 10s |  |
| L70 | `/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_quant` | no-silu | OK | OK | 0.999993 | 12s |  |
| L71 | `/model.19/m.0/m/m.0/cv2/conv/Conv_quant` | residual | OK | OK | 0.999070 | 18s |  |
| L72 | `/model.19/m.0/m/m.1/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.998694 | 10s |  |
| L73 | `/model.19/m.0/m/m.1/cv2/conv/Conv_quant` | residual | OK | OK | 0.999844 | 19s |  |
| L74 | `/model.19/m.0/cv3/conv/Conv_quant` | ordinary | OK | OK | 0.999798 | 8s |  |
| L75 | `/model.19/cv2/conv/Conv_quant` | ordinary | OK | OK | 0.999456 | 15s |  |
| L76 | `/model.20/conv/Conv_quant` | no-silu | OK | OK | 0.999904 | 19s |  |
| L77 | `/model.23/one2one_cv2.1/one2one_cv2.1.0/conv/Conv_quant` | no-silu | OK | OK | 0.999966 | 10s |  |
| L78 | `/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv_quant` | grouped/depthwise | OK | FAIL |  |  | GEN-partial: grouped/depthwise (no extract/test emitted) |
| L79 | `/model.23/one2one_cv2.1/one2one_cv2.1.1/conv/Conv_quant` | no-silu | OK | OK | 0.999970 | 6s |  |
| L80 | `/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/conv/Conv_quant` | no-silu | OK | OK | 0.999847 | 13s |  |
| L81 | `/model.22/cv1/conv/Conv_quant` | ordinary | OK | OK | 0.999111 | 25s |  |
| L82 | `/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_quant` | no-silu | OK | FAIL |  |  | %Error: Exiting due to 5 warning(s) |
| L83 | `/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv_quant` | grouped/depthwise | OK | FAIL |  |  | GEN-partial: grouped/depthwise (no extract/test emitted) |
| L84 | `/model.22/m.0/m.0.0/cv1/conv/Conv_quant` | no-silu | OK | OK | 0.999955 | 13s |  |
| L85 | `/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv_quant` | ordinary | OK | OK | 0.995437 |  | TEST FAILED: 5 error(s) |
| L86 | `/model.22/m.0/m.0.0/cv2/conv/Conv_quant` | no-silu | OK | OK | 0.999939 | 13s |  |
| L87 | `/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_quant` | no-silu | OK | OK | 0.999990 | 9s |  |
| L88 | `/model.22/m.0/m.0.1/attn/qkv/conv/Conv_quant` | no-silu | OK | OK | 0.999881 | 16s |  |
| L89 | `/model.22/m.0/m.0.1/attn/pe/conv/Conv_quant` | grouped/depthwise+no-silu | OK | FAIL |  |  | GEN-partial: grouped/depthwise (no extract/test emitted) |
| L90 | `/model.22/m.0/m.0.1/attn/proj/conv/Conv_quant` | residual+no-silu | FAIL | - |  |  | AssertionError: residual+no-silu not supported by generator |
| L91 | `/model.22/m.0/m.0.1/ffn/ffn.0/conv/Conv_quant` | ordinary | OK | OK | 0.999117 | 17s |  |
| L92 | `/model.22/m.0/m.0.1/ffn/ffn.1/conv/Conv_quant` | residual+no-silu | FAIL | - |  |  | AssertionError: residual+no-silu not supported by generator |
| L93 | `/model.22/cv2/conv/Conv_quant` | ordinary | OK | OK | 0.999732 | 25s |  |
| L94 | `/model.23/one2one_cv2.2/one2one_cv2.2.0/conv/Conv_quant` | ordinary | OK | OK | 0.999954 | 10s |  |
| L95 | `/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/conv/Conv_quant` | grouped/depthwise | OK | FAIL |  |  | GEN-partial: grouped/depthwise (no extract/test emitted) |
| L96 | `/model.23/one2one_cv2.2/one2one_cv2.2.1/conv/Conv_quant` | no-silu | OK | OK | 0.999943 | 5s |  |
| L97 | `/model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/conv/Conv_quant` | ordinary | OK | OK | 0.999457 | 11s |  |
| L98 | `/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_quant` | no-silu | OK | FAIL |  |  | %Error: Exiting due to 13 warning(s) |
| L99 | `/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv_quant` | grouped/depthwise | OK | FAIL |  |  | GEN-partial: grouped/depthwise (no extract/test emitted) |
| L100 | `/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv_quant` | ordinary | OK | OK | 0.985891 |  | TEST FAILED: 6 error(s) |
| L101 | `/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_quant` | no-silu | OK | OK | 0.999983 | 10s |  |

## Failure-pattern histogram

| Cause | Count | Layers |
|-------|------:|--------|
| PASS | 54 | (passing) |
| GEN-partial: grouped/depthwise | 8 | L35, L60, L65, L78, L83, L89, L95, L99 |
| DV-test-cos | 7 | L37, L43, L45, L50, L68, L85, L100 |
| GEN: residual+no-silu | 4 | L36, L38, L90, L92 |
| DV-build-or-skel | 2 | L82, L98 |
