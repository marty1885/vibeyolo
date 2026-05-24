// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// fp16_lat_pkg — single source of truth for the pipeline latency of the
// two shared fp16 leaf IPs. Consumers that bypass these blocks with
// matched delay-lines, valid shift-registers, or drain counters MUST size
// them from these localparams so a future depth change stays consistent.
//
//   FP16_FMA_LAT    : cycles from a_i/b_i/c_i to y_o for fp16_fma.
//   I32_TO_FP16_LAT : cycles from x_i to y_o/shift_o for i32_to_fp16.
//
// These values are validated against the real RTL by every consumer's
// lockstep / golden DV: if a leaf's actual register depth and the value
// here ever disagree, the integration tests that use the package will
// fail. There is therefore no separate compile-time assertion in the leaf.

package fp16_lat_pkg;

  // fp16_fma: stage1 unpack+multiply+align, stage2 add+normalize,
  //           stage3 round+pack.
  localparam int unsigned FP16_FMA_LAT = 3;

  // i32_to_fp16: stage1 abs+leading-one+prescale+shift,
  //              stage2 align+round+pack.
  localparam int unsigned I32_TO_FP16_LAT = 2;

endpackage
