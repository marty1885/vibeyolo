#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# Generate weight/scale/bias hex files for weight_rom DV.
#
# Pattern (deterministic, easy to predict in C++):
#   row N, byte b: (N * RowLen + b) & 0xFF, interpreted as int8 via two's
#                  complement (we just write the unsigned byte; the DUT
#                  reinterprets via signed slice).
#   scale[N]:      fp16 encode of 1.0 / (N + 1)
#   bias[N]:       fp16 encode of N * 0.5
#
# Output files (under <out_prefix>):
#   <out_prefix>.w.hex   one line per output channel, RowBits-wide hex word
#   <out_prefix>.s.hex   one line per output channel, 4 hex chars (fp16)
#   <out_prefix>.b.hex   one line per output channel, 4 hex chars (fp16)
#
# Usage:
#   gen_hex.py --kh 3 --kw 3 --ic 4 --oc 8 --out build/wrom_small

import argparse
import os
import struct


def encode_fp16(x: float) -> int:
    # IEEE-754 binary16 via struct's 'e' format (Python 3.6+).
    return struct.unpack("<H", struct.pack("<e", x))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kh", type=int, required=True)
    ap.add_argument("--kw", type=int, required=True)
    ap.add_argument("--ic", type=int, required=True)
    ap.add_argument("--oc", type=int, required=True)
    ap.add_argument("--out", type=str, required=True,
                    help="output prefix; writes <out>.w.hex/.s.hex/.b.hex")
    args = ap.parse_args()

    row_len = args.kh * args.kw * args.ic
    row_nibbles = row_len * 2  # 2 hex chars per byte

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Weights: pack RowBits-wide values, MSB-first in hex (so byte 0 is at
    # the LSB end of the word, matching SV [b*8 +: 8] indexing).
    with open(args.out + ".w.hex", "w") as f:
        for n in range(args.oc):
            # Build little-endian byte list (byte 0 = LSB).
            row_bytes = [(n * row_len + b) & 0xFF for b in range(row_len)]
            # Hex string is MSB-first → reverse for printing.
            hex_str = "".join(f"{b:02x}" for b in reversed(row_bytes))
            assert len(hex_str) == row_nibbles
            f.write(hex_str + "\n")

    with open(args.out + ".s.hex", "w") as f:
        for n in range(args.oc):
            f.write(f"{encode_fp16(1.0 / (n + 1)):04x}\n")

    with open(args.out + ".b.hex", "w") as f:
        for n in range(args.oc):
            f.write(f"{encode_fp16(n * 0.5):04x}\n")


if __name__ == "__main__":
    main()
