#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# merge_liberty.py — merge several single-group liberty files into one.
#
# ASAP7 ships its standard cells split across functional groups (AO, INVBUF,
# OA, SEQ, SIMPLE), one .lib each. yosys `dfflibmap`/`abc`/`stat -liberty` each
# take a SINGLE liberty, so we fold the groups into one: keep the first file's
# header (units + lu_table_templates), then append every top-level `cell (...)`
# block from all files. Templates are identical across a corner's group files,
# so we take them once from the base and drop them from the rest.
#
#   python3 tools/merge_liberty.py out.lib in1.lib in2.lib ...

import sys, re

def top_cell_blocks(text):
    """Yield each top-level `cell (...) { ... }` block by brace matching."""
    for m in re.finditer(r'\bcell\s*\(', text):
        i = text.index('{', m.end() - 1)
        depth, j = 0, i
        while j < len(text):
            c = text[j]
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    yield text[m.start():j + 1]
                    break
            j += 1

def main():
    out, base, *rest = sys.argv[1], sys.argv[2], *sys.argv[3:]
    btext = open(base).read()
    # split the base library at its final closing brace (the `library(){...}` end)
    end = btext.rstrip().rfind('}')
    head, tail = btext[:end], btext[end:]
    n = btext.count('\n  cell (')      # cells already in base
    extra = []
    for f in rest:
        blocks = list(top_cell_blocks(open(f).read()))
        n += len(blocks)
        extra.append(f"\n  /* ---- merged from {f.split('/')[-1]} ---- */\n")
        extra.extend(b + "\n" for b in blocks)
    open(out, "w").write(head + "".join(extra) + tail)
    print(f"wrote {out}: {n} cells from {1 + len(rest)} libs")

if __name__ == "__main__":
    main()
