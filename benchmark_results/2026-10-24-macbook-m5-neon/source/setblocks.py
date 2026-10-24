#!/usr/bin/env python3
#  setblocks.py BLOCKSIZE_H MC KC [NC]: set MC, KC (and NC) for double in the
#  USE_NEON branch of ulmblas/config/blocksize.h (exactly one line each).
import re, sys
path, mc, kc = sys.argv[1], sys.argv[2], sys.argv[3]
nc = sys.argv[4] if len(sys.argv) > 4 else None
s = open(path).read()
i = s.index("#elif defined(USE_NEON)")
j = s.index("#elif defined(USE_SSE)", i)
block = s[i:j]
pat = r"(std::is_same<double, T>::value\s+\? )(\d+)\n"
vals = [v for _, v in re.findall(pat, block)]
assert len(vals) == 5, vals          # MC, KC, NC, MR, NR
new = iter([mc, kc, nc or vals[2], vals[3], vals[4]])
block = re.sub(pat, lambda m: m.group(1) + next(new) + "\n", block)
open(path, "w").write(s[:i] + block + s[j:])
print("MC=%s KC=%s NC=%s MR=%s NR=%s"
      % tuple(v for _, v in re.findall(pat, block)))
