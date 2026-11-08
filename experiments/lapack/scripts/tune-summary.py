#!/usr/bin/env python3
"""Best configurations of a tuning run: tune-summary.py FILE [TOP]

Lines: tag routine lib n threads nb ib pt serial_n median min max samples
gflops check backward_error.  Per n: OpenBLAS (tag ref), the serial
algorithm (tag serial) and the TOP fastest configurations (GFLOP/s of the
median time, and the ratio to OpenBLAS)."""
import collections
import sys

top = int(sys.argv[2]) if len(sys.argv) > 2 else 5
d = collections.defaultdict(dict)
bad = 0
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) < 16 or line.startswith("#"):
        continue
    if f[14] != "ok":
        bad += 1
    d[(f[1], int(f[3]))][f[0]] = float(f[13])
for (r, n) in sorted(d):
    x = d[(r, n)]
    ref = x.get("ref", float("nan"))
    best = sorted(((g, t) for t, g in x.items() if t != "ref"), reverse=True)
    print("%s n=%-5d OpenBLAS %6.1f  serial %6.1f |" % (r, n, ref, x.get("serial", 0)),
          "  ".join("%s %.1f (%.2f)" % (t, g, g / ref) for g, t in best[:top]))
print("results not ok:", bad)
