#!/usr/bin/env python3
"""Summarize a row-major campaign (cluster/rowmajor.sbatch).

    summarize-rowmajor.py RESULTDIR

RESULTDIR/raw/rROUND-ROUTINE-LIB-LAYOUT.txt: one line per n.  Per group
(library, layout) and n the median time of each round; reported is the
median over the rounds.  The speedup over OpenBLAS column major is computed
per round, its median over the rounds is reported.

Writes RESULTDIR/summary-<routine>-rowmajor.dat with the columns

    n  then for the groups openblas:colmajor, openblas:rowmajor,
       ulmblas:rowmajor, ulmblas:colmajor:  t gflops speedup

and prints a table.
"""
import collections
import glob
import os
import re
import statistics
import sys

FLOPS = {"getrf": lambda n: 2.0 / 3 * n**3, "potrf": lambda n: 1.0 / 3 * n**3}
GROUPS = ("openblas:colmajor", "openblas:rowmajor", "ulmblas:rowmajor", "ulmblas:colmajor")


def main():
    rdir = sys.argv[1]
    t = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(dict)))         # t[routine][group][n][round]
    bad = []
    pat = re.compile(r"(r\d+)-(getrf|potrf)-(ulmblas|openblas)-(rowmajor|colmajor)\.txt$")
    for path in sorted(glob.glob(os.path.join(rdir, "raw", "r*.txt"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        rnd, r, lib, lay = m.groups()
        want = "rm-inplace" if lay == "rowmajor" else "inplace"
        for line in open(path):
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            if f[13] != "ok" or f[16] != want or f[1] != lib:
                bad.append(os.path.basename(path) + ": " + line.strip())
            t[r][lib + ":" + lay][int(f[2])][rnd] = float(f[8])
    for r in sorted(t):
        ns = sorted(t[r]["openblas:colmajor"])
        out = os.path.join(rdir, "summary-%s-rowmajor.dat" % r)
        with open(out, "w") as o:
            o.write("# n" + "".join("  t_%s gflops_%s speedup_%s" % (g, g, g) for g in GROUPS) + "\n")
            print("%s: GFLOP/s (speedup over OpenBLAS column major)" % r)
            print("     n " + " ".join("%22s" % g for g in GROUPS))
            for n in ns:
                o.write("%5d" % n)
                row = []
                for g in GROUPS:
                    d = t[r][g][n]
                    if not d:
                        o.write("  nan nan nan")
                        row.append("%22s" % "-")
                        continue
                    tm = statistics.median(d.values())
                    base = t[r]["openblas:colmajor"][n]
                    sp = statistics.median(base[k] / d[k] for k in d if k in base)
                    gf = FLOPS[r](n) / tm * 1e-9
                    o.write("  %.6e %9.3f %7.4f" % (tm, gf, sp))
                    row.append("%14.1f (%5.2f)" % (gf, sp))
                o.write("\n")
                if n in (50, 100, 200, 500, 1000, 2000, 3000, 4000):
                    print("  %4d " % n + " ".join(row))
        print("  -> %s" % out)
    print("results not ok: %d" % len(bad))
    for line in bad[:20]:
        print("  " + line)


if __name__ == "__main__":
    main()
