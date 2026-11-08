#!/usr/bin/env python3
"""Summarize a row stride sweep (cluster/sweep.sbatch).

    summarize-sweep.py RESULTDIR

RESULTDIR/raw/sweep-rROUND-ROUTINE-incR-VARIANT.txt: one line per n.  Per
(routine, variant, R, n) the median over the rounds of the round medians.

Writes RESULTDIR/summary-sweep-<routine>.dat, one line per row stride R:

    R  then for the variants inplace, colmajor, tiled and every n:
       gflops  relative (time of in place with R = 1 / time)

(nan where not measured; the header names the columns) and prints a table.
"""
import collections
import glob
import os
import re
import statistics
import sys

FLOPS = {"getrf": lambda n: 2.0 / 3 * n**3, "potrf": lambda n: 1.0 / 3 * n**3}
VARIANTS = ("inplace", "colmajor", "tiled")


def main():
    rdir = sys.argv[1]
    # t[routine][variant][R][n][round] = median time
    t = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(dict))))
    bad = []
    pat = re.compile(r"sweep-(r\d+)-(getrf|potrf)-inc(\d+)-([\w+]+)\.txt$")
    for path in sorted(glob.glob(os.path.join(rdir, "raw", "sweep-*.txt"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        rnd, routine, inc, var = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        for line in open(path):
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            if len(f) < 17 or f[0] != routine:
                continue
            n, med, ok = int(f[2]), float(f[8]), f[13]
            # the benchmark labels prepack +pp and outer blocking +outerS
            label = f[16].replace("+outer", "+o")
            if ok != "ok" or int(f[15]) != inc or label != var:
                bad.append(os.path.basename(path) + ": " + line.strip())
            t[routine][var][inc][n][rnd] = med
    for r in sorted(t):
        variants = list(VARIANTS) + sorted(v for v in t[r] if v not in VARIANTS)
        incs = sorted(set(i for v in t[r] for i in t[r][v]))
        ns = sorted(set(n for v in t[r] for i in t[r][v] for n in t[r][v][i]))
        med = lambda v, i, n: (statistics.median(t[r][v][i][n].values())
                               if t[r][v][i][n] else float("nan"))
        out = os.path.join(rdir, "summary-sweep-%s.dat" % r)
        with open(out, "w") as o:
            o.write("# R")
            for v in variants:
                for n in ns:
                    o.write("  gf_%s_%d rel_%s_%d" % (v, n, v, n))
            o.write("\n")
            for i in incs:
                o.write("%4d" % i)
                for v in variants:
                    for n in ns:
                        tt = med(v, i, n)
                        t1 = med("inplace", 1, n)
                        o.write("  %9.3f %7.4f" % (FLOPS[r](n) / tt * 1e-9, t1 / tt))
                o.write("\n")
        print("%s: R %s, n %s -> %s" % (r, incs, ns, out))
        for v in variants:
            if not any(t[r][v][i][n] for i in incs for n in ns):
                continue
            print("  %s, GFLOP/s (rows R, columns n %s)" % (v, ns))
            for i in incs:
                print("   %4d " % i + " ".join("%7.1f" % (FLOPS[r](n) / med(v, i, n) * 1e-9)
                                              for n in ns))
    print("results not ok: %d" % len(bad))
    for line in bad[:20]:
        print("  " + line)


if __name__ == "__main__":
    main()
