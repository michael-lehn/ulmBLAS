#!/usr/bin/env python3
"""Summarize a campaign of the ulmBLAS templates (cluster/core.sbatch).

    summarize-core.py RESULTDIR

RESULTDIR/raw/rROUND-PREC-ROUTINE-LIB.txt (bench/core.cc), LIB = ulmblas,
openblas, mkl.  Per (prec, routine, lib, n) the median time over the
rounds; speedups of ulmBLAS per round, median over the rounds.

Writes RESULTDIR/summary-ROUTINE.dat, one line per n:

    n  then for PREC in s d c z:
       gflops_ulmblas gflops_openblas gflops_mkl
       speedup_over_openblas speedup_over_mkl

(nan where not measured) and prints a table and a check report.
"""
import collections
import glob
import os
import re
import statistics
import sys

PRECS = ("s", "d", "c", "z")
LIBS = ("ulmblas", "openblas", "mkl")


def main():
    rdir = sys.argv[1]
    # t[routine][prec][lib][n][round] = median time; g = gflops
    t = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(dict))))
    gf = collections.defaultdict(dict)
    bad = []
    pat = re.compile(r"(r\d+)-([sdcz])-(getrf|potrf)-(ulmblas|openblas|mkl)\.txt$")
    for path in sorted(glob.glob(os.path.join(rdir, "raw", "r*.txt"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        rnd, p, r, lib = m.groups()
        for line in open(path):
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            if len(f) < 12 or f[0] != p or f[1] != r or f[2] != lib:
                bad.append(os.path.basename(path) + ": " + line.strip())
                continue
            if f[10] != "ok":
                bad.append(os.path.basename(path) + ": " + line.strip())
            n, med, g = int(f[3]), float(f[5]), float(f[9])
            t[r][p][lib][n][rnd] = med
            gf[(r, p, lib, n, rnd)] = g
    nan = float("nan")
    for r in sorted(t):
        ns = sorted(set(n for p in t[r] for lib in t[r][p] for n in t[r][p][lib]))
        out = os.path.join(rdir, "summary-%s.dat" % r)
        rows = {}
        with open(out, "w") as o:
            o.write("# n")
            for p in PRECS:
                o.write("  gf_ulmblas_%s gf_openblas_%s gf_mkl_%s sp_openblas_%s sp_mkl_%s"
                        % (p, p, p, p, p))
            o.write("\n")
            for n in ns:
                o.write("%5d" % n)
                for p in PRECS:
                    vals = []
                    for lib in LIBS:
                        d = t[r][p][lib][n]
                        if d:
                            tm = statistics.median(d.values())
                            # gflops of the median time (flops = gflops * time)
                            k = next(iter(d))
                            vals.append(gf[(r, p, lib, n, k)] * d[k] / tm)
                        else:
                            vals.append(nan)
                    for lib in ("openblas", "mkl"):
                        u, b = t[r][p]["ulmblas"][n], t[r][p][lib][n]
                        common = [k for k in u if k in b]
                        vals.append(statistics.median(b[k] / u[k] for k in common)
                                    if common else nan)
                    rows[(p, n)] = vals
                    o.write("  " + " ".join("%9.3f" % v for v in vals))
                o.write("\n")
        print("%s -> %s" % (r, out))
        print("   n   prec  ulmBLAS  OpenBLAS   oneMKL   /OpenBLAS  /oneMKL")
        for n in (100, 200, 500, 1000, 2000, 3000, 4000):
            for p in PRECS:
                if (p, n) in rows:
                    v = rows[(p, n)]
                    print("%5d   %s   %8.1f %8.1f %8.1f   %6.2f   %6.2f"
                          % (n, p, v[0], v[1], v[2], v[3], v[4]))
    print("results not ok: %d" % len(bad))
    for line in bad[:20]:
        print("  " + line)


if __name__ == "__main__":
    main()
