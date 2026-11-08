#!/usr/bin/env python3
"""Summarize the benchmark campaign of ulmlapack vs. OpenBLAS.

    summarize.py RESULTDIR

RESULTDIR/raw/*.txt are outputs of the benchmark programs (one file per
round, routine and library).  For every routine, library and n the median
time of each round is taken (the median of the samples of that round); the
reported time is the median of the round medians.  The speedup is computed
per round (OpenBLAS time / ulmBLAS time of the same round) and its median
over the rounds is reported.

Writes RESULTDIR/summary-<routine>.dat (gnuplot) and prints a check report.
"""
import collections
import glob
import os
import statistics
import sys

FLOPS = {"getrf": lambda n: 2.0 / 3 * n**3, "potrf": lambda n: 1.0 / 3 * n**3}


def main():
    rdir = sys.argv[1]
    # data[routine][lib][n][round] = (median, min, max, samples, check, threads)
    data = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(dict)))
    params = {}
    bad = []
    for path in sorted(glob.glob(os.path.join(rdir, "raw", "*.txt"))):
        rnd = os.path.basename(path).split("-")[0]          # r1, r2, ...
        for line in open(path):
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            if len(f) < 15 or f[0] not in FLOPS:
                continue
            r, lib, n, threads = f[0], f[1], int(f[2]), int(f[3])
            med, mn, mx, k = float(f[8]), float(f[9]), float(f[10]), int(f[11])
            ok = f[13]
            if ok != "ok":
                bad.append(line.strip())
            data[r][lib][n][rnd] = (med, mn, mx, k, ok, threads)
            if lib == "ulmblas":
                params[(r, n)] = tuple(int(x) for x in f[3:8])
    for r in sorted(data):
        libs = data[r]
        ns = sorted(set(libs["ulmblas"]) & set(libs["openblas"]))
        out = os.path.join(rdir, "summary-%s.dat" % r)
        with open(out, "w") as o:
            o.write("# n  t_ulmblas t_openblas  gflops_ulmblas gflops_openblas"
                    "  speedup speedup_min speedup_max  rounds"
                    "  ulm_threads nb ib pt serial_n\n")
            for n in ns:
                u, b = libs["ulmblas"][n], libs["openblas"][n]
                rounds = sorted(set(u) & set(b))
                tu = statistics.median(u[x][0] for x in rounds)
                tb = statistics.median(b[x][0] for x in rounds)
                sp = [b[x][0] / u[x][0] for x in rounds]
                fl = FLOPS[r](n)
                p = params[(r, n)]
                o.write("%5d %.6e %.6e %9.3f %9.3f %7.4f %7.4f %7.4f %d"
                        "  %d %d %d %d %d\n"
                        % (n, tu, tb, fl / tu * 1e-9, fl / tb * 1e-9,
                           statistics.median(sp), min(sp), max(sp),
                           len(rounds), *p))
        print("%s: %d sizes, rounds per size: %s -> %s"
              % (r, len(ns), sorted(set(len(set(libs['ulmblas'][n]))
                                        for n in ns)), out))
    print("results not ok: %d" % len(bad))
    for line in bad[:20]:
        print("  " + line)


if __name__ == "__main__":
    main()
