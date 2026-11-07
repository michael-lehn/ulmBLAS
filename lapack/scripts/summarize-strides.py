#!/usr/bin/env python3
"""Summarize a row stride campaign (cluster/bench.sbatch with INCS).

    summarize-strides.py RESULTDIR

RESULTDIR/raw/rROUND-ROUTINE-LIB-incR[-VARIANT].txt are outputs of the
benchmark programs (VARIANT: colmajor or tiled; none: inplace).  As in
summarize.py: per variant (library, row stride) and n the median time of
each round, reported is the median over the rounds.  The
speedup over OpenBLAS (column major) is computed per round and its median,
min and max over the rounds are reported.

Writes RESULTDIR/summary-<routine>-strides.dat with the columns

    n  t_openblas gflops_openblas
       then per row stride R of ulmBLAS, in place (in increasing order):
       t_R gflops_R speedup_R speedup_min_R speedup_max_R

and, if there are copying variants, RESULTDIR/summary-<routine>-variants.dat
with the same columns for the groups (variant, R) in the order inplace,
colmajor, tiled and R = 1, 2, 42 (nan where not measured), followed by the
ratio t_inplace(R) / t_variant(R) per group (median over the rounds).
Prints a check report.
"""
import collections
import glob
import os
import re
import statistics
import sys

FLOPS = {"getrf": lambda n: 2.0 / 3 * n**3, "potrf": lambda n: 1.0 / 3 * n**3}


def main():
    rdir = sys.argv[1]
    # data[routine][(lib, inc)][n][round] = median time
    data = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(dict)))
    bad = []
    pat = re.compile(r"(r\d+)-(getrf|potrf)-(ulmblas|openblas)-inc(\d+)"
                     r"(?:-(colmajor|tiled))?\.txt$")
    for path in sorted(glob.glob(os.path.join(rdir, "raw", "*.txt"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        rnd, routine, lib, inc = m.group(1), m.group(2), m.group(3), int(m.group(4))
        var = m.group(5) or "inplace"
        for line in open(path):
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            if len(f) < 16 or f[0] != routine:
                continue
            n, med, ok = int(f[2]), float(f[8]), f[13]
            fvar = f[16] if len(f) > 16 else "inplace"
            if ok != "ok" or int(f[15]) != inc or fvar != var:
                bad.append(os.path.basename(path) + ": " + line.strip())
            key = (lib, inc) if var == "inplace" else (lib, inc, var)
            data[routine][key][n][rnd] = med
    for r in sorted(data):
        v = data[r]
        ob = v[("openblas", 1)]
        incs = sorted(k[1] for k in v if k[0] == "ulmblas" and len(k) == 2)
        ns = sorted(set(ob).intersection(*(set(v[("ulmblas", i)]) for i in incs)))
        out = os.path.join(rdir, "summary-%s-strides.dat" % r)
        with open(out, "w") as o:
            o.write("# n  t_openblas gflops_openblas")
            for i in incs:
                o.write("  t_%d gflops_%d speedup_%d smin_%d smax_%d" % ((i,) * 5))
            o.write("\n")
            for n in ns:
                fl = FLOPS[r](n)
                tb = statistics.median(ob[n].values())
                o.write("%5d %.6e %9.3f" % (n, tb, fl / tb * 1e-9))
                for i in incs:
                    u = v[("ulmblas", i)][n]
                    rounds = sorted(set(u) & set(ob[n]))
                    tu = statistics.median(u[x] for x in rounds)
                    sp = [ob[n][x] / u[x] for x in rounds]
                    o.write("  %.6e %9.3f %7.4f %7.4f %7.4f"
                            % (tu, fl / tu * 1e-9, statistics.median(sp),
                               min(sp), max(sp)))
                o.write("\n")
        groups = [(var, i) for var in ("inplace", "colmajor", "tiled")
                  for i in (1, 2, 42)]
        if any(len(k) == 3 for k in v):
            outv = os.path.join(rdir, "summary-%s-variants.dat" % r)

            def key(var, i):
                return ("ulmblas", i) if var == "inplace" else ("ulmblas", i, var)
            with open(outv, "w") as o:
                o.write("# n  t_openblas gflops_openblas  then per group "
                        "(t gflops speedup smin smax): %s  then t_inplace/t "
                        "per group\n" % " ".join("%s:%d" % g for g in groups))
                nsv = sorted(n for n in ob if all(
                    n in v[key(*g)] for g in groups if key(*g) in v))
                for n in nsv:
                    fl = FLOPS[r](n)
                    tb = statistics.median(ob[n].values())
                    o.write("%5d %.6e %9.3f" % (n, tb, fl / tb * 1e-9))
                    ratios = []
                    for var, i in groups:
                        k = key(var, i)
                        if k not in v:
                            o.write("  nan nan nan nan nan")
                            ratios.append("nan")
                            continue
                        u = v[k][n]
                        rounds = sorted(set(u) & set(ob[n]))
                        tu = statistics.median(u[x] for x in rounds)
                        sp = [ob[n][x] / u[x] for x in rounds]
                        o.write("  %.6e %9.3f %7.4f %7.4f %7.4f"
                                % (tu, fl / tu * 1e-9, statistics.median(sp),
                                   min(sp), max(sp)))
                        ip = v[("ulmblas", i)][n]
                        rr = sorted(set(u) & set(ip))
                        ratios.append("%.4f" % statistics.median(
                            ip[x] / u[x] for x in rr))
                    o.write("  " + " ".join(ratios) + "\n")
            print("%s: variants -> %s (%d sizes)" % (r, outv, len(nsv)))
        rounds = sorted(set(len(v[k][n]) for k in v for n in ns))
        print("%s: %d sizes, row strides %s, rounds per size %s -> %s"
              % (r, len(ns), incs, rounds, out))
    print("results not ok: %d" % len(bad))
    for line in bad[:20]:
        print("  " + line)


if __name__ == "__main__":
    main()
