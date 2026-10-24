#!/usr/bin/env python3
#
#  summarize.py: converts the raw outputs of the benchmark programs
#  (data/raw/run-*/*.txt, written by bench2 and bench3, see README.md) into
#
#      data/dgemm.csv     all dgemm measurements (one line per size, variant
#                         and round)
#      data/level3.csv    all dsyrk, dsymm, dtrmm, dtrsm measurements
#      data/summary/*.csv medians over the rounds, the input of the plots
#
#  Usage (from the campaign directory):  python3 source/summarize.py
#
#  The raw files are named KIND-VARIANT-ROUND.txt.  The block sizes of each
#  variant are listed in VARIANTS (from ulmblas/config/blocksize.h of the
#  measured source, from the BLIS and OpenBLAS sources).  Nothing is
#  interpolated: a summary entry is the median of the rounds of exactly that
#  measurement, missing measurements are written as NaN.
#
import csv
import glob
import os
import statistics
import sys

#  run, variant -> library, kernel, MR, NR, MC, KC, NC
VARIANTS = {
    ("A", "ref"):         ("ulmBLAS", "reference (C++)", 4, 4, 384, 384, 4096),
    ("A", "sse"):         ("ulmBLAS", "SSE 4x4", 4, 4, 384, 384, 4096),
    ("A", "avx"):         ("ulmBLAS", "AVX/FMA 6x8", 6, 8, 72, 256, 4080),
    ("A", "blis"):        ("BLIS 2.0", "haswell 6x8", 6, 8, 72, 256, 4080),
    ("A", "openblas"):    ("OpenBLAS 0.3.34", "HASWELL 4x8", 4, 8, 512, 256, 13824),
    ("B", "sse"):         ("ulmBLAS", "SSE 4x4", 4, 4, 384, 384, 4096),
    ("B", "avx"):         ("ulmBLAS", "AVX/FMA 6x8", 6, 8, 288, 256, 4080),
    ("B", "blis"):        ("BLIS 2.0", "haswell 6x8", 6, 8, 72, 256, 4080),
    ("B", "openblas"):    ("OpenBLAS 0.3.34", "HASWELL 4x8", 4, 8, 512, 256, 13824),
}
for mc in (72, 144, 288):
    for kc in (256, 384, 512):
        VARIANTS[("A", "avx-%d-%d" % (mc, kc))] = \
            ("ulmBLAS", "AVX/FMA 6x8", 6, 8, mc, kc, 4080)

HEADER = ["run", "library", "variant", "kernel", "MR", "NR", "MC", "KC", "NC"]


def read_raw(top):
    dgemm, level3 = [], []
    for path in sorted(glob.glob(os.path.join(top, "data/raw/run-*/*.txt"))):
        run = os.path.basename(os.path.dirname(path))[len("run-"):]
        kind, rest = os.path.basename(path)[:-4].split("-", 1)
        variant, rnd = rest.rsplit("-", 1)
        meta = [run] + [VARIANTS[(run, variant)][0], variant] \
            + list(VARIANTS[(run, variant)][1:])
        samples = 5 if kind in ("sweep", "l3") else 7
        for line in open(path):
            x = line.split()
            if kind == "l3":
                level3.append(meta + [x[0], int(x[1]), int(rnd), samples,
                                      float(x[2]), float(x[3]), float(x[4]),
                                      int(x[5])])
            else:
                dgemm.append(meta + [kind, int(x[0]), int(x[1]), int(x[2]),
                                     1.0, float(x[3]), int(rnd), samples,
                                     float(x[4]), float(x[5]), float(x[6]),
                                     int(x[7]), x[8]])
    return dgemm, level3


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def median(rows, **key):
    values = [r["gflops_median"] for r in rows
              if all(r[k] == v for k, v in key.items())]
    return statistics.median(values) if values else float("nan")


def fmt(x):
    return "NaN" if x != x else "%.3f" % x


def main():
    top = sys.argv[1] if len(sys.argv) > 1 else "."
    dgemm, level3 = read_raw(top)
    dh = HEADER + ["set", "m", "n", "k", "alpha", "beta", "round",
                   "samples", "gflops_median", "gflops_min", "gflops_max",
                   "calls_per_sample", "check"]
    lh = HEADER + ["routine", "n", "round", "samples", "gflops_median",
                   "gflops_min", "gflops_max", "calls_per_sample"]
    write(os.path.join(top, "data/dgemm.csv"), dh, dgemm)
    write(os.path.join(top, "data/level3.csv"), lh, level3)
    D = [dict(zip(dh, r)) for r in dgemm]
    L = [dict(zip(lh, r)) for r in level3]
    S = os.path.join(top, "data/summary")
    os.makedirs(S, exist_ok=True)

    # square dgemm, beta = 1
    cols = [("ref_mc384", "A", "ref"), ("sse_mc384", "B", "sse"),
            ("avx_mc72", "A", "avx"), ("avx_mc288", "B", "avx"),
            ("blis", "B", "blis"), ("openblas", "B", "openblas")]
    sizes = sorted({r["n"] for r in D if r["set"] == "main"
                    and r["m"] == r["n"] == r["k"]})
    rows = []
    for n in sizes:
        v = {c: median(D, run=run, variant=var, set="main", m=n, n=n, k=n)
             for c, run, var in cols}
        rows.append([n] + [fmt(v[c]) for c, _, _ in cols]
                    + [fmt(v["avx_mc288"] / v["sse_mc384"]),
                       fmt(v["avx_mc72"] / v["sse_mc384"]),
                       fmt(v["avx_mc288"] / v["openblas"]),
                       fmt(v["avx_mc288"] / v["blis"]),
                       fmt(v["blis"] / v["openblas"]),
                       fmt(v["sse_mc384"] / v["openblas"])])
    write(os.path.join(S, "dgemm_square.csv"),
          ["n"] + [c for c, _, _ in cols]
          + ["avx_mc288_over_sse", "avx_mc72_over_sse",
             "avx_mc288_over_openblas", "avx_mc288_over_blis",
             "blis_over_openblas", "sse_over_openblas"], rows)

    # thin k (m = n), beta = 1, and beta = 0
    for name, kind, shapes in (
            ("dgemm_thin_k.csv", "main",
             [(2000, 2000, 32), (2000, 2000, 128), (2000, 2000, 256),
              (4000, 4000, 64)]),
            ("dgemm_beta0.csv", "beta0",
             [(256, 256, 256), (1024, 1024, 1024), (2048, 2048, 2048),
              (2000, 2000, 32)])):
        rows = []
        for m, n, k in shapes:
            rows.append(["%dx%dx%d" % (m, n, k)]
                        + [fmt(median(D, run=run, variant=var, set=kind,
                                      m=m, n=n, k=k)) for _, run, var in cols])
        write(os.path.join(S, name), ["shape"] + [c for c, _, _ in cols],
              rows)

    # block size sweep (run A), beta = 1
    sweep = ["avx-%d-%d" % (mc, kc) for mc in (72, 144, 288)
             for kc in (256, 384, 512)]
    shapes = sorted({(r["m"], r["n"], r["k"]) for r in D
                     if r["set"] == "sweep"}, key=lambda s: (s[2] < s[0], s))
    rows = [["%dx%dx%d" % s] + [fmt(median(D, run="A", variant=v,
                                            set="sweep", m=s[0], n=s[1],
                                            k=s[2])) for v in sweep]
            for s in shapes]
    write(os.path.join(S, "dgemm_sweep.csv"),
          ["shape"] + ["mc%d_kc%d" % tuple(map(int, v.split("-")[1:]))
                       for v in sweep], rows)

    # level 3: run B (sse, avx MC=288, blis, openblas), run A (MC 72/144/288)
    lcols = [("sse_mc384", "B", "sse"), ("avx_mc288", "B", "avx"),
             ("blis", "B", "blis"), ("openblas", "B", "openblas"),
             ("runA_avx_mc72", "A", "avx-72-256"),
             ("runA_avx_mc144", "A", "avx-144-256"),
             ("runA_avx_mc288", "A", "avx-288-256")]
    for routine in ("dsyrk", "dsymm", "dtrmm", "dtrsm"):
        rows = []
        for n in sorted({r["n"] for r in L if r["routine"] == routine}):
            v = {c: median(L, run=run, variant=var, routine=routine, n=n)
                 for c, run, var in lcols}
            rows.append([n] + [fmt(v[c]) for c, _, _ in lcols]
                        + [fmt(v["runA_avx_mc144"] / v["runA_avx_mc72"]),
                           fmt(v["runA_avx_mc288"] / v["runA_avx_mc72"])])
        write(os.path.join(S, "level3_%s.csv" % routine),
              ["n"] + [c for c, _, _ in lcols]
              + ["runA_mc144_over_mc72", "runA_mc288_over_mc72"], rows)


if __name__ == "__main__":
    main()
