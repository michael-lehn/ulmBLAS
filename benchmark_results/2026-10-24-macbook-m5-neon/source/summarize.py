#!/usr/bin/env python3
#
#  summarize.py: converts the raw outputs of the benchmark programs
#  (data/raw/run-*/*.txt, written by bench2, bench3, ukernel and peak, see
#  README.md) into
#
#      data/dgemm.csv     all dgemm measurements (one line per size, variant
#                         and round or cycle)
#      data/level3.csv    all dsyrk, dsymm, dtrmm, dtrsm measurements
#      data/summary/*.csv medians over the rounds, the input of the plots
#
#  Usage (from the campaign directory):  python3 source/summarize.py
#
#  The raw files are named KIND-VARIANT-ROUND.txt (VARIANT may contain '-').
#  The block sizes of each variant are listed in VARIANTS (from
#  ulmblas/config/blocksize.h of the measured source, from the BLIS and
#  OpenBLAS sources).  Nothing is interpolated: a summary entry is the
#  median of the rounds of exactly that measurement, missing measurements
#  are written as NaN.  Run A0 (invalid, thermal throttling) is converted
#  into data/dgemm.csv, but not used for any summary.
#
import csv
import glob
import os
import statistics
import sys

NAN = float("nan")

#  variant -> library, kernel, MR, NR, MC, KC, NC
VARIANTS = {
    "base":       ("ulmBLAS 8266ae3", "reference (C++)", 4, 4, 384, 384, 4096),
    "ref":        ("ulmBLAS", "reference (C++), -DULMBLAS_NO_NEON", 4, 4,
                   384, 384, 4096),
    "neon":       ("ulmBLAS", "NEON 6x8", 6, 8, 960, 1024, 4080),
    "neon-gcc":   ("ulmBLAS", "NEON 6x8, GCC 16", 6, 8, 960, 1024, 4080),
    "blis":       ("BLIS 2.0", "firestorm armv8a 8x6r", 8, 6, 256, 3072,
                   8184),
    "openblas":   ("OpenBLAS 0.3.34", "VORTEXM4 (NEON) 8x4", 8, 4, 256, 512,
                   4096),
    "accelerate": ("Apple Accelerate", "SME/AMX (matrix unit)", "", "", "",
                   "", ""),
}
for mc in (120, 240, 480, 960):
    for kc in (256, 512, 1024, 2048):
        VARIANTS["neon-%d-%d" % (mc, kc)] = ("ulmBLAS", "NEON 6x8", 6, 8,
                                             mc, kc, 4080)

HEADER = ["run", "library", "variant", "kernel", "MR", "NR", "MC", "KC", "NC"]
#  kinds of raw files with level 3 (bench3) output
LEVEL3_KINDS = ("l3", "l3n512", "l3n1024", "l3n2048", "l3n3072")
SAMPLES = {"l3": 5, "l3n512": 5, "l3n1024": 5, "l3n2048": 5, "l3n3072": 5}


def read_raw(top):
    dgemm, level3 = [], []
    for path in sorted(glob.glob(os.path.join(top, "data/raw/run-*/*.txt"))):
        run = os.path.basename(os.path.dirname(path))[len("run-"):]
        kind, rest = os.path.basename(path)[:-4].split("-", 1)
        variant, rnd = rest.rsplit("-", 1)
        meta = [run, VARIANTS[variant][0], variant] \
            + list(VARIANTS[variant][1:])
        samples = SAMPLES.get(kind, 7)
        for line in open(path):
            x = line.split()
            if kind in LEVEL3_KINDS:
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


def values(rows, **key):
    return [r["gflops_median"] for r in rows
            if all(r[k] == v for k, v in key.items())]


def median(rows, **key):
    v = values(rows, **key)
    return statistics.median(v) if v else NAN


def spread(rows, **key):
    """(max - min) / median of the round medians, in percent"""
    v = values(rows, **key)
    return 100 * (max(v) - min(v)) / statistics.median(v) if v else NAN


def fmt(x):
    return "NaN" if x != x else "%.3f" % x


def ratio(a, b):
    return a / b if b == b and b != 0 else NAN


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

    # run B: square dgemm, beta = 1 (sets small, mid, sq*)
    cols = [("base", "base"), ("ref", "ref"), ("neon", "neon"),
            ("neon_gcc", "neon-gcc"), ("blis", "blis"),
            ("openblas", "openblas"), ("accelerate", "accelerate")]
    B = [r for r in D if r["run"] == "B" and r["beta"] == 1.0]
    sizes = sorted({r["n"] for r in B if r["m"] == r["n"] == r["k"]})
    rows = []
    for n in sizes:
        v = {c: median(B, variant=var, m=n, n=n, k=n) for c, var in cols}
        rows.append([n] + [fmt(v[c]) for c, _ in cols]
                    + [fmt(ratio(v["neon"], v["base"])),
                       fmt(ratio(v["neon"], v["blis"])),
                       fmt(ratio(v["neon"], v["openblas"])),
                       fmt(ratio(v["blis"], v["openblas"])),
                       fmt(ratio(v["base"], v["openblas"])),
                       fmt(ratio(v["neon_gcc"], v["neon"])),
                       fmt(ratio(v["ref"], v["base"]))])
    write(os.path.join(S, "dgemm_square.csv"),
          ["n"] + [c for c, _ in cols]
          + ["neon_over_base", "neon_over_blis", "neon_over_openblas",
             "blis_over_openblas", "base_over_openblas",
             "neon_gcc_over_neon", "ref_over_base"], rows)

    # run B: thin k (m = n), beta = 1, and beta = 0
    for name, beta, shapes in (
            ("dgemm_thin_k.csv", 1.0,
             [(2000, 2000, 32), (2000, 2000, 128), (2000, 2000, 256),
              (4000, 4000, 64)]),
            ("dgemm_beta0.csv", 0.0,
             [(256, 256, 256), (1024, 1024, 1024), (2048, 2048, 2048),
              (2000, 2000, 32)])):
        R = [r for r in D if r["run"] == "B" and r["beta"] == beta]
        rows = []
        for m, n, k in shapes:
            rows.append(["%dx%dx%d" % (m, n, k)]
                        + [fmt(median(R, variant=var, m=m, n=n, k=k))
                           for _, var in cols])
        write(os.path.join(S, name), ["shape"] + [c for c, _ in cols], rows)

    # run B: level 3
    for routine in ("dsyrk", "dsymm", "dtrmm", "dtrsm"):
        R = [r for r in L if r["run"] == "B" and r["routine"] == routine]
        rows = []
        for n in sorted({r["n"] for r in R}):
            v = {c: median(R, variant=var, n=n) for c, var in cols}
            rows.append([n] + [fmt(v[c]) for c, _ in cols]
                        + [fmt(ratio(v["neon"], v["base"])),
                           fmt(ratio(v["neon"], v["blis"])),
                           fmt(ratio(v["neon"], v["openblas"]))])
        write(os.path.join(S, "level3_%s.csv" % routine),
              ["n"] + [c for c, _ in cols]
              + ["neon_over_base", "neon_over_blis", "neon_over_openblas"],
              rows)

    # run B: variability (spread of the round medians, beta = 1, square)
    #  and of rounds 2 and 3 only (round 1 was disturbed by other load on
    #  the machine for all variants, see README.md)
    rows = []
    for n in sizes:
        rows.append([n] + [fmt(spread(B, variant=var, m=n, n=n, k=n))
                           for _, var in cols]
                    + [fmt(spread([r for r in B if r["round"] > 1],
                                  variant=var, m=n, n=n, k=n))
                       for _, var in cols])
    write(os.path.join(S, "variability_square.csv"),
          ["n"] + ["%s_spread_percent" % c for c, _ in cols]
          + ["%s_spread_rounds23_percent" % c for c, _ in cols], rows)

    # run A: block size sweep (medians of 3 rounds)
    A = [r for r in D if r["run"] == "A"]
    sweep = sorted({r["variant"] for r in A},
                   key=lambda v: tuple(map(int, v.split("-")[1:])))
    shapes = sorted({(r["m"], r["n"], r["k"]) for r in A},
                    key=lambda s: (s[2] < s[0], s))
    rows = [["%dx%dx%d" % s] + [fmt(median(A, variant=v, m=s[0], n=s[1],
                                           k=s[2])) for v in sweep]
            for s in shapes]
    write(os.path.join(S, "dgemm_sweep.csv"),
          ["shape"] + ["mc%s_kc%s" % tuple(v.split("-")[1:]) for v in sweep],
          rows)

    # run A2: interleaved comparison of candidates (medians of the cycles),
    # relative to MC = 240, KC = 256
    A2 = [r for r in D if r["run"] == "A2"]
    cand = sorted({r["variant"] for r in A2},
                  key=lambda v: (v != "neon-240-256",
                                 tuple(map(int, v.split("-")[1:]))))
    shapes = sorted({(r["m"], r["n"], r["k"]) for r in A2},
                    key=lambda s: (s[2] < s[0], s))
    rows = []
    for s in shapes:
        v = [median(A2, variant=c, m=s[0], n=s[1], k=s[2]) for c in cand]
        rows.append(["%dx%dx%d" % s] + [fmt(x) for x in v]
                    + [fmt(ratio(x, v[0])) for x in v])
    names = ["mc%s_kc%s" % tuple(c.split("-")[1:]) for c in cand]
    write(os.path.join(S, "blocksize_dgemm.csv"),
          ["shape"] + names + [n + "_rel" for n in names], rows)

    # run A3: level 3 routines, interleaved (medians of the cycles),
    # relative to MC = 240, KC = 256
    A3 = [r for r in L if r["run"] == "A3"]
    cand = sorted({r["variant"] for r in A3},
                  key=lambda v: (v != "neon-240-256",
                                 tuple(map(int, v.split("-")[1:]))))
    rows = []
    for routine in ("dsyrk", "dsymm", "dtrmm", "dtrsm"):
        for n in sorted({r["n"] for r in A3 if r["routine"] == routine}):
            v = [median(A3, variant=c, routine=routine, n=n) for c in cand]
            rows.append(["%s n=%d" % (routine, n)] + [fmt(x) for x in v]
                        + [fmt(ratio(x, v[0])) for x in v])
    names = ["mc%s_kc%s" % tuple(c.split("-")[1:]) for c in cand]
    write(os.path.join(S, "blocksize_level3.csv"),
          ["case"] + names + [n + "_rel" for n in names], rows)

    # micro kernel throughput (data/raw/ukernel/*.txt: kc MR NR med min max)
    rows = []
    for path in sorted(glob.glob(os.path.join(top, "data/raw/ukernel/*.txt"))):
        variant = os.path.basename(path)[:-4]
        for line in open(path):
            x = line.split()
            if x[0] == "peak":
                continue
            rows.append([variant, int(x[0]), int(x[1]), int(x[2]),
                         float(x[3]), float(x[4]), float(x[5])])
    write(os.path.join(S, "ukernel.csv"),
          ["variant", "kc", "MR", "NR", "gflops_median", "gflops_min",
           "gflops_max"], rows)

    # FMA peak measured at the start of the rounds and after each variant
    # (lines "... peak X GFLOP/s, thermal ..." of data/logs/run-*.log)
    rows = []
    for path in sorted(glob.glob(os.path.join(top, "data/logs/run-*.log"))):
        run = os.path.basename(path)[len("run-"):].split("-")[0]
        for line in open(path):
            if "peak " in line and " GFLOP/s" in line:
                x = line.split("peak ")[1].split()
                rows.append([run, float(x[0])])
    write(os.path.join(S, "peak.csv"), ["run", "gflops"], rows)

    # thermal diagnosis: sustained dgemm n = 2048 (data/raw/diag)
    diag = os.path.join(top, "data/raw/diag")
    if os.path.exists(os.path.join(diag, "sustained.txt")):
        def secs(t):
            h, m, s = map(int, t.split(":"))
            return 3600 * h + 60 * m + s
        state = [(secs(l.split()[0]), l.split()[2])
                 for l in open(os.path.join(diag, "thermal-sustained.txt"))]
        lines = [l.split() for l in open(os.path.join(diag, "sustained.txt"))]
        t0 = secs(lines[0][0])
        rows = []
        for x in lines:
            t = secs(x[0])
            st = [s for ts, s in state if ts <= t]
            rows.append([t - t0, float(x[5]),
                         1 if (st and st[-1] == "fair") else 0])
        write(os.path.join(S, "thermal_sustained.csv"),
              ["seconds", "gflops", "thermal_fair"], rows)


if __name__ == "__main__":
    main()
