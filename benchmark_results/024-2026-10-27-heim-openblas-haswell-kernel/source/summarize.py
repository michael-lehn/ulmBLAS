#!/usr/bin/env python3
#
#  summarize.py: converts the raw outputs data/raw/E/SIZES-VARIANT-ROUND.txt
#  of the experiments E (bench2: m n k beta median min max calls check per
#  line) into
#
#    data/runs.csv           one line per program run and shape
#    data/summary/E.csv      per variant and shape: median, minimum and
#                            maximum over the rounds of the per-run medians,
#                            spread, time per call, ratio to native OpenBLAS
#                            (REF) and verdict
#
#  Verdict against REF: "faster"/"slower" if the ranges over the rounds do
#  not overlap and the medians differ by at least 1%, else "noise".  Run in
#  the campaign directory.
#
import csv, glob, os, re, statistics, sys

REF = 'openblas'

runs = []
for path in sorted(glob.glob('data/raw/*/*.txt')):
    E = os.path.basename(os.path.dirname(path))
    mt = re.match(r'(main|small)-(.+)-(\d+)\.txt$', os.path.basename(path))
    if not mt:
        continue
    sizes, v, r = mt.group(1), mt.group(2), int(mt.group(3))
    for line in open(path):
        f = line.split()
        if len(f) != 9:
            sys.exit('unexpected line in %s: %s' % (path, line))
        m, n, k = (int(x) for x in f[:3])
        med = float(f[4])
        runs.append(dict(experiment=E, sizes=sizes, variant=v, round=r,
                         m=m, n=n, k=k, beta=f[3], gflops=med,
                         gflops_min=float(f[5]), gflops_max=float(f[6]),
                         us_per_call='%.3f' % (2.0*m*n*k/med*1e-3),
                         calls_per_sample=int(f[7]), check=f[8]))

with open('data/runs.csv', 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=list(runs[0].keys()))
    w.writeheader()
    w.writerows(runs)

def stats(x):
    return statistics.median(x), min(x), max(x)

def verdict(a, b):
    ratio = a[0]/b[0]
    if a[1] > b[2] and ratio >= 1.01:
        return ratio, 'faster'
    if a[2] < b[1] and ratio <= 1/1.01:
        return ratio, 'slower'
    return ratio, 'noise'

os.makedirs('data/summary', exist_ok=True)
for E in sorted({r['experiment'] for r in runs}):
    st = {}
    for r in runs:
        if r['experiment'] == E:
            st.setdefault((r['variant'], r['m'], r['n'], r['k']), []).append(r['gflops'])
    rows = []
    for (v, m, n, k), x in sorted(st.items(), key=lambda t: (t[0][1:], t[0][0])):
        s = stats(x)
        row = dict(variant=v, m=m, n=n, k=k, gflops='%.3f' % s[0],
                   gflops_min='%.3f' % s[1], gflops_max='%.3f' % s[2],
                   rounds=len(x), spread_pct='%.2f' % (100*(s[2]-s[1])/s[0]),
                   us_per_call='%.3f' % (2.0*m*n*k/s[0]*1e-3))
        ref = st.get((REF, m, n, k))
        if ref:
            ratio, vd = verdict(s, stats(ref))
            row.update(vs_openblas='%.4f' % ratio, verdict=vd)
        else:
            row.update(vs_openblas='NaN', verdict='')
        rows.append(row)
    with open('data/summary/%s.csv' % E, 'w', newline='') as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    R = [r for r in runs if r['experiment'] == E]
    print('%s: %d results, %d variants, %d shapes, %d WRONG' % (
        E, len(R), len({r['variant'] for r in R}),
        len({(r['m'], r['n'], r['k']) for r in R}),
        sum(r['check'] == 'WRONG' for r in R)))

#  Profile (prof-hsw, experiment prof): data/raw/prof/prof-N-R.txt and
#  cblas-N-R.txt, one line per mode: mode order L2 KC L3 m n k GFLOPS
#  t_call packB packA mgemm other PA:PB.  data/summary/prof.csv: per mode
#  and size the median, minimum and maximum over the rounds of GFLOP/s and
#  the medians of the time fractions.
prof = {}
for path in sorted(glob.glob('data/raw/prof/*-*-*.txt')):
    mt = re.match(r'(prof|cblas)-(\d+)-(\d+)\.txt$', os.path.basename(path))
    if not mt:
        continue
    for line in open(path):
        f = line.split()
        if len(f) < 14 or f[0] in ('mcheck',):
            continue
        key = (f[0], int(f[5]), int(f[6]), int(f[7]))
        prof.setdefault(key, []).append((float(f[8]), float(f[10]),
                                         float(f[11]), float(f[12])))
if prof:
    order = ['lib', 'inst', 'nopack', 'macro', 'macronopack', 'kern', 'cblas']
    rows = []
    for key in sorted(prof, key=lambda t: (t[1:], order.index(t[0]))):
        x = prof[key]
        g = [y[0] for y in x]
        rows.append(dict(mode=key[0], m=key[1], n=key[2], k=key[3],
                         gflops='%.3f' % statistics.median(g),
                         gflops_min='%.3f' % min(g), gflops_max='%.3f' % max(g),
                         rounds=len(x),
                         packB='%.4f' % statistics.median(y[1] for y in x),
                         packA='%.4f' % statistics.median(y[2] for y in x),
                         macrokernel='%.4f' % statistics.median(y[3] for y in x)))
    with open('data/summary/prof.csv', 'w', newline='') as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('prof: %d modes x sizes' % len(rows))
