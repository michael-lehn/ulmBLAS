#!/usr/bin/env python3
#
#  summarize-uc3.py: converts the raw outputs of bench.sbatch
#  (data/raw/*.txt) into CSV files (data/runs.csv, data/summary/*.csv).
#  Run in the campaign directory.
#
#  As summarize-eigen.py of campaign 009-2026-10-26-heim-eigen: for every size
#  and variant the median over the rounds of the per-round medians, with
#  the minimum and maximum over the rounds.  A comparison of two variants is
#  "faster" or "slower" if the ranges over the rounds do not overlap and the
#  medians differ by at least 1%, otherwise "noise".  The column "best"
#  names the variant with the highest median, "best_clear" says whether it
#  is faster than every other variant in that sense (else the variants
#  within the noise of the best are listed, joined by '+').
#
#  Additionally: ulmBLAS relative to every other variant, and every variant
#  relative to the FMA peak of the core with 256-bit vectors (peak_ymm) and,
#  where available, 512-bit vectors (peak_zmm), as measured by peak.c
#  before and after the rounds (data/raw/peak-*.txt; the higher value).
#
import csv, glob, os, re, statistics

VARIANTS = ['ulmblas', 'blis-auto', 'blis-haswell', 'openblas', 'mkl',
            'aocl', 'eigen']
COL = {v: v.replace('-', '_') for v in VARIANTS}

def read(paths):
    runs = []
    for path in sorted(paths):
        m = re.match(r'(main|beta0|sweep)-(.+)-(\d+)\.txt$',
                     os.path.basename(path))
        if not m:
            continue
        kind, variant, rnd = m.group(1), m.group(2), int(m.group(3))
        for line in open(path):
            f = line.split()
            runs.append(dict(kind=kind, variant=variant, round=rnd,
                             routine='dgemm', m=int(f[0]), n=int(f[1]),
                             k=int(f[2]), beta=float(f[3]),
                             median=float(f[4]), min=float(f[5]),
                             max=float(f[6]), calls=int(f[7]), check=f[8]))
    return runs

def peak():
    best = {}
    for path in glob.glob('data/raw/peak-*.txt'):
        for line in open(path):
            f = line.split()
            if f and f[0] in ('fma-ymm', 'fma-zmm'):
                best[f[0]] = max(best.get(f[0], 0.0), float(f[1]))
    return best.get('fma-ymm'), best.get('fma-zmm')

runs = read(glob.glob('data/raw/*.txt'))
VARIANTS = [v for v in VARIANTS if any(r['variant'] == v for r in runs)]
PEAK_YMM, PEAK_ZMM = peak()

os.makedirs('data/summary', exist_ok=True)
keys = ['kind', 'variant', 'round', 'routine', 'm', 'n', 'k', 'beta',
        'median', 'min', 'max', 'calls', 'check']
with open('data/runs.csv', 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=keys)
    w.writeheader()
    for r in runs:
        w.writerow(r)

def stats(runs, kind, shape, variant):
    v = [r['median'] for r in runs
         if r['kind'] == kind and (r['m'], r['n'], r['k']) == shape
         and r['variant'] == variant]
    if not v:
        return None
    return statistics.median(v), min(v), max(v), len(v)

def clear(a, b):
    """+1 if a is clearly faster than b, -1 if clearly slower, else 0"""
    ratio = a[0]/b[0]
    if a[1] > b[2] and ratio >= 1.01:
        return 1
    if a[2] < b[1] and ratio <= 1/1.01:
        return -1
    return 0

def verdict(a, b):
    if a is None or b is None:
        return 'NaN', ''
    return '%.3f' % (a[0]/b[0]), {1: 'faster', -1: 'slower', 0: 'noise'}[clear(a, b)]

PAIRS = [('ulmblas', v) for v in VARIANTS if v != 'ulmblas']

def table(name, kind, shapes):
    head = ['m', 'n', 'k']
    for v in VARIANTS:
        c = COL[v]
        head += [c, c+'_min', c+'_max', c+'_us', c+'_of_peak_ymm']
        if PEAK_ZMM:
            head += [c+'_of_peak_zmm']
    for a, b in PAIRS:
        head += ['%s_over_%s' % (COL[a], COL[b]),
                 '%s_vs_%s' % (COL[a], COL[b])]
    head += ['best', 'best_clear']
    rows = []
    for s in shapes:
        row = list(s)
        st = {}
        for v in VARIANTS:
            x = stats(runs, kind, s, v)
            st[v] = x
            if x is None:
                row += ['NaN']*(5 + (1 if PEAK_ZMM else 0))
            else:
                row += ['%.3f' % x[0], '%.3f' % x[1], '%.3f' % x[2],
                        '%.4f' % (2.0*s[0]*s[1]*s[2]/x[0]*1e-3),
                        '%.3f' % (x[0]/PEAK_YMM)]
                if PEAK_ZMM:
                    row += ['%.3f' % (x[0]/PEAK_ZMM)]
        for a, b in PAIRS:
            row += list(verdict(st[a], st[b]))
        present = [v for v in VARIANTS if st[v]]
        best = max(present, key=lambda v: st[v][0])
        tied = [v for v in present
                if v != best and clear(st[best], st[v]) != 1]
        row += ['+'.join([best] + tied), 'yes' if not tied else 'no']
        rows.append(row)
    with open('data/summary/%s.csv' % name, 'w', newline='') as out:
        w = csv.writer(out)
        w.writerow(head)
        w.writerows(rows)

def shapes(runs, kind, square=None):
    out = sorted({(r['m'], r['n'], r['k']) for r in runs
                  if r['kind'] == kind})
    if square is not None:
        out = [s for s in out if (s[0] == s[1] == s[2]) == square]
    return out

table('dgemm_sweep', 'sweep', shapes(runs, 'sweep'))
table('dgemm_square', 'main', shapes(runs, 'main', square=True))
table('dgemm_thin_k', 'main', shapes(runs, 'main', square=False))
table('dgemm_beta0', 'beta0', shapes(runs, 'beta0'))

with open('data/summary/peak.csv', 'w', newline='') as out:
    w = csv.writer(out)
    w.writerow(['vector', 'gflops'])
    w.writerow(['ymm', '%.2f' % PEAK_YMM])
    if PEAK_ZMM:
        w.writerow(['zmm', '%.2f' % PEAK_ZMM])

wrong = sum(1 for r in runs if r['check'] == 'WRONG')
print('%d measurements, %d WRONG; peak ymm %s, zmm %s GFLOP/s'
      % (len(runs), wrong, PEAK_YMM, PEAK_ZMM))
