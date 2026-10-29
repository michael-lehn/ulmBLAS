#!/usr/bin/env python3
#
#  summarize-full.py: converts the raw outputs of full-bench.sbatch
#  (data/raw/main-VARIANT-ROUND.txt, verify-VARIANT.txt) into the CSV files
#  of the campaign and prints the summary used in its README.  Run in the
#  campaign directory:
#
#      python3 source/summarize-full.py
#
#  Statistics: bench2.c reports the median of 7 samples per size; per size
#  and variant the result is the median over the rounds of these medians,
#  with the minimum and maximum over the rounds.  Ratios are computed per
#  round (the variants of a round are measured within the same hour, in
#  rotated order) and summarized the same way.
#
#  Writes
#      data/runs.csv                 every measurement (variant, round, n, ...)
#      data/summary/dgemm.csv        n, VARIANT, VARIANT_min, VARIANT_max
#      data/summary/ratios.csv       n, ULMBLAS_over_REF (+ _min, _max) for the
#                                    ulmBLAS variants and REF = openblas, mkl
#      data/summary/ranges.csv       size range, variant, median GFLOP/s and
#                                    median ratios to OpenBLAS and oneMKL
#
import csv
import glob
import os
import re
import statistics as st
from collections import defaultdict

ULMBLAS = ['ulmblas-openblas-skylakex_16x12-it3',
           'ulmblas-openblas-skylakex_16x12-it2',
           'ulmblas-openblas-skylakex_16x12-it1',
           'ulmblas-zen4_32x6-f1', 'ulmblas-skx_16x14-f1', 'ulmblas-avx-f1']
NATIVE = ['openblas', 'mkl', 'blis-auto', 'eigen', 'aocl']
VARIANTS = ULMBLAS + NATIVE
REFS = ['openblas', 'mkl']
RANGES = [(50, 250), (300, 500), (550, 1000), (1050, 2000), (2050, 4000)]
col = lambda v: v.replace('-', '_')

# raw data
runs = []
data = defaultdict(dict)            # (variant, n) -> {round: median}
for f in sorted(glob.glob('data/raw/main-*.txt')):
    m = re.match(r'main-(.+)-(\d+)\.txt$', os.path.basename(f))
    v, r = m.group(1), int(m.group(2))
    if v not in VARIANTS:
        raise SystemExit('unknown variant %s' % v)
    for line in open(f):
        x = line.split()
        if len(x) != 9:
            continue
        mm, n, k = int(x[0]), int(x[1]), int(x[2])
        if not (mm == n == k):
            raise SystemExit('%s: not square: %s' % (f, line))
        runs.append(dict(variant=v, round=r, n=n, beta=x[3], median=x[4],
                         min=x[5], max=x[6], calls=x[7], check=x[8]))
        data[(v, n)][r] = float(x[4])
rounds = sorted({r['round'] for r in runs})
sizes = sorted({r['n'] for r in runs})
wrong = sum(r['check'] == 'WRONG' for r in runs)
checked = sum(r['check'] == 'ok' for r in runs)
missing = [(v, n, r) for v in VARIANTS for n in sizes for r in rounds
           if r not in data[(v, n)]]

os.makedirs('data/summary', exist_ok=True)
with open('data/runs.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
    w.writeheader()
    for r in sorted(runs, key=lambda r: (r['variant'], r['round'], r['n'])):
        w.writerow(r)

def med(v, n):
    x = list(data[(v, n)].values())
    return st.median(x), min(x), max(x)

def ratio(v, ref, n):
    x = [data[(v, n)][r] / data[(ref, n)][r] for r in rounds
         if r in data[(v, n)] and r in data[(ref, n)]]
    return st.median(x), min(x), max(x)

with open('data/summary/dgemm.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['n'] + [c for v in VARIANTS
                        for c in (col(v), col(v) + '_min', col(v) + '_max')])
    for n in sizes:
        w.writerow([n] + ['%.3f' % y for v in VARIANTS for y in med(v, n)])

with open('data/summary/ratios.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['n'] + [c for ref in REFS for v in ULMBLAS
                        for c in ('%s_over_%s' % (col(v), ref),
                                  '%s_over_%s_min' % (col(v), ref),
                                  '%s_over_%s_max' % (col(v), ref))])
    for n in sizes:
        w.writerow([n] + ['%.4f' % y for ref in REFS for v in ULMBLAS
                          for y in ratio(v, ref, n)])

with open('data/summary/ranges.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['n_from', 'n_to', 'sizes', 'variant', 'gflops_median',
                'gflops_min', 'gflops_max', 'over_openblas_median',
                'over_openblas_min', 'over_openblas_max', 'over_mkl_median',
                'over_mkl_min', 'over_mkl_max'])
    for a, b in RANGES:
        ns = [n for n in sizes if a <= n <= b]
        if not ns:
            continue
        for v in VARIANTS:
            g = [med(v, n)[0] for n in ns]
            row = [a, b, len(ns), v, '%.2f' % st.median(g), '%.2f' % min(g),
                   '%.2f' % max(g)]
            for ref in REFS:
                q = [ratio(v, ref, n)[0] for n in ns]
                row += ['%.3f' % st.median(q), '%.3f' % min(q), '%.3f' % max(q)]
            w.writerow(row)

# report
print('%d measurements (%d variants x %d sizes x %d rounds), %d checked '
      'against the naive product, %d WRONG, %d missing'
      % (len(runs), len(VARIANTS), len(sizes), len(rounds), checked, wrong,
         len(missing)))
viol = 0
for f in sorted(glob.glob('data/raw/verify-*.txt')):
    for line in open(f):
        m = re.search(r'(\d+) violations', line)
        if m:
            viol += int(m.group(1))
print('verify: %d files, %d violations' % (len(glob.glob('data/raw/verify-*.txt')), viol))

print('\nrepeatability: (max - min) / median over the rounds, per size')
for v in VARIANTS:
    s = [(med(v, n)[2] - med(v, n)[1]) / med(v, n)[0] for n in sizes]
    big = [n for n, x in zip(sizes, s) if x > 0.02]
    print('  %-38s median %.4f, max %.4f, sizes above 2%%: %s'
          % (v, st.median(s), max(s), big or 'none'))

print('\nsize ranges: median GFLOP/s (min ... max) / OpenBLAS / oneMKL')
for a, b in RANGES:
    print('  n = %d ... %d' % (a, b))
    for r in csv.DictReader(open('data/summary/ranges.csv')):
        if int(r['n_from']) == a:
            print('    %-38s %6s (%6s ... %6s)  %s  %s'
                  % (r['variant'], r['gflops_median'], r['gflops_min'],
                     r['gflops_max'], r['over_openblas_median'],
                     r['over_mkl_median']))

print('\nfastest per size (median), with ulmBLAS it3 rank')
best = defaultdict(list)
for n in sizes:
    order = sorted(VARIANTS, key=lambda v: -med(v, n)[0])
    best[order[0]].append(n)
for v, ns in best.items():
    print('  %-38s %d sizes: %s' % (v, len(ns), ns))

def crossings(v, ref):
    # sizes where v is faster than ref in every round, slower in every
    # round, or neither (the rounds disagree)
    out = {'faster': [], 'slower': [], 'mixed': []}
    for n in sizes:
        q = ratio(v, ref, n)
        out['faster' if q[1] > 1 else 'slower' if q[2] < 1 else 'mixed'].append(n)
    return out

print('\ncomparisons (faster/slower in every round, mixed: rounds disagree)')
for v, ref in [('ulmblas-openblas-skylakex_16x12-it3', 'openblas'),
               ('ulmblas-openblas-skylakex_16x12-it3', 'mkl'),
               ('ulmblas-openblas-skylakex_16x12-it3', 'eigen'),
               ('ulmblas-openblas-skylakex_16x12-it3', 'blis-auto'),
               ('ulmblas-zen4_32x6-f1', 'openblas'),
               ('ulmblas-openblas-skylakex_16x12-it3', 'ulmblas-zen4_32x6-f1'),
               ('mkl', 'openblas'), ('eigen', 'openblas')]:
    c = crossings(v, ref)
    print('  %s vs %s:' % (v, ref))
    for k in ('faster', 'slower', 'mixed'):
        print('    %-6s %2d: %s' % (k, len(c[k]), c[k]))

print('\nn >= 1000: ratios (median, min ... max over sizes)')
large = [n for n in sizes if n >= 1000]
for v in ULMBLAS + ['mkl', 'eigen', 'blis-auto', 'aocl']:
    for ref in REFS:
        if v == ref:
            continue
        q = [ratio(v, ref, n)[0] for n in large]
        g = [med(v, n)[0] for n in large]
        print('  %-38s / %-8s %.3f (%.3f ... %.3f)   %.1f ... %.1f GFLOP/s'
              % (v, ref, st.median(q), min(q), max(q), min(g), max(g)))
