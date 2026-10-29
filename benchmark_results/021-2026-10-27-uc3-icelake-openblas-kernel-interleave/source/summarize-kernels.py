#!/usr/bin/env python3
#
#  summarize-kernels.py REFS: converts the raw outputs of kernels-bench.sbatch
#  or kernels-heim.sh (data/raw/main-VARIANT-ROUND.txt) into CSV files
#  (data/runs.csv, data/summary/*.csv).  Run in the campaign directory.
#
#  REFS names the native reference of every micro kernel family, e.g.
#
#      haswell=blis-haswell,skx=blis-auto,zen4=aocl,zen5=aocl,avx=blis-haswell
#
#  A variant ulmblas-KERNEL[-TAG] (KERNEL = FAMILY_MRxNR or avx, sse) is
#  compared with the reference of its family (the longest matching prefix).
#
#  Statistics as in summarize-uc3.py: per variant and size the median,
#  minimum and maximum over the rounds of the per-round medians; a
#  comparison is "faster"/"slower" if the ranges over the rounds do not
#  overlap and the medians differ by at least 1%, else "noise".
#
#  data/summary/kernels_large.csv: for every ulmBLAS variant and every
#  square size n >= 1000 the ratio to its native reference and to
#  ulmblas-avx (ulmBLAS's own AVX/FMA kernel), and per variant the median,
#  minimum and maximum of the ratios with the rating of the median:
#  >= 0.95 "excellent", >= 0.90 "target", >= 0.85 "close", else
#  "investigate".
#
import csv, glob, os, re, statistics, sys

if len(sys.argv) != 2:
    sys.exit('usage: summarize-kernels.py FAMILY=REF[,FAMILY=REF...]')
REFS = dict(x.split('=') for x in sys.argv[1].split(','))

runs = []
for path in sorted(glob.glob('data/raw/main-*.txt')):
    m = re.match(r'main-(.+)-(\d+)\.txt$', os.path.basename(path))
    variant, rnd = m.group(1), int(m.group(2))
    for line in open(path):
        f = line.split()
        runs.append(dict(kind='main', variant=variant, round=rnd,
                         routine='dgemm', m=int(f[0]), n=int(f[1]), k=int(f[2]),
                         beta=float(f[3]), median=float(f[4]), min=float(f[5]),
                         max=float(f[6]), calls=int(f[7]), check=f[8]))

VARIANTS = sorted({r['variant'] for r in runs},
                  key=lambda v: (not v.startswith('ulmblas'), v))
COL = {v: v.replace('-', '_') for v in VARIANTS}

def kernel(v):
    m = re.match(r'ulmblas-(.+?)(?:-(k\d+|it\d+\w*|[a-z]+\d+))?$', v)
    return m.group(1) if m else None

def reference(v):
    k = kernel(v)
    if k is None:
        return None
    fam = max((f for f in REFS if k == f or k.startswith(f + '_')),
              key=len, default=None)
    ref = REFS.get(fam)
    return ref if ref in VARIANTS else None

def ownavx():
    c = [v for v in VARIANTS if kernel(v) == 'avx']
    return c[0] if c else None

os.makedirs('data/summary', exist_ok=True)
keys = ['kind', 'variant', 'round', 'routine', 'm', 'n', 'k', 'beta',
        'median', 'min', 'max', 'calls', 'check']
with open('data/runs.csv', 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=keys)
    w.writeheader()
    w.writerows(runs)

def stats(shape, variant):
    v = [r['median'] for r in runs
         if (r['m'], r['n'], r['k']) == shape and r['variant'] == variant]
    return (statistics.median(v), min(v), max(v), len(v)) if v else None

def clear(a, b):
    ratio = a[0]/b[0]
    if a[1] > b[2] and ratio >= 1.01:
        return 1
    if a[2] < b[1] and ratio <= 1/1.01:
        return -1
    return 0

shapes_all = sorted({(r['m'], r['n'], r['k']) for r in runs})

def table(name, shapes):
    head = ['m', 'n', 'k']
    for v in VARIANTS:
        head += [COL[v], COL[v] + '_min', COL[v] + '_max']
    for v in VARIANTS:
        if reference(v):
            head += [COL[v] + '_over_ref', COL[v] + '_vs_ref']
    head += ['best', 'best_clear']
    rows = []
    for s in shapes:
        st = {v: stats(s, v) for v in VARIANTS}
        row = list(s)
        for v in VARIANTS:
            row += (['%.3f' % st[v][0], '%.3f' % st[v][1], '%.3f' % st[v][2]]
                    if st[v] else ['NaN'] * 3)
        for v in VARIANTS:
            ref = reference(v)
            if ref:
                a, b = st[v], st[ref]
                row += ['%.3f' % (a[0]/b[0]),
                        {1: 'faster', -1: 'slower', 0: 'noise'}[clear(a, b)]]
        present = [v for v in VARIANTS if st[v]]
        best = max(present, key=lambda v: st[v][0])
        tied = [v for v in present if v != best and clear(st[best], st[v]) != 1]
        row += ['+'.join([best] + tied), 'yes' if not tied else 'no']
        rows.append(row)
    with open('data/summary/%s.csv' % name, 'w', newline='') as out:
        w = csv.writer(out)
        w.writerow(head)
        w.writerows(rows)

table('dgemm_square', [s for s in shapes_all if s[0] == s[1] == s[2]])
table('dgemm_thin_k', [s for s in shapes_all if not s[0] == s[1] == s[2]])

def rating(x):
    return ('excellent' if x >= 0.95 else 'target' if x >= 0.90
            else 'close' if x >= 0.85 else 'investigate')

large = [s for s in shapes_all if s[0] == s[1] == s[2] and s[0] >= 1000]
avx = ownavx()
with open('data/summary/kernels_large.csv', 'w', newline='') as out:
    w = csv.writer(out)
    w.writerow(['variant', 'kernel', 'reference']
               + ['n%d_gflops' % s[0] for s in large]
               + ['n%d_over_ref' % s[0] for s in large]
               + ['over_ref_median', 'over_ref_min', 'over_ref_max', 'rating',
                  'over_ulmblas_avx_median'])
    for v in VARIANTS:
        if kernel(v) is None:
            continue
        ref = reference(v)
        g = [stats(s, v)[0] for s in large]
        r = [stats(s, v)[0]/stats(s, ref)[0] for s in large] if ref else []
        a = ([stats(s, v)[0]/stats(s, avx)[0] for s in large]
             if avx and v != avx else [])
        w.writerow([v, kernel(v), ref or '']
                   + ['%.2f' % x for x in g]
                   + (['%.3f' % x for x in r] if r else [''] * len(large))
                   + (['%.3f' % statistics.median(r), '%.3f' % min(r),
                       '%.3f' % max(r), rating(statistics.median(r))]
                      if r else [''] * 4)
                   + ['%.3f' % statistics.median(a) if a else ''])

wrong = sum(1 for r in runs if r['check'] == 'WRONG')
print('%d measurements, %d WRONG, variants: %s' % (len(runs), wrong,
                                                   ' '.join(VARIANTS)))
