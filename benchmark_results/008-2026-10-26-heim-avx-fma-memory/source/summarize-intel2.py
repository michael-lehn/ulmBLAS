#!/usr/bin/env python3
#
#  summarize-intel2.py: converts the raw outputs of bench-intel2.sh
#  (data/raw/*.txt) into CSV files (data/runs.csv, data/summary/*.csv).
#  Run in the campaign directory.
#
#  For every size and variant the median over the rounds of the per-round
#  medians is reported, with the minimum and maximum over the rounds.  A
#  comparison of two variants is "faster" or "slower" if the ranges over the
#  rounds do not overlap (the minimum of one exceeds the maximum of the
#  other) and the medians differ by at least 1%, otherwise "noise".
#
import csv, glob, os, re, statistics

VARIANTS = ['avx-pub', 'avx-pre', 'avx-cache', 'avx-cur', 'sse-pub',
            'sse-cur', 'blis', 'openblas']
COL = {v: v.replace('-', '_') for v in VARIANTS}

runs = []
for path in sorted(glob.glob('data/raw/*.txt')):
    m = re.match(r'(main|beta0|sweep|l3)-(.+)-(\d+)\.txt$',
                 os.path.basename(path))
    if not m:
        continue
    kind, variant, rnd = m.group(1), m.group(2), int(m.group(3))
    for line in open(path):
        f = line.split()
        if kind == 'l3':
            runs.append(dict(kind=kind, variant=variant, round=rnd,
                             routine=f[0], m=int(f[1]), n=int(f[1]),
                             k=int(f[1]), beta=1, median=float(f[2]),
                             min=float(f[3]), max=float(f[4]),
                             calls=int(f[5]), check=''))
        else:
            runs.append(dict(kind=kind, variant=variant, round=rnd,
                             routine='dgemm', m=int(f[0]), n=int(f[1]),
                             k=int(f[2]), beta=float(f[3]),
                             median=float(f[4]), min=float(f[5]),
                             max=float(f[6]), calls=int(f[7]), check=f[8]))

os.makedirs('data/summary', exist_ok=True)
keys = ['kind', 'variant', 'round', 'routine', 'm', 'n', 'k', 'beta',
        'median', 'min', 'max', 'calls', 'check']
with open('data/runs.csv', 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=keys)
    w.writeheader()
    for r in runs:
        w.writerow(r)

def stats(kind, routine, shape, variant):
    v = [r['median'] for r in runs
         if r['kind'] == kind and r['routine'] == routine
         and (r['m'], r['n'], r['k']) == shape and r['variant'] == variant]
    if not v:
        return None
    return statistics.median(v), min(v), max(v), len(v)

def verdict(a, b):
    if a is None or b is None:
        return 'NaN', ''
    ratio = a[0]/b[0]
    if a[1] > b[2] and ratio >= 1.01:
        return '%.3f' % ratio, 'faster'
    if a[2] < b[1] and ratio <= 1/1.01:
        return '%.3f' % ratio, 'slower'
    return '%.3f' % ratio, 'noise'

PAIRS = [('avx-cur', 'avx-pub'), ('avx-cache', 'avx-pre'),
         ('avx-pre', 'avx-pub'), ('avx-cur', 'avx-cache'),
         ('sse-cur', 'sse-pub'), ('avx-cur', 'blis'),
         ('avx-cur', 'openblas'), ('avx-pub', 'blis'),
         ('avx-pub', 'openblas')]

def table(name, kind, routine, shapes, flops):
    head = ['m', 'n', 'k']
    for v in VARIANTS:
        c = COL[v]
        head += [c, c+'_min', c+'_max', c+'_us']
    for a, b in PAIRS:
        head += ['%s_over_%s' % (COL[a], COL[b]),
                 '%s_vs_%s' % (COL[a], COL[b])]
    rows = []
    for s in shapes:
        row = list(s)
        st = {}
        for v in VARIANTS:
            x = stats(kind, routine, s, v)
            st[v] = x
            if x is None:
                row += ['NaN']*4
            else:
                row += ['%.3f' % x[0], '%.3f' % x[1], '%.3f' % x[2],
                        '%.4f' % (flops(*s)/x[0]*1e-3)]
        for a, b in PAIRS:
            row += list(verdict(st[a], st[b]))
        rows.append(row)
    with open('data/summary/%s.csv' % name, 'w', newline='') as out:
        w = csv.writer(out)
        w.writerow(head)
        w.writerows(rows)

def shapes(kind, routine='dgemm', square=None):
    out = sorted({(r['m'], r['n'], r['k']) for r in runs
                  if r['kind'] == kind and r['routine'] == routine})
    if square is not None:
        out = [s for s in out if (s[0] == s[1] == s[2]) == square]
    return out

gemm = lambda m, n, k: 2.0*m*n*k
table('dgemm_sweep', 'sweep', 'dgemm', shapes('sweep'), gemm)
table('dgemm_square', 'main', 'dgemm', shapes('main', square=True), gemm)
table('dgemm_thin_k', 'main', 'dgemm', shapes('main', square=False), gemm)
table('dgemm_beta0', 'beta0', 'dgemm', shapes('beta0'), gemm)
for routine, f in (('dsyrk', lambda m, n, k: 1.0*n**3),
                   ('dsymm', lambda m, n, k: 2.0*n**3),
                   ('dtrmm', lambda m, n, k: 1.0*n**3),
                   ('dtrsm', lambda m, n, k: 1.0*n**3)):
    table('level3_' + routine, 'l3', routine, shapes('l3', routine), f)

wrong = sum(1 for r in runs if r['check'] == 'WRONG')
print('%d measurements, %d WRONG' % (len(runs), wrong))
