#!/usr/bin/env python3
#
#  summarize.py: converts the raw outputs data/raw/E/V-m_n_k-R.txt of the
#  experiments E (not used: see EXCLUDED) into
#
#    data/runs.csv            one line per program run (experiment,
#                             configuration, shape, round, median, min and
#                             max GFLOP/s of the samples, time per call)
#    data/summary/E.csv       per experiment, configuration and shape: median,
#                             minimum and maximum over the rounds of the
#                             per-run medians, spread, time per call, ratio
#                             to the production configuration (MC, KC, NC =
#                             384, 1024, 1024) and verdict, block
#                             decomposition (decomp.py)
#
#    data/summary/rect-used.csv
#                             rect with the shapes of rect2 replaced by
#                             rect2 (the rows used in README.md and plots)
#    data/summary/best.csv    per square shape of the MC/NC grids (coarse,
#                             fine): production and the configuration with
#                             the highest median (selected after the fact
#                             from 19 or 7 configurations, i.e. biased
#                             upwards by the noise; see README.md)
#
#  Verdict: "faster"/"slower" if the ranges over the rounds do not overlap
#  (the minimum of one exceeds the maximum of the other) and the medians
#  differ by at least 1%, else "noise".  Run in the campaign directory.
#
import csv, glob, os, re, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decomp import decomp

PROD = 'b384_1024_1024'
#  BLIS PR #921 (configuration m4sme_p): its block sizes MC, KC, NC (the
#  remainders are not merged in this configuration, see README.md)
BLIS = ('blis-sme', 608, 2240, 3072)

def blocksizes(v):
    if v == BLIS[0]:
        return BLIS[1:]
    return tuple(int(y) for y in v[1:].split('_'))

#  Further rounds of an experiment E were measured in the directory E-r4
#  (rounds 4, ...).  Not used (they stay in runs.csv with used = 0, see
#  README.md): round 1 of rect2 (it ran from 23:42 to 06:12 while other
#  jobs on the machine started and stopped) and rounds 4 to 7 of blis
#  (measured under intermittent load: median spread of the samples within
#  a run 4.2-5.2%, in every other round of the campaign 0.7-3.1%).
#  The stopped first attempt of blis (directory blis-aborted) is not used
#  at all.
EXCLUDED = {('rect2', 1)} | {('blis', r) for r in range(4, 8)}

def used(E, r):
    return E != 'blis-aborted' and (E, r) not in EXCLUDED

runs = []
for path in sorted(glob.glob('data/raw/*/*.txt')):
    E = os.path.basename(os.path.dirname(path))
    if E.endswith('-r4'):
        E = E[:-3]
    mt = re.match(r'(b\d+_\d+_\d+|blis-sme)-(\d+)_(\d+)_(\d+)-(\d+)\.txt$',
                  os.path.basename(path))
    if not mt:
        continue
    v = mt.group(1)
    MC, KC, NC = blocksizes(v)
    m, n, k, r = (int(mt.group(i)) for i in range(2, 6))
    f = open(path).read().split()
    if len(f) < 9 or [int(x) for x in f[:3]] != [m, n, k]:
        sys.exit('unexpected output in %s' % path)
    med, lo, hi = float(f[4]), float(f[5]), float(f[6])
    runs.append(dict(experiment=E, config=v, MC=MC, KC=KC, NC=NC, m=m, n=n,
                     k=k, round=r, gflops=med, gflops_min=lo, gflops_max=hi,
                     us_per_call='%.2f' % (2.0*m*n*k/med*1e-3),
                     calls_per_sample=int(f[7]), check=f[8],
                     used=int(used(E, r))))

os.makedirs('data/summary', exist_ok=True)
with open('data/runs.csv', 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=list(runs[0].keys()))
    w.writeheader()
    w.writerows(runs)

def verdict(a, b):
    ratio = a[0]/b[0]
    if a[1] > b[2] and ratio >= 1.01:
        return ratio, 'faster'
    if a[2] < b[1] and ratio <= 1/1.01:
        return ratio, 'slower'
    return ratio, 'noise'

def stats(x):
    return (statistics.median(x), min(x), max(x))

def bestrows(E, st):
    out = []
    for (m, n, k) in sorted({key[1:] for key in st}):
        if not m == n == k or (PROD, m, n, k) not in st:
            continue
        cand = [(stats(x), v) for (v, *s), x in st.items() if tuple(s) == (m, n, k)]
        b, v = max(cand)
        p = stats(st[(PROD, m, n, k)])
        ratio, vd = verdict(b, p)
        out.append(dict(experiment=E, n=n, configurations=len(cand),
                        production='%.2f' % p[0], production_min='%.2f' % p[1],
                        production_max='%.2f' % p[2], best_config=v,
                        best='%.2f' % b[0], best_min='%.2f' % b[1],
                        best_max='%.2f' % b[2], ratio='%.4f' % ratio,
                        verdict=vd))
    return out

allrows = {}
best = []
for E in sorted({r['experiment'] for r in runs}):
    R = [r for r in runs if r['experiment'] == E]
    if not any(r['used'] for r in R):
        continue
    st = {}
    for r in R:
        if not r['used']:
            continue
        st.setdefault((r['config'], r['m'], r['n'], r['k']), []).append(r['gflops'])
    rows = []
    for (v, m, n, k), x in sorted(st.items(), key=lambda t: (t[0][1:], blocksizes(t[0][0]), t[0][0])):
        s = (statistics.median(x), min(x), max(x))
        MC, KC, NC = blocksizes(v)
        row = dict(config=v, MC=MC, KC=KC, NC=NC, m=m, n=n, k=k,
                   gflops='%.2f' % s[0], gflops_min='%.2f' % s[1],
                   gflops_max='%.2f' % s[2], rounds=len(x),
                   spread_pct='%.2f' % (100*(s[2]-s[1])/s[0]),
                   us_per_call='%.1f' % (2.0*m*n*k/s[0]*1e-3))
        p = st.get((PROD, m, n, k))
        if p:
            ps = (statistics.median(p), min(p), max(p))
            ratio, vd = verdict(s, ps)
            row.update(vs_production='%.4f' % ratio, verdict=vd)
        else:
            row.update(vs_production='NaN', verdict='')
        row.update(decomp(m, n, k, MC, KC, NC))
        rows.append(row)
    allrows[E] = rows
    with open('data/summary/%s.csv' % E, 'w', newline='') as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if E in ('coarse', 'fine'):
        best.extend(bestrows(E, st))
    print('%s: %d runs (%d used), %d configurations, %d shapes, %d WRONG' % (
        E, len(R), sum(r['used'] for r in R), len({r['config'] for r in R}),
        len({(r['m'], r['n'], r['k']) for r in R}),
        sum(r['check'] == 'WRONG' for r in R)))

with open('data/summary/best.csv', 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=list(best[0].keys()))
    w.writeheader()
    w.writerows(best)

if 'rect' in allrows and 'rect2' in allrows:
    key = lambda r: (r['m'], r['n'], r['k'])
    redo = {key(r) for r in allrows['rect2']}
    used = [dict(r, experiment='rect') for r in allrows['rect'] if key(r) not in redo]
    used += [dict(r, experiment='rect2') for r in allrows['rect2']]
    used.sort(key=lambda r: (key(r), blocksizes(r['config'])))
    with open('data/summary/rect-used.csv', 'w', newline='') as out:
        w = csv.DictWriter(out, fieldnames=list(used[0].keys()))
        w.writeheader()
        w.writerows(used)
