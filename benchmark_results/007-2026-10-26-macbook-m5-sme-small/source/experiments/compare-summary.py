#!/usr/bin/env python3
#
#  compare-summary.py RES OUT BASE [VARIANT...] : results of compare.sh
#  (RES/V-m_n_k-ROUND.txt, output of bench2).
#
#      OUT/runs.csv      every program run
#      OUT/summary.csv   per shape and variant: median, min, max over the
#                        rounds, spread, us per call, ratio to BASE and
#                        verdict (faster, slower, noise)
#
#  Clear (faster/slower): the slowest round of the faster variant is faster
#  than the fastest round of the other one and the medians differ by at
#  least 1%; otherwise noise.  VARIANT...: order of the columns of the
#  printed table (default: all, sorted).
#
import csv
import glob
import os
import statistics
import sys

RES, OUT, BASE = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
runs, g = [], {}
for path in sorted(glob.glob(os.path.join(RES, '*.txt'))):
    name = os.path.basename(path)[:-4]
    v, shape, rnd = name.rsplit('-', 2)
    m, n, k = (int(x) for x in shape.split('_'))
    for line in open(path):
        w = line.split()
        if len(w) >= 9 and w[0].isdigit():
            assert (int(w[0]), int(w[1]), int(w[2])) == (m, n, k), path
            runs.append([v, m, n, k, int(rnd), float(w[4]), float(w[5]),
                         float(w[6]), int(w[7]), w[8]])
            g.setdefault((m, n, k), {}).setdefault(v, []).append(float(w[4]))

variants = sys.argv[4:] or sorted({r[0] for r in runs})
with open(os.path.join(OUT, 'runs.csv'), 'w', newline='') as f:
    c = csv.writer(f)
    c.writerow(['variant', 'm', 'n', 'k', 'round', 'gflops',
                'gflops_min_sample', 'gflops_max_sample', 'calls_per_sample',
                'check'])
    c.writerows(sorted(runs, key=lambda r: (r[1]*r[2]*r[3], r[1:4], r[0], r[4])))


def verdict(a, b):
    ma, mb = statistics.median(a), statistics.median(b)
    if min(a) > max(b) and ma >= 1.01*mb:
        return 'faster'
    if min(b) > max(a) and mb >= 1.01*ma:
        return 'slower'
    return 'noise'


shapes = sorted(g, key=lambda s: (s[0]*s[1]*s[2], s))
with open(os.path.join(OUT, 'summary.csv'), 'w', newline='') as f:
    c = csv.writer(f)
    c.writerow(['m', 'n', 'k', 'variant', 'rounds', 'gflops', 'min', 'max',
                'spread_percent', 'us_per_call', 'over_' + BASE,
                'vs_' + BASE])
    for s in shapes:
        for v in variants:
            x = g[s].get(v)
            if not x:
                continue
            md = statistics.median(x)
            b = g[s].get(BASE)
            c.writerow([*s, v, len(x), round(md, 2), round(min(x), 2),
                        round(max(x), 2), round(100*(max(x)-min(x))/md, 1),
                        round(2e-3*s[0]*s[1]*s[2]/md, 4),
                        round(md/statistics.median(b), 3) if b else '',
                        verdict(x, b) if b and v != BASE else ''])

wrong = sum(1 for r in runs if r[9] == 'WRONG')
print('%d runs, %d shapes, %d variants, %d WRONG; us per call (median), '
      '[ratio to %s, verdict]' % (len(runs), len(g), len(variants), wrong,
                                   BASE))
print('%-14s' % 'm x n x k' + ''.join('%22s' % v[:22] for v in variants))
for s in shapes:
    line = '%-14s' % ('%dx%dx%d' % s)
    for v in variants:
        x = g[s].get(v)
        if not x:
            line += '%22s' % '-'
            continue
        md = statistics.median(x)
        us = 2e-3*s[0]*s[1]*s[2]/md
        b = g[s].get(BASE)
        tag = ''
        if b and v != BASE:
            r = md/statistics.median(b)
            tag = ' %.2f%s' % (r, {'faster': '+', 'slower': '-',
                                   'noise': '~'}[verdict(x, b)])
        line += '%22s' % ('%.3f%s' % (us, tag))
    print(line)
