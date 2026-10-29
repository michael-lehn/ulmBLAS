#!/usr/bin/env python3
#
#  summarize-packtail.py RES OUT : the benchmark of packtail-bench.sh
#  (RES/V-m_n_k-ROUND.txt, output of bench2; V = before, after).
#
#      OUT/runs.csv      every program run: variant, m, n, k, round, median,
#                        min, max of the samples, calls per sample, check
#      OUT/summary.csv   per shape: median, min, max over the rounds and
#                        spread (max - min) / median of both variants, time
#                        per call, after / before, verdict
#
#  Clear: the slowest round of the faster variant is faster than the fastest
#  round of the other one and the medians differ by at least 1%; otherwise
#  noise.
#
import csv
import glob
import os
import statistics
import sys

RES, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
runs, g = [], {}
for path in sorted(glob.glob(os.path.join(RES, '*.txt'))):
    v, shape, rnd = os.path.basename(path)[:-4].split('-')
    m, n, k = (int(x) for x in shape.split('_'))
    for line in open(path):
        w = line.split()
        if len(w) >= 9 and w[0].isdigit():
            assert (int(w[0]), int(w[1]), int(w[2])) == (m, n, k), path
            runs.append([v, m, n, k, int(rnd), float(w[4]), float(w[5]),
                         float(w[6]), int(w[7]), w[8]])
            g.setdefault((m, n, k), {}).setdefault(v, []).append(float(w[4]))

with open(os.path.join(OUT, 'runs.csv'), 'w', newline='') as f:
    c = csv.writer(f)
    c.writerow(['variant', 'm', 'n', 'k', 'round', 'gflops',
                'gflops_min_sample', 'gflops_max_sample', 'calls_per_sample',
                'check'])
    c.writerows(sorted(runs, key=lambda r: (r[1]*r[2]*r[3], r[1:4], r[0], r[4])))


def verdict(a, b):
    if min(a) > max(b) and statistics.median(a) >= 1.01*statistics.median(b):
        return 'faster'
    if min(b) > max(a) and statistics.median(b) >= 1.01*statistics.median(a):
        return 'slower'
    return 'noise'


rows = []
for (m, n, k), d in sorted(g.items(), key=lambda x: (x[0][0]*x[0][1]*x[0][2], x[0])):
    b, a = d['before'], d['after']
    mb, ma = statistics.median(b), statistics.median(a)
    rows.append([m, n, k, len(b), len(a),
                 round(mb, 2), round(min(b), 2), round(max(b), 2),
                 round(100*(max(b)-min(b))/mb, 1),
                 round(ma, 2), round(min(a), 2), round(max(a), 2),
                 round(100*(max(a)-min(a))/ma, 1),
                 round(2e-3*m*n*k/mb, 3), round(2e-3*m*n*k/ma, 3),
                 round(ma/mb, 3), verdict(a, b)])
with open(os.path.join(OUT, 'summary.csv'), 'w', newline='') as f:
    c = csv.writer(f)
    c.writerow(['m', 'n', 'k', 'rounds_before', 'rounds_after',
                'before_gflops', 'before_min', 'before_max',
                'before_spread_percent', 'after_gflops', 'after_min',
                'after_max', 'after_spread_percent', 'before_us_per_call',
                'after_us_per_call', 'after_over_before', 'after_vs_before'])
    c.writerows(rows)
wrong = sum(1 for r in runs if r[9] == 'WRONG')
print('%d runs, %d shapes, %d WRONG' % (len(runs), len(g), wrong))
print('%-16s %8s %8s %6s   %8s %8s %6s   %6s  %s' % (
    'm x n x k', 'before', 'us/call', 'spr%', 'after', 'us/call', 'spr%',
    'ratio', 'verdict'))
for r in rows:
    print('%-16s %8.2f %8.3f %6.1f   %8.2f %8.3f %6.1f   %6.3f  %s' % (
        '%dx%dx%d' % tuple(r[:3]), r[5], r[13], r[8], r[9], r[14], r[12],
        r[15], r[16]))
