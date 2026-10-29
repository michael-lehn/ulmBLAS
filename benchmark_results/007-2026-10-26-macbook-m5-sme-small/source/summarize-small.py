#!/usr/bin/env python3
#
#  summarize-small.py RES OUT : results of compare.sh for the campaign
#  2026-10-26-macbook-m5-sme-small (RES/V-m_n_k-ROUND.txt, output of bench2).
#
#      OUT/runs.csv               every program run
#      OUT/summary/squares.csv    per n (m = n = k): for every variant V the
#                                 median over the rounds (V), its minimum and
#                                 maximum (V_min, V_max) in GFLOP/s and the
#                                 time per call in us (V_us)
#      OUT/summary/shapes.csv     the same for the other shapes (one line per
#                                 shape and variant)
#      OUT/summary/steps.csv      per shape: each step over the previous one
#                                 and final over published, with verdict;
#                                 final vs. Accelerate and BLIS PR #921
#
#  Variants: the ulmBLAS steps published, packtail, cache, direct, final and
#  the references accelerate, blis-sme.  Clear (faster/slower): the slowest
#  round of the faster one is faster than the fastest round of the other
#  and the medians differ by at least 1%; otherwise noise.
#
import csv
import glob
import os
import statistics
import sys

RES, OUT = sys.argv[1:3]
os.makedirs(os.path.join(OUT, 'summary'), exist_ok=True)
STEPS = ['published', 'packtail', 'cache', 'direct', 'final']
REFS = ['accelerate', 'blis-sme']
VARIANTS = STEPS + REFS

runs, g = [], {}
for path in sorted(glob.glob(os.path.join(RES, '*.txt'))):
    v, shape, rnd = os.path.basename(path)[:-4].rsplit('-', 2)
    if v == 'blis':
        continue
    m, n, k = (int(x) for x in shape.split('_'))
    for line in open(path):
        w = line.split()
        if len(w) >= 9 and w[0].isdigit():
            assert (int(w[0]), int(w[1]), int(w[2])) == (m, n, k), path
            runs.append([v, m, n, k, int(rnd), float(w[4]), float(w[5]),
                         float(w[6]), int(w[7]), w[8]])
            g.setdefault((m, n, k), {}).setdefault(v, []).append(float(w[4]))

key = lambda s: (s[0]*s[1]*s[2], s)
with open(os.path.join(OUT, 'runs.csv'), 'w', newline='') as f:
    c = csv.writer(f)
    c.writerow(['variant', 'm', 'n', 'k', 'round', 'gflops',
                'gflops_min_sample', 'gflops_max_sample', 'calls_per_sample',
                'check'])
    c.writerows(sorted(runs, key=lambda r: (key(tuple(r[1:4])), r[0], r[4])))

col = lambda v: v.replace('-', '_')
med = lambda x: statistics.median(x)
us = lambda s, gf: 2e-3*s[0]*s[1]*s[2]/gf

squares = sorted((s for s in g if s[0] == s[1] == s[2]), key=key)
others = sorted((s for s in g if not s[0] == s[1] == s[2]), key=key)

with open(os.path.join(OUT, 'summary', 'squares.csv'), 'w', newline='') as f:
    c = csv.writer(f)
    c.writerow(['n'] + [x for v in VARIANTS
                        for x in (col(v), col(v)+'_min', col(v)+'_max',
                                  col(v)+'_us')])
    for s in squares:
        row = [s[0]]
        for v in VARIANTS:
            x = g[s].get(v)
            row += ([round(med(x), 3), round(min(x), 3), round(max(x), 3),
                     round(us(s, med(x)), 4)] if x else ['NaN']*4)
        c.writerow(row)

with open(os.path.join(OUT, 'summary', 'shapes.csv'), 'w', newline='') as f:
    c = csv.writer(f)
    c.writerow(['m', 'n', 'k', 'variant', 'rounds', 'gflops', 'min', 'max',
                'us_per_call'])
    for s in others:
        for v in VARIANTS:
            x = g[s].get(v)
            if x:
                c.writerow([*s, v, len(x), round(med(x), 3), round(min(x), 3),
                            round(max(x), 3), round(us(s, med(x)), 4)])


def verdict(a, b):
    if min(a) > max(b) and med(a) >= 1.01*med(b):
        return 'faster'
    if min(b) > max(a) and med(b) >= 1.01*med(a):
        return 'slower'
    return 'noise'


def cmp(s, a, b):
    x, y = g[s].get(a), g[s].get(b)
    if not x or not y:
        return ['NaN', '']
    return [round(med(x)/med(y), 3), verdict(x, y)]


with open(os.path.join(OUT, 'summary', 'steps.csv'), 'w', newline='') as f:
    c = csv.writer(f)
    head = ['m', 'n', 'k']
    for a, b in zip(STEPS[1:], STEPS[:-1]):
        head += [a + '_over_' + b, a + '_vs_' + b]
    head += ['final_over_published', 'final_vs_published',
             'final_over_accelerate', 'final_vs_accelerate',
             'final_over_blis_sme', 'final_vs_blis_sme']
    c.writerow(head)
    for s in squares + others:
        row = list(s)
        for a, b in zip(STEPS[1:], STEPS[:-1]):
            row += cmp(s, a, b)
        row += cmp(s, 'final', 'published') + cmp(s, 'final', 'accelerate') \
             + cmp(s, 'final', 'blis-sme')
        c.writerow(row)

wrong = sum(1 for r in runs if r[9] == 'WRONG')
print('%d runs, %d shapes, %d WRONG' % (len(runs), len(g), wrong))
print('%-14s' % 'shape' + ''.join('%11s' % v[:10] for v in VARIANTS)
      + '   final/published  final/accelerate')
for s in squares + others:
    line = '%-14s' % ('%dx%dx%d' % s)
    for v in VARIANTS:
        x = g[s].get(v)
        line += '%11s' % ('%.3f' % us(s, med(x)) if x else '-')
    fp, fa = cmp(s, 'final', 'published'), cmp(s, 'final', 'accelerate')
    line += '   %6s %-6s  %6s %-6s' % (fp[0], fp[1], fa[0], fa[1])
    print(line)
