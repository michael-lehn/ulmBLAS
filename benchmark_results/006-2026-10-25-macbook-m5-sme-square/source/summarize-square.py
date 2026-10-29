#!/usr/bin/env python3
#
#  summarize-square.py RES OUT PLOTS : the square DGEMM benchmark of
#  square-interleaved.sh (RES/V-n-ROUND.txt, output of bench2).
#
#      OUT/square.csv             every program run: variant, n, round,
#                                 median, min and max of the samples, calls
#                                 per sample, check
#      OUT/summary/square.csv     per n: median over the rounds of every
#                                 variant, and its minimum and maximum over
#                                 the rounds (columns V, V_min, V_max)
#      OUT/summary/spread.csv     per variant and n: rounds, median, min,
#                                 max, spread (max - min) / median
#      OUT/summary/best.csv       per n: the best fixed SME2 configuration,
#                                 the runner-up, the gap, whether the gap
#                                 is clear or within the noise, the tied
#                                 configurations, the comparison with the
#                                 default 384, 1024, 1024, with BLIS PR #921
#                                 and with Accelerate
#      PLOTS/_selection.gp        the SME2 configurations of the readable
#                                 plot (gnuplot string SEL)
#
#  Clear: the slowest round of the better one is faster than the fastest
#  round of the other one, and the medians differ by at least 1%; otherwise
#  noise.  Every curve is one fixed configuration; nothing is combined.
#
import csv
import re
import glob
import os
import statistics
import sys

RES, OUT, PLOTS = sys.argv[1:4]
os.makedirs(os.path.join(OUT, 'summary'), exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

runs, g = [], {}
for path in sorted(glob.glob(os.path.join(RES, '*.txt'))):
    v, n, rnd = os.path.basename(path)[:-4].rsplit('-', 2)
    for line in open(path):
        w = line.split()
        if len(w) >= 9 and w[0].isdigit():
            assert int(w[0]) == int(w[1]) == int(w[2]) == int(n), path
            runs.append([v, int(n), int(rnd), float(w[4]), float(w[5]),
                         float(w[6]), int(w[7]), w[8]])
            g.setdefault((v, int(n)), []).append(float(w[4]))


def write(name, header, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        c = csv.writer(f)
        c.writerow(header)
        c.writerows(rows)


def col(v):
    return v.replace('-', '_')


def is_config(v):
    return re.fullmatch(r'b\d+_\d+_\d+', v) is not None


def label(v):
    if is_config(v):
        mc, kc, nc = v[1:].split('_')
        return 'MC=%s KC=%s NC=%s' % (mc, kc, nc)
    return v


write('square.csv', ['variant', 'n', 'round', 'gflops', 'gflops_min_sample',
                     'gflops_max_sample', 'calls_per_sample', 'check'], runs)
REFS = [v for v in ('accelerate', 'blis-sme', 'neon') if any(k[0] == v for k in g)]
CONFIGS = sorted({v for v, _ in g if is_config(v)})
V = REFS + CONFIGS
N = sorted({n for _, n in g})
med = {k: statistics.median(x) for k, x in g.items()}

rows = []
for n in N:
    r = [n]
    for v in V:
        x = g.get((v, n))
        r += ([round(med[(v, n)], 1), round(min(x), 1), round(max(x), 1)]
              if x else ['NaN'] * 3)
    rows.append(r)
write('summary/square.csv',
      ['n'] + [c for v in V for c in (col(v), col(v) + '_min', col(v) + '_max')],
      rows)
write('summary/spread.csv', ['variant', 'n', 'rounds', 'gflops', 'min', 'max',
                             'spread_percent'],
      [[v, n, len(g[(v, n)]), round(med[(v, n)], 1), round(min(g[(v, n)]), 1),
        round(max(g[(v, n)]), 1),
        round(100 * (max(g[(v, n)]) - min(g[(v, n)])) / med[(v, n)], 1)]
       for v in V for n in N if (v, n) in g])


def verdict(a, b, n):
    """'clear' if a is clearly faster than b at size n, otherwise 'noise'
    (or 'slower' if b is clearly faster than a)."""
    if min(g[(a, n)]) > max(g[(b, n)]) and med[(a, n)] >= 1.01 * med[(b, n)]:
        return 'clear'
    if min(g[(b, n)]) > max(g[(a, n)]) and med[(b, n)] >= 1.01 * med[(a, n)]:
        return 'slower'
    return 'noise'


best, wins = [], {}
D = 'b384_1024_1024'
for n in N:
    ranked = sorted(CONFIGS, key=lambda v: -med[(v, n)])
    b, r = ranked[0], ranked[1]
    wins[b] = wins.get(b, 0) + 1
    tied = [v for v in ranked[1:] if verdict(b, v, n) == 'noise']
    row = [n, label(b), round(med[(b, n)], 1), label(r), round(med[(r, n)], 1),
           round(100 * (med[(b, n)] / med[(r, n)] - 1), 1), verdict(b, r, n),
           ' '.join(label(v) for v in tied),
           round(med[(D, n)], 1),
           round(100 * (med[(b, n)] / med[(D, n)] - 1), 1),
           'default' if b == D else verdict(b, D, n)]
    for ref in ('blis-sme', 'accelerate'):
        if (ref, n) in g:
            row += [round(med[(ref, n)], 1),
                    round(med[(b, n)] / med[(ref, n)], 3), verdict(b, ref, n),
                    round(med[(D, n)] / med[(ref, n)], 3), verdict(D, ref, n)]
        else:
            row += ['NaN'] * 5
    best.append(row)
write('summary/best.csv',
      ['n', 'best', 'best_gflops', 'runner_up', 'runner_up_gflops',
       'gap_percent', 'verdict', 'tied_with_best', 'default_gflops',
       'best_over_default_percent', 'best_vs_default']
      + [c for ref in ('blis_sme', 'accelerate')
         for c in (ref + '_gflops', 'best_over_' + ref, 'best_vs_' + ref,
                   'default_over_' + ref, 'default_vs_' + ref)], best)

#  Readable plot: the configurations that are the best (highest median)
#  for at least 2 sizes, the default and the block sizes of BLIS PR #921.
sel = sorted({v for v, k in wins.items() if k >= 2} | {D, 'b608_2240_3072'})
with open(os.path.join(PLOTS, '_selection.gp'), 'w') as f:
    f.write('#  SME2 configurations of the readable plot (written by\n'
            '#  summarize-square.py: the best for at least 2 sizes, the\n'
            '#  default 384, 1024, 1024 and the block sizes of BLIS PR #921\n'
            '#  608, 2240, 3072)\n')
    f.write('SEL = "%s"\n' % ' '.join(sel))
wrong = sum(1 for r in runs if r[7] == 'WRONG')
print('%d runs, %d variants, %d sizes, %d WRONG; best for (sizes): %s'
      % (len(runs), len(V), len(N), wrong,
         ', '.join('%s (%d)' % (label(v), k) for v, k in sorted(wins.items()))))
