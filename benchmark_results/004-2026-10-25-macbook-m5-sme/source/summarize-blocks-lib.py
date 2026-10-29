#!/usr/bin/env python3
#
#  summarize-blocks-lib.py RES OUT : outputs of ab-blocks2.sh (RES/bMC_KC_NC-
#  ROUND.txt, bench2) into OUT/blocksizes_lib.csv (one line per run) and
#  OUT/summary/blocksizes_lib.csv (medians over the rounds and the ratio to
#  the block sizes of BLIS, MC, KC, NC = 608, 2240, 3072).
#
import csv
import glob
import os
import statistics
import sys

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, 'summary'), exist_ok=True)
rows, g = [], {}
for path in sorted(glob.glob(os.path.join(RES, 'b*.txt'))):
    v, rnd = os.path.basename(path)[1:-4].rsplit('-', 1)
    for line in open(path):
        w = line.split()
        if len(w) >= 9 and w[0].isdigit():
            shape = '%sx%sx%s' % tuple(w[:3])
            rows.append([v, shape, int(rnd), float(w[4])])
            g.setdefault(shape, {}).setdefault(v, []).append(float(w[4]))
with open(os.path.join(OUT, 'blocksizes_lib.csv'), 'w', newline='') as f:
    c = csv.writer(f)
    c.writerow(['mc_kc_nc', 'shape', 'round', 'gflops'])
    c.writerows(rows)
V = sorted({r[0] for r in rows})
BASE = '608_2240_3072'
order = sorted(g, key=lambda s: -eval(s.replace('x', '*')))
with open(os.path.join(OUT, 'summary', 'blocksizes_lib.csv'), 'w',
          newline='') as f:
    c = csv.writer(f)
    c.writerow(['shape'] + ['b' + v for v in V] + ['b' + v + '_rel' for v in V])
    for s in order:
        med = {v: statistics.median(g[s][v]) for v in V}
        c.writerow([s] + [round(med[v], 1) for v in V]
                   + [round(med[v] / med[BASE], 3) for v in V])
