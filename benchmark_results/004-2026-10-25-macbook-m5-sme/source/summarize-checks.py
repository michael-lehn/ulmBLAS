#!/usr/bin/env python3
#
#  summarize-checks.py RAW OUT : the two later checks of bench2 into
#  OUT/summary/ab_v5_final.csv (RAW/ab-v5-final, ab-blocks.sh: 889aa15 with
#  MC, KC, NC = 608, 2240, 3072 against 8fe961a with 384, 1024, 1024) and
#  OUT/summary/session_check.csv (RAW/session-check, ab-runlength.sh: the
#  program bench2-sme of the campaign again, 4096 x 4096 x 4096 alone or
#  after other sizes in the same program run): GFLOP/s of each round and
#  the median.
#
import csv
import glob
import os
import statistics
import sys

RAW, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, 'summary'), exist_ok=True)
for sub, name in (('ab-v5-final', 'ab_v5_final.csv'),
                  ('session-check', 'session_check.csv')):
    g = {}
    for path in sorted(glob.glob(os.path.join(RAW, sub, '*.txt'))):
        v, rnd = os.path.basename(path)[:-4].rsplit('-', 1)
        for line in open(path):
            w = line.split()
            if len(w) >= 9 and w[0].isdigit():
                g.setdefault(('%sx%sx%s' % tuple(w[:3]), v), []).append(
                    (int(rnd), float(w[4])))
    with open(os.path.join(OUT, 'summary', name), 'w', newline='') as f:
        c = csv.writer(f)
        c.writerow(['shape', 'variant', 'rounds', 'gflops', 'median'])
        for (shape, v), x in sorted(g.items()):
            x.sort()
            c.writerow([shape, v, len(x), ' '.join('%.1f' % y for _, y in x),
                        round(statistics.median(y for _, y in x), 1)])
