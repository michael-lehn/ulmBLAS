#!/usr/bin/env python3
#
#  summarize-bench.py DIR : medians over the rounds of the outputs of
#  bench2 and bench3 (DIR/rROUND/GROUP-VARIANT.txt).  One CSV line per
#  routine, size and variant:
#
#      routine,m,n,k,variant,rounds,gflops,gflops_min,gflops_max,wrong
#
#  routine is dgemm (bench2) or the routine of bench3; gflops is the median
#  over the rounds of the medians of the samples, gflops_min and gflops_max
#  the extremes over the rounds; wrong counts results marked WRONG.
#
import glob
import os
import statistics
import sys

d = {}
for f in glob.glob(os.path.join(sys.argv[1], 'r*', '*.txt')):
    name = os.path.basename(f)[:-4]
    if name.startswith('prof-'):
        continue
    variant = name.split('-', 1)[1]
    for line in open(f):
        w = line.split()
        if len(w) >= 9 and w[0].isdigit():          # bench2
            key = ('dgemm', int(w[0]), int(w[1]), int(w[2]), variant)
            d.setdefault(key, []).append((float(w[4]), 'WRONG' in line))
        elif len(w) >= 5 and w[0].startswith('d') and w[1].isdigit():
            n = int(w[1])
            key = (w[0], n, n, n, variant)
            d.setdefault(key, []).append((float(w[2]), False))
print('routine,m,n,k,variant,rounds,gflops,gflops_min,gflops_max,wrong')
for key in sorted(d, key=lambda k: (k[0], k[1]*k[2]*k[3], k[1:])):
    g = [x[0] for x in d[key]]
    print('%s,%d,%d,%d,%s,%d,%.3f,%.3f,%.3f,%d' % (key + (len(g),
          statistics.median(g), min(g), max(g), sum(x[1] for x in d[key]))))
