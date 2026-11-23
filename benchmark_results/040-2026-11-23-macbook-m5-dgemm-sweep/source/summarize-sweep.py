#!/usr/bin/env python3
#
#  summarize-sweep.py RES OUT : results of compare.sh for the sweep of
#  campaign 040 (RES/V-n_n_n-ROUND.txt, output of bench2: m n k beta
#  median min max ...).
#
#      OUT/runs.csv              every program run (variant, n, round,
#                                median, min, max in GFLOP/s)
#      OUT/summary/squares.csv   per n: for every variant V the median over
#                                the rounds (V), its minimum and maximum
#                                (V_min, V_max) in GFLOP/s and the time per
#                                call in us (V_us); ratios of ulmBLAS
#
#  Variants: ulmblas (0.4.2), accelerate, blis-sme.  Clear (faster/slower):
#  the slowest round of the faster one is faster than the fastest round of
#  the other and the medians differ by at least 1%; otherwise noise.
import os, re, sys, statistics as st, collections as C
res, out = sys.argv[1], sys.argv[2]
V = ['ulmblas', 'accelerate', 'blis-sme']
R = C.defaultdict(list); runs = []
for f in sorted(os.listdir(res)):
    m = re.match(r'(.+)-(\d+)_\d+_\d+-(\d+)\.txt$', f)
    if not m: continue
    v, n, r = m.group(1), int(m.group(2)), int(m.group(3))
    p = open(os.path.join(res, f)).read().split()
    med, mn, mx = float(p[4]), float(p[5]), float(p[6])
    R[(v, n)].append(med); runs.append((v, n, r, med, mn, mx))
os.makedirs(os.path.join(out, 'summary'), exist_ok=True)
with open(os.path.join(out, 'runs.csv'), 'w') as f:
    f.write('variant,n,round,median,min,max\n')
    for x in sorted(runs): f.write('%s,%d,%d,%g,%g,%g\n' % x)
def verdict(a, b):
    ma, mb = st.median(a), st.median(b)
    if min(a) > max(b) and ma >= 1.01 * mb: return 'faster'
    if max(a) < min(b) and ma <= 0.99 * mb: return 'slower'
    return 'noise'
ns = sorted({n for _, n in R})
with open(os.path.join(out, 'summary', 'squares.csv'), 'w') as f:
    head = ['n']
    for v in V:
        k = v.replace('-', '_'); head += [k, k + '_min', k + '_max', k + '_us']
    head += ['ulmblas_over_accelerate', 'vs_accelerate', 'ulmblas_over_blis_sme', 'vs_blis_sme']
    f.write(','.join(head) + '\n')
    for n in ns:
        row = [str(n)]
        for v in V:
            x = R[(v, n)]; m = st.median(x)
            row += ['%.3f' % m, '%.3f' % min(x), '%.3f' % max(x), '%.4f' % (2e-3 * n**3 / m / 1e3)]
        for o in ('accelerate', 'blis-sme'):
            row += ['%.3f' % (st.median(R[('ulmblas', n)]) / st.median(R[(o, n)])),
                    verdict(R[('ulmblas', n)], R[(o, n)])]
        f.write(','.join(row) + '\n')
