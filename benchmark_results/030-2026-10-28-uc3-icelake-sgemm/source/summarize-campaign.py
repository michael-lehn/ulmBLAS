#!/usr/bin/env python3
#
#  summarize-campaign.py RESDIR OUTDIR PREC REF1[,REF2...]: the archived
#  data of a GEMM campaign from the benchx results RESDIR/PREC-VARIANT-ROUND.txt
#  (columns m n k median min max calls err check, one line per size).
#
#  OUTDIR/runs.csv               all measurements (variant, round, n, ...)
#  OUTDIR/summary/gemm.csv       per size and variant: median, minimum and
#                                maximum over the rounds of the per-run
#                                medians (GFLOP/s)
#  OUTDIR/summary/ratios.csv     per size, ulmBLAS variant (ulmblas-*) and
#                                reference REF: the ratio computed within
#                                each round, median/min/max over the rounds
#  OUTDIR/summary/ranges.csv     per size range and variant: median, min,
#                                max over the sizes of the per-size medians,
#                                and of the ratios to each REF
#
#  Column names: the variant with '-' and '.' replaced by '_'.  A variant
#  measured in fewer rounds (e.g. once) is summarized over its rounds; its
#  ratios use the references of the same rounds.
#
import csv
import os
import re
import statistics
import sys

RANGES = [(50, 250), (300, 500), (550, 1000), (1050, 2000), (2050, 4000),
          (1000, 4000), (50, 200)]

res, out, prec, refs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(',')
runs = {}                                       # (variant, round) -> {n: row}
for f in sorted(os.listdir(res)):
    m = re.match(r'%s-(.+)-(\d+)\.txt$' % re.escape(prec), f)
    if not m:
        continue
    rows = {}
    for line in open(os.path.join(res, f)):
        w = line.split()
        if not w or w[0].startswith('#'):
            continue
        rows[int(w[1])] = w
    runs[(m.group(1), int(m.group(2)))] = rows

variants = sorted({v for v, _ in runs})
order = [v for v in variants if v.startswith('ulmblas-')] + \
        [v for v in variants if not v.startswith('ulmblas-')]
sizes = sorted({n for rows in runs.values() for n in rows})
col = lambda v: re.sub(r'[-.]', '_', v)
rounds = lambda v: sorted(r for w, r in runs if w == v)
gf = lambda v, r, n: float(runs[(v, r)][n][3])

os.makedirs(os.path.join(out, 'summary'), exist_ok=True)
with open(os.path.join(out, 'runs.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['variant', 'round', 'n', 'median', 'min', 'max', 'calls', 'err', 'check'])
    for v in order:
        for r in rounds(v):
            for n in sizes:
                x = runs[(v, r)].get(n)
                if x:
                    w.writerow([v, r, n] + x[3:9])


def stats(xs):
    return statistics.median(xs), min(xs), max(xs)


per = {}                                        # (variant, n) -> (med, min, max)
for v in order:
    for n in sizes:
        xs = [gf(v, r, n) for r in rounds(v) if n in runs[(v, r)]]
        if xs:
            per[(v, n)] = stats(xs)
ratio = {}                                      # (variant, ref, n) -> (med, min, max)
for v in order:
    if not v.startswith('ulmblas-'):
        continue
    for ref in refs:
        for n in sizes:
            xs = [gf(v, r, n) / gf(ref, r, n) for r in rounds(v)
                  if (ref, r) in runs and n in runs[(v, r)] and n in runs[(ref, r)]]
            if xs:
                ratio[(v, ref, n)] = stats(xs)

with open(os.path.join(out, 'summary', 'gemm.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['n'] + [c for v in order for c in (col(v), col(v) + '_min', col(v) + '_max')])
    for n in sizes:
        w.writerow([n] + [('%.3f' % x if (v, n) in per else '') for v in order
                          for x in per.get((v, n), (0, 0, 0))])
uv = [v for v in order if v.startswith('ulmblas-')]
with open(os.path.join(out, 'summary', 'ratios.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['n'] + [c for ref in refs for v in uv
                        for c in ('%s_over_%s' % (col(v), col(ref)),
                                  '%s_over_%s_min' % (col(v), col(ref)),
                                  '%s_over_%s_max' % (col(v), col(ref)))])
    for n in sizes:
        w.writerow([n] + ['%.4f' % x for ref in refs for v in uv
                          for x in ratio.get((v, ref, n), (float('nan'),) * 3)])
with open(os.path.join(out, 'summary', 'ranges.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['n_from', 'n_to', 'sizes', 'variant', 'gflops_median', 'gflops_min',
                'gflops_max'] + [c for ref in refs for c in
                                 ('over_%s_median' % col(ref), 'over_%s_min' % col(ref),
                                  'over_%s_max' % col(ref))])
    for lo, hi in RANGES:
        ns = [n for n in sizes if lo <= n <= hi]
        for v in order:
            g = [per[(v, n)][0] for n in ns if (v, n) in per]
            if not g:
                continue
            row = [lo, hi, len(g), v] + ['%.2f' % x for x in stats(g)]
            for ref in refs:
                if v == ref:
                    row += ['1', '1', '1']
                    continue
                if v.startswith('ulmblas-'):
                    q = [ratio[(v, ref, n)][0] for n in ns if (v, ref, n) in ratio]
                else:
                    q = [statistics.median(gf(v, r, n) / gf(ref, r, n) for r in rounds(v)
                                           if (ref, r) in runs and n in runs[(v, r)])
                         for n in ns if (v, n) in per]
                row += ['%.3f' % x for x in stats(q)] if q else ['', '', '']
            w.writerow(row)
bad = sum(1 for rows in runs.values() for x in rows.values() if x[8] != 'ok')
print('%d runs, %d measurements, %d not ok; variants: %s'
      % (len(runs), sum(len(r) for r in runs.values()), bad, ' '.join(order)))
