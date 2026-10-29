#!/usr/bin/env python3
#
#  prof2csv.py: converts the outputs of prof/prof-ob (data/raw/*/PROGRAM-N-ROUND.txt)
#  into data/runs.csv (one line per output line) and data/summary/breakdown.csv
#  (median over the runs per program, mode and n; GFLOP/s and the fractions
#  of the time of a call in packing B, packing A and the macro kernel).
#  Run in the campaign directory.
#
import csv, glob, os, re, statistics
runs = []
for path in sorted(glob.glob('data/raw/*/*.txt')):
    job = path.split('/')[2]
    m = re.match(r'(.+)-(\d+)-(\d+)\.txt$', os.path.basename(path))
    if not m:
        continue
    prog, n, rnd = m.group(1), int(m.group(2)), int(m.group(3))
    for line in open(path):
        f = line.split()
        if len(f) < 15 or f[0] in ('check', 'mcheck'):
            continue
        runs.append(dict(job=job, program=prog, mode=f[0], MC=int(f[2]),
                         KC=int(f[3]), NC=int(f[4]), n=n, round=rnd,
                         gflops=float(f[8]), packB=float(f[10]),
                         packA=float(f[11]), mgemm=float(f[12])))
os.makedirs('data/summary', exist_ok=True)
keys = ['job', 'program', 'mode', 'MC', 'KC', 'NC', 'n', 'round', 'gflops',
        'packB', 'packA', 'mgemm']
with open('data/runs.csv', 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=keys); w.writeheader(); w.writerows(runs)
groups = {}
for r in runs:
    groups.setdefault((r['job'], r['program'], r['mode'], r['n']), []).append(r)
with open('data/summary/breakdown.csv', 'w', newline='') as out:
    w = csv.writer(out)
    w.writerow(['job', 'program', 'mode', 'n', 'gflops', 'gflops_min',
                'gflops_max', 'runs', 'packB', 'packA', 'mgemm'])
    for (job, prog, mode, n), rs in sorted(groups.items()):
        g = [r['gflops'] for r in rs]
        w.writerow([job, prog, mode, n, '%.2f' % statistics.median(g),
                    '%.2f' % min(g), '%.2f' % max(g), len(g),
                    '%.4f' % statistics.median(r['packB'] for r in rs),
                    '%.4f' % statistics.median(r['packA'] for r in rs),
                    '%.4f' % statistics.median(r['mgemm'] for r in rs)])
#  table for the plot: per n of the second job (sizes where the kernel can be
#  called per block) the GFLOP/s of the modes of the OpenBLAS kernel and of
#  native OpenBLAS
with open('data/summary/stages.csv', 'w', newline='') as out:
    w = csv.writer(out)
    modes = ['lib', 'macro', 'nopack', 'macronopack', 'kern']
    w.writerow(['n'] + modes + ['native_openblas'])
    for n in sorted({k[3] for k in groups if k[0] == 'profob-8009801'}):
        row = [n]
        for mo in modes:
            rs = groups.get(('profob-8009801', 'ob', mo, n), [])
            row.append('%.2f' % statistics.median(r['gflops'] for r in rs) if rs else 'NaN')
        rs = groups.get(('profob-8009801', 'cblas', 'cblas', n), [])
        row.append('%.2f' % statistics.median(r['gflops'] for r in rs) if rs else 'NaN')
        w.writerow(row)
print('%d runs' % len(runs))
