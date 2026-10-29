#!/usr/bin/env python3
#
#  summarize.py: converts the raw outputs of the jobs data/raw/run-JOB/
#  into
#
#    data/sweep.csv          bench2 per variant, n and round (sweep-V-R.txt)
#    data/summary/sweep.csv  per n: medians over the rounds (min, max) of
#                            ulmBLAS and native OpenBLAS, ratio, time per
#                            call, n mod 16 and n mod 12 (tile MR x NR =
#                            16 x 12), OpenBLAS's path (small kernel for
#                            n^3 <= 1e6)
#    data/summary/prof.csv   prof-ob per mode and n: medians over the rounds
#                            of GFLOP/s, time per call and the fractions of
#                            the time in packing B, packing A, macro kernel
#                            and the rest (mode inst)
#
#  Run in the campaign directory.
#
import csv, glob, os, re, statistics

rows = []
for path in sorted(glob.glob('data/raw/run-*/sweep-*-*.txt')):
    job = os.path.basename(os.path.dirname(path))[4:]
    v, r = re.match(r'sweep-(.+)-(\d+)\.txt$', os.path.basename(path)).groups()
    for line in open(path):
        f = line.split()
        rows.append(dict(job=job, variant=v, round=int(r), n=int(f[0]),
                         gflops=float(f[4]), gflops_min=float(f[5]),
                         gflops_max=float(f[6]), calls=int(f[7]), check=f[8]))
os.makedirs('data/summary', exist_ok=True)
with open('data/sweep.csv', 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

st = {}
for r in rows:
    if r['job'] == '8026489':          # the sweep job (8026592: n = 8, 16 only)
        st.setdefault((r['variant'], r['n']), []).append(r['gflops'])
out = []
for n in sorted({n for v, n in st}):
    u, o = st[('ulmblas', n)], st[('openblas', n)]
    mu, mo = statistics.median(u), statistics.median(o)
    out.append(dict(n=n, ulmblas='%.2f' % mu, ulmblas_min='%.2f' % min(u),
                    ulmblas_max='%.2f' % max(u), openblas='%.2f' % mo,
                    openblas_min='%.2f' % min(o), openblas_max='%.2f' % max(o),
                    ratio='%.4f' % (mu/mo),
                    us_ulmblas='%.3f' % (2.0*n**3/mu*1e-3),
                    us_openblas='%.3f' % (2.0*n**3/mo*1e-3),
                    n_mod_16=n % 16, n_mod_12=n % 12,
                    full_tiles=int(n % 16 == 0 and n % 12 == 0),
                    openblas_path='small' if n**3 <= 1e6 else 'driver'))
with open('data/summary/sweep.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

prof = {}
for path in sorted(glob.glob('data/raw/run-*/*-*-*.txt')):
    if not re.match(r'(prof|cblas)-\d+-\d+\.txt$', os.path.basename(path)):
        continue
    for line in open(path):
        f = line.split()
        if len(f) < 14:
            continue
        prof.setdefault((f[0], int(f[5])), []).append(
            [float(x) for x in f[8:14]])
order = ['lib', 'inst', 'nopack', 'pack', 'kern', 'pool', 'cblas']
pr = []
for (mode, n) in sorted(prof, key=lambda t: (t[1], order.index(t[0]))):
    x = prof[(mode, n)]
    med = lambda i: statistics.median(y[i] for y in x)
    pr.append(dict(mode=mode, n=n, rounds=len(x), gflops='%.3f' % med(0),
                   gflops_min='%.3f' % min(y[0] for y in x),
                   gflops_max='%.3f' % max(y[0] for y in x),
                   us_per_call='%.4f' % (med(1)*1e6), packB='%.4f' % med(2),
                   packA='%.4f' % med(3), macrokernel='%.4f' % med(4),
                   other='%.4f' % med(5)))
with open('data/summary/prof.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(pr[0].keys()))
    w.writeheader()
    w.writerows(pr)
print('%d sweep results (%d WRONG), %d sizes; %d profile rows'
      % (len(rows), sum(r['check'] == 'WRONG' for r in rows), len(out), len(pr)))
