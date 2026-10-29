#!/usr/bin/env python3
#
#  summarize.py OUT: summary of a run of run.sh (make bench-gemm).  Reads
#  OUT/raw/LIB-ROUND.txt (benchx.c: m n k median min max calls err check),
#  writes OUT/gflops.dat (n and the GFLOP/s of every library: median over
#  the rounds) and prints a Markdown summary: GFLOP/s at a few sizes, the
#  ratio to ulmBLAS (median over the sizes of the per-size ratios), the
#  results that are not "ok", and the library each program reported.
#
import glob, os, re, statistics, sys

out = sys.argv[1]
data, bad, ident = {}, {}, {}
for f in sorted(glob.glob(os.path.join(out, 'raw', '*-*.txt'))):
    m = re.match(r'(.+)-(\d+)\.txt$', os.path.basename(f))
    lib, rnd = m.group(1), int(m.group(2))
    for line in open(f):
        if line.startswith('#'):
            if line.startswith('# library') or ' from ' in line:
                ident.setdefault(lib, []).append(line[2:].strip())
            continue
        w = line.split()
        if len(w) < 9:
            continue
        data.setdefault(lib, {}).setdefault(int(w[0]), []).append(float(w[3]))
        if w[8] != 'ok':
            bad[lib] = bad.get(lib, 0) + 1

libs = sorted(data, key=lambda l: (l != 'ulmblas', l))
sizes = sorted(set(n for l in libs for n in data[l]))
med = {l: {n: statistics.median(v) for n, v in data[l].items()} for l in libs}

with open(os.path.join(out, 'gflops.dat'), 'w') as f:
    f.write('# n ' + ' '.join(libs) + '\n')
    for n in sizes:
        f.write('%d %s\n' % (n, ' '.join('%.3f' % med[l][n] if n in med[l]
                                        else 'NaN' for l in libs)))

show = [n for n in (50, 100, 200, 500, 1000, 2000, 4000) if n in sizes]
if len(show) < 3:
    show = sizes[:8]
print('GFLOP/s (median over %d round(s)):\n' % max(len(v) for l in libs
                                                    for v in data[l].values()))
print('| library | ' + ' | '.join('n = %d' % n for n in show)
      + ' | vs. ulmBLAS | not ok |')
print('|---|' + '---:|' * (len(show) + 2))
for l in libs:
    r = '-'
    if 'ulmblas' in med and l != 'ulmblas':
        q = [med[l][n] / med['ulmblas'][n] for n in med[l] if n in med['ulmblas']]
        r = '%.2f' % statistics.median(q)
    print('| %s | %s | %s | %d |' % (
        l, ' | '.join('%.1f' % med[l][n] if n in med[l] else '-' for n in show),
        r, bad.get(l, 0)))
print('\n"vs. ulmBLAS": median over all sizes of GFLOP/s(library) / '
      'GFLOP/s(ulmBLAS).  "not ok": results that failed the check.\n')
print('Libraries as reported by the programs:\n')
for l in libs:
    seen = []
    for s in ident.get(l, []):
        if s not in seen:
            seen.append(s)
    print('- %s: %s' % (l, '; '.join(seen)))
