#!/usr/bin/env python3
#
#  summarize-sme.py RES OUT [SWEEP...] : converts the raw outputs of bench-sme.sh
#  (RES/KIND-VARIANT-ROUND.txt) into
#
#      OUT/dgemm.csv      all dgemm measurements (bench2), one line per size,
#                         variant and round
#      OUT/level3.csv     all dsyrk, dsymm, dtrmm, dtrsm measurements
#      OUT/ukernel.csv    the micro kernels alone (ukernel, ukernel-blis)
#      OUT/smepeak.csv    the FMOPA peak (smepeak)
#      OUT/summary/*.csv  medians over the rounds with the spread
#                         (max - min) / median, and the same in wide tables
#                         (one column per variant), the input of the plots
#      OUT/blocksizes.csv the block size sweeps (prof -m inst, one line per
#                         run) from the files SWEEP...; with the summary
#                         summary/blocksizes.csv (GFLOP/s relative to the
#                         block sizes of BLIS, MC, KC, NC = 608, 2240, 3072)
#
#  Nothing is interpolated; missing measurements are left out.
#
import csv
import glob
import os
import re
import statistics
import sys

RES, OUT, SWEEPS = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(os.path.join(OUT, 'summary'), exist_ok=True)

VARIANTS = ['sme', 'neon', 'blis-sme', 'blis', 'openblas', 'accelerate']
NAMES = {
    'sme':        'ulmBLAS SME2 (32x16)',
    'neon':       'ulmBLAS NEON (6x8)',
    'blis-sme':   'BLIS PR #921 SME2 (32x16)',
    'blis':       'BLIS 2.0 NEON (8x6)',
    'openblas':   'OpenBLAS 0.3.34 NEON',
    'accelerate': 'Apple Accelerate',
}

def parse(path):
    name = os.path.basename(path)[:-4]
    m = re.match(r'(.*)-(\d+)$', name)
    stem, rnd = m.group(1), int(m.group(2))
    return stem, rnd

dgemm, level3, ukernel, peak = [], [], [], []
for path in sorted(glob.glob(os.path.join(RES, '*.txt'))):
    stem, rnd = parse(path)
    if stem == 'smepeak':
        for line in open(path):
            w = line.split()
            if w and w[0] == 'regs':
                peak.append(('regs', 'tiles=' + w[2], rnd, float(w[3])))
            elif w and w[0] == 'panel':
                peak.append(('panel', 'kc=' + w[2], rnd, float(w[-2])))
        continue
    if stem.startswith('ukernel-'):
        variant = stem[len('ukernel-'):]
        for line in open(path):
            w = line.split()
            if len(w) == 6 and w[0].isdigit():
                ukernel.append((variant, int(w[0]), int(w[1]), int(w[2]), rnd,
                                float(w[3]), float(w[4]), float(w[5])))
        continue
    kind, variant = None, None
    for v in sorted(VARIANTS, key=len, reverse=True):
        if stem.endswith('-' + v):
            kind, variant = stem[:-len(v)-1], v
            break
    if kind is None:
        continue
    for line in open(path):
        w = line.split()
        if len(w) >= 9 and w[0].isdigit():
            dgemm.append((kind, variant, int(w[0]), int(w[1]), int(w[2]),
                          float(w[3]), rnd, float(w[4]), float(w[5]),
                          float(w[6]), int(w[7]), w[8]))
        elif len(w) >= 6 and w[0].startswith('d') and w[1].isdigit():
            level3.append((w[0], variant, int(w[1]), rnd, float(w[2]),
                           float(w[3]), float(w[4]), int(w[5])))

def write(name, header, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        c = csv.writer(f)
        c.writerow(header)
        c.writerows(rows)

write('dgemm.csv', ['kind', 'variant', 'm', 'n', 'k', 'beta', 'round',
                    'gflops_median', 'gflops_min', 'gflops_max',
                    'calls_per_sample', 'check'], dgemm)
write('level3.csv', ['routine', 'variant', 'n', 'round', 'gflops_median',
                     'gflops_min', 'gflops_max', 'calls_per_sample'], level3)
write('ukernel.csv', ['variant', 'kc', 'MR', 'NR', 'round', 'gflops_median',
                      'gflops_min', 'gflops_max'], ukernel)
write('smepeak.csv', ['test', 'parameter', 'round', 'gflops'], peak)

def summarize(rows, key, value):
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(value(r))
    out = []
    for k in sorted(groups):
        g = groups[k]
        med = statistics.median(g)
        out.append(k + (len(g), round(med, 3),
                        round((max(g) - min(g)) / med, 4) if med else 0))
    return out

write('summary/dgemm.csv', ['variant', 'm', 'n', 'k', 'beta', 'rounds',
                            'gflops', 'spread'],
      summarize(dgemm, lambda r: (r[1], r[2], r[3], r[4], r[5]),
                lambda r: r[7]))
write('summary/level3.csv', ['routine', 'variant', 'n', 'rounds', 'gflops',
                             'spread'],
      summarize(level3, lambda r: (r[0], r[1], r[2]), lambda r: r[4]))
write('summary/ukernel.csv', ['variant', 'kc', 'rounds', 'gflops', 'spread'],
      summarize(ukernel, lambda r: (r[0], r[1]), lambda r: r[5]))
write('summary/smepeak.csv', ['test', 'parameter', 'rounds', 'gflops',
                              'spread'],
      summarize(peak, lambda r: (r[0], r[1]), lambda r: r[3]))
wrong = sum(1 for r in dgemm if r[11] == 'WRONG')
print('dgemm %d, level3 %d, ukernel %d, peak %d measurements, %d WRONG'
      % (len(dgemm), len(level3), len(ukernel), len(peak), wrong))

#  Wide tables for the gnuplot scripts (one column per variant, NaN where
#  a variant was not measured); the ratios are relative to ulmBLAS SME2.
def wide(rows, key, label, header, keys):
    table = {}
    for r in rows:
        table.setdefault(key(r), {})[r[0]] = r[-2]
    out = []
    for k in keys:
        if k in table:
            t = table[k]
            v = [t.get(x, 'NaN') for x in VARIANTS]
            rel = [round(t[x] / t['sme'], 3) if x in t and 'sme' in t
                   else 'NaN' for x in ('blis-sme', 'accelerate')]
            out.append([label(k)] + v + rel)
    write(header, ['case'] + [x.replace('-', '_') for x in VARIANTS]
          + ['blis_sme_over_sme', 'accelerate_over_sme'], out)

sd = summarize(dgemm, lambda r: (r[1], r[2], r[3], r[4], r[5]),
               lambda r: r[7])
dkey = lambda r: (r[1], r[2], r[3], r[4])
sizes = sorted({r[1] for r in sd if r[1] == r[2] == r[3]})
wide(sd, dkey, lambda k: k[0], 'summary/dgemm_square.csv',
     [(n, n, n, 1.0) for n in sizes])
SHAPES = [(1024, 1024, 4096), (4096, 1024, 1024), (1024, 4096, 1024),
          (3000, 2000, 1000), (1000, 3000, 2000),
          (2000, 2000, 128), (2000, 2000, 32), (4000, 4000, 64),
          (1024, 1024, 32),
          (2048, 256, 2048), (4096, 96, 4096), (2048, 64, 2048),
          (4096, 16, 4096),
          (256, 2048, 2048), (96, 4096, 4096), (64, 2048, 2048),
          (16, 4096, 4096)]
wide(sd, dkey, lambda k: '%dx%dx%d' % k[:3], 'summary/dgemm_shapes.csv',
     [s + (1.0,) for s in SHAPES])
wide(sd, dkey, lambda k: '%dx%dx%d' % k[:3], 'summary/dgemm_beta0.csv',
     [s + (0.0,) for s in [(256, 256, 256), (1024, 1024, 1024),
                           (2048, 2048, 2048), (2000, 2000, 32)]])
sl = summarize(level3, lambda r: (r[1], r[0], r[2]), lambda r: r[4])
for routine in ('dsyrk', 'dsymm', 'dtrmm', 'dtrsm'):
    wide([r for r in sl if r[1] == routine], lambda r: r[2], lambda k: k,
         'summary/level3_%s.csv' % routine, [512, 1024, 2048])
su = summarize(ukernel, lambda r: (r[0], r[1]), lambda r: r[5])
wide(su, lambda r: r[1], lambda k: k, 'summary/ukernel_kc.csv',
     sorted({r[1] for r in su}))

#  Block size sweeps: lines "inst ulm MC KC NC m n k GFLOPS seconds
#  packB packA macro rest packing" of prof (fractions of the time).
sweep = []
for path in SWEEPS:
    name = os.path.basename(os.path.dirname(path)) + '/' + \
        os.path.basename(path)[:-4]
    for line in open(path):
        w = line.split()
        if len(w) >= 14 and w[0] == 'inst':
            sweep.append([name] + [int(x) for x in w[2:8]]
                         + [float(w[8])] + [float(x) for x in w[10:13]])
if sweep:
    write('blocksizes.csv', ['sweep', 'mc', 'kc', 'nc', 'm', 'n', 'k',
                             'gflops', 'frac_pack_b', 'frac_pack_a',
                             'frac_macro'], sweep)
    CONFIGS = [(608, 2240, 3072), (384, 2240, 3072), (384, 1024, 3072),
               (608, 1024, 1024), (256, 1024, 1024), (384, 1024, 1024),
               (384, 1536, 1024)]
    g = {}
    for r in sweep:
        if r[0].endswith('/sweep'):
            g[(tuple(r[1:4]), tuple(r[4:7]))] = r[7]
    out = []
    for shape in sorted({k[1] for k in g}, key=lambda s: -s[0]*s[1]*s[2]):
        base = g.get((CONFIGS[0], shape))
        out.append(['%dx%dx%d' % shape]
                   + [round(g[(c, shape)] / base, 3) if (c, shape) in g
                      else 'NaN' for c in CONFIGS])
    write('summary/blocksizes.csv',
          ['shape'] + ['mc%d_kc%d_nc%d' % c for c in CONFIGS], out)
