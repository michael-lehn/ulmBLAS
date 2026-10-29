#!/usr/bin/env python3
#
#  tables.py RES OUT : tables of the profile campaign from the medians in
#  RES (P1.csv, P3.csv of summarize.py; P4.csv, P6.csv, heim_*.csv of
#  summarize-bench.py; see make-summaries.sh) into the directory OUT:
#
#  breakdown.csv   (P1) time of one DGEMM call of the instrumented driver
#                  split into: kernel (flops at the speed of the micro
#                  kernel alone, mode kern), macro kernel (time of the
#                  driver without packing, mode nopack, minus kernel), pack A,
#                  pack B (timers of mode inst), rest (inst minus the parts);
#                  as fractions of the time of mode inst; plus GFLOP/s of
#                  the library, the driver, BLIS and OpenBLAS
#  packing.csv     (P3) GFLOP/s of the driver with the packing variants,
#                  relative to the packing of ulmBLAS (ulm:ulm)
#  blocksizes.csv  (P3) GFLOP/s with other block sizes and the loop order of
#                  BLIS, relative to the block sizes of ulmBLAS
#  overhead.csv    (P3) time of one call for small n: library, driver and
#                  the calls of the memory pool
#  before_after.csv (P4) GFLOP/s of the CBLAS library: master, p4, p8, BLIS
#  final.csv       (P6) master, pT (the committed change) and pTG
#  heim_*_rel.csv  (heim) master and the ratios of the variants to master
#  final_all.csv   pT relative to master on the M5 (P6) and on heim (SSE3,
#                  AVX/FMA)
#
import csv
import os
import sys

res, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)


def load(name):
    return list(csv.DictReader(open(os.path.join(res, name))))


def shape(r):
    return '%sx%sx%s' % (r['m'], r['n'], r['k'])


def flops(r):
    return 2.0*int(r['m'])*int(r['n'])*int(r['k'])


#  breakdown (P1)
p1 = {}
for r in load('P1.csv'):
    p1.setdefault((int(r['m']), int(r['n']), int(r['k'])), {})[
        (r['lib'], r['mode'], r['order'])] = r
with open(os.path.join(out, 'breakdown.csv'), 'w') as f:
    f.write('shape,kernel,macro,packA,packB,rest,lib,inst,nopack,kern,'
            'blis,openblas,lib_over_blis\n')
    for s in sorted(p1, key=lambda s: (s[0]*s[1]*s[2], s)):
        x = p1[s]
        inst = x[('ulmblas', 'inst', 'ulm')]
        nop = x[('ulmblas', 'nopack', 'ulm')]
        kern = x[('ulmblas', 'kern', 'ulm')]
        g = lambda k: float(x[k]['gflops'])
        t = float(inst['t_call'])
        tk = flops(inst)/(float(kern['gflops'])*1e9)
        tm = float(nop['t_call']) - tk
        pa, pb = float(inst['packA'])*t, float(inst['packB'])*t
        rest = t - tk - tm - pa - pb
        f.write('%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,'
                '%.4f\n' % ('x'.join(map(str, s)), tk/t, tm/t, pa/t, pb/t,
                            rest/t, g(('ulmblas', 'lib', 'ulm')),
                            g(('ulmblas', 'inst', 'ulm')),
                            g(('ulmblas', 'nopack', 'ulm')),
                            g(('ulmblas', 'kern', 'ulm')),
                            g(('blis', 'cblas', '-')),
                            g(('openblas', 'cblas', '-')),
                            g(('ulmblas', 'lib', 'ulm')) /
                            g(('blis', 'cblas', '-'))))

#  packing variants and block sizes (P3)
p3 = load('P3.csv')
V = ['ulm:ulm', 'unit:unit', 'blk4:unit', 'blk8:unit', 'blk4:trn',
     'blk8:trn']
pk = {}
for r in p3:
    if (r['mode'] == 'inst' and r['order'] == 'ulm'
            and (r['L2'], r['KC'], r['L3']) == ('960', '1024', '4080')):
        pk.setdefault(shape(r), {})[r['pack']] = r
with open(os.path.join(out, 'packing.csv'), 'w') as f:
    f.write('shape,' + ','.join(V) + ',' +
            ','.join(v + '_rel' for v in V) + '\n')
    for s, x in pk.items():
        if not all(v in x for v in V):
            continue
        b = float(x['ulm:ulm']['gflops'])
        f.write(s + ',' + ','.join(x[v]['gflops'] for v in V) + ',' +
                ','.join('%.4f' % (float(x[v]['gflops'])/b) for v in V) +
                '\n')
B = [('ulm', '960', '1024', '4080'), ('ulm', '960', '1024', '8160'),
     ('ulm', '480', '2048', '4080'), ('ulm', '320', '3072', '4080'),
     ('blis', '256', '3072', '8184')]
bs = {}
for r in p3:
    if r['mode'] == 'inst' and r['pack'] == 'blk4:unit':
        bs.setdefault(shape(r), {})[(r['order'], r['L2'], r['KC'],
                                     r['L3'])] = r
names = ['%s_%s_%s_%s' % b for b in B]
with open(os.path.join(out, 'blocksizes.csv'), 'w') as f:
    f.write('shape,' + ','.join(names) + ',' +
            ','.join(n + '_rel' for n in names) + '\n')
    for s, x in bs.items():
        if not all(b in x for b in B):
            continue
        b0 = float(x[B[0]]['gflops'])
        f.write(s + ',' + ','.join(x[b]['gflops'] for b in B) + ',' +
                ','.join('%.4f' % (float(x[b]['gflops'])/b0) for b in B) +
                '\n')
with open(os.path.join(out, 'overhead.csv'), 'w') as f:
    f.write('n,lib_ns,inst_ns,pool_ns\n')
    ov = {}
    for r in p3:
        if r['mode'] in ('lib', 'inst', 'pool') and int(r['m']) <= 64 \
                and r['m'] == r['n'] == r['k']:
            ov.setdefault(int(r['m']), {})[r['mode']] = \
                float(r['t_call'])*1e9
    for n in sorted(ov):
        f.write('%d,%.1f,%.1f,%.1f\n' % (n, ov[n]['lib'], ov[n]['inst'],
                                         ov[n]['pool']))

#  before / after (P4)
p4 = {}
for r in load('P4.csv'):
    p4.setdefault((r['routine'], int(r['m']), int(r['n']), int(r['k'])),
                  {})[r['variant']] = r
with open(os.path.join(out, 'before_after.csv'), 'w') as f:
    f.write('routine,shape,master,p4,p8,blis,p4_over_master,'
            'p8_over_master,blis_over_master,p4_over_blis,master_spread,'
            'p4_spread\n')
    for k in p4:
        x = p4[k]
        g = lambda v: float(x[v]['gflops'])
        sp = lambda v: (float(x[v]['gflops_max']) -
                        float(x[v]['gflops_min']))/g(v)
        f.write('%s,%dx%dx%d,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,'
                '%.4f\n'
                % (k + (g('master'), g('p4'), g('p8'), g('blis'),
                        g('p4')/g('master'), g('p8')/g('master'),
                        g('blis')/g('master'), g('p4')/g('blis'),
                        sp('master'), sp('p4'))))

#  final comparison on the M5 (P6): master, pT (the committed change), pTG
p6 = {}
for r in load('P6.csv'):
    p6.setdefault((r['routine'], int(r['m']), int(r['n']), int(r['k'])),
                  {})[r['variant']] = r
with open(os.path.join(out, 'final.csv'), 'w') as f:
    f.write('routine,shape,master,pT,pTG,pT_over_master,pTG_over_master,'
            'master_spread,pT_spread\n')
    for k in p6:
        x = p6[k]
        g = lambda v: float(x[v]['gflops'])
        sp = lambda v: (float(x[v]['gflops_max']) -
                        float(x[v]['gflops_min']))/g(v)
        f.write('%s,%dx%dx%d,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f\n'
                % (k + (g('master'), g('pT'), g('pTG'), g('pT')/g('master'),
                        g('pTG')/g('master'), sp('master'), sp('pT'))))

#  heim (summaries of summarize-bench.py, variants NAME-sse and NAME-avx):
#  GFLOP/s of master and the ratios of the other variants to master
for name, variants in (('heim_pack', ['p4', 'p8']),
                       ('heim_isolation_ab', ['p4', 'pA', 'pB']),
                       ('heim_isolation_ab2', ['pA', 'pG', 'pU']),
                       ('heim_threshold', ['pG', 'pU']),
                       ('heim_final', ['pT', 'pTG'])):
    hd = {}
    for r in load(name + '.csv'):
        v, a = r['variant'].rsplit('-', 1)
        hd.setdefault((a, r['routine'], int(r['m']), int(r['n']),
                       int(r['k'])), {})[v] = r
    with open(os.path.join(out, name + '_rel.csv'), 'w') as f:
        f.write('arch,routine,shape,master,' +
                ','.join(v + '_over_master' for v in variants) + '\n')
        for k in sorted(hd, key=lambda k: (k[0], k[1], k[2]*k[3]*k[4])):
            x = hd[k]
            g = lambda v: float(x[v]['gflops'])
            f.write('%s,%s,%dx%dx%d,%.3f,' % (k + (g('master'),)) +
                    ','.join('%.4f' % (g(v)/g('master')) for v in variants)
                    + '\n')

#  final change on both machines: pT relative to master (M5: P6; heim:
#  heim_final, SSE3 and AVX/FMA)
hf = {}
for r in load('heim_final.csv'):
    v, a = r['variant'].rsplit('-', 1)
    hf.setdefault((r['routine'], int(r['m']), int(r['n']), int(r['k']), a),
                  {})[v] = float(r['gflops'])
with open(os.path.join(out, 'final_all.csv'), 'w') as f:
    f.write('routine,shape,m5,heim_avx,heim_sse\n')
    for k in p6:
        x = p6[k]
        m5 = float(x['pT']['gflops'])/float(x['master']['gflops'])
        h = [hf[k + (a,)]['pT']/hf[k + (a,)]['master']
             if k + (a,) in hf else float('nan') for a in ('avx', 'sse')]
        f.write('%s,%dx%dx%d,%.4f,%.4f,%.4f\n' % (k + (m5, h[0], h[1])))
