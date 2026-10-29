#!/usr/bin/env python3
#
#  summarize.py RES RUN... : medians over the rounds of the outputs of
#  prof.cc (RES/RUN/rROUND/NAME.txt).  Prints one CSV line per measurement:
#
#      run,lib,mode,order,L2,KC,L3,pack,m,n,k,rounds,gflops,gflops_min,
#      gflops_max,t_call,packB,packA,mgemm,other
#
#  lib is ulmblas, or blis / openblas for the mode cblas (from the job
#  name); on heim ulmblas-sse or ulmblas-avx (job names with -sse- or
#  -avx-).  gflops_min and gflops_max are the extremes over the rounds; the
#  time fractions are the medians over the rounds.
#
import glob
import os
import statistics
import sys


def rows(res, run):
    for f in sorted(glob.glob(os.path.join(res, run, 'r*', '*.txt'))):
        name = os.path.basename(f)[:-4]
        rnd = os.path.basename(os.path.dirname(f))
        for line in open(f):
            w = line.split()
            if len(w) < 14 or w[0] == 'check':
                continue
            mode, order = w[0], w[1]
            pack = w[14] if len(w) > 14 else 'ulm:ulm'
            lib = 'ulmblas'
            for a in ('sse', 'avx'):
                if '-%s-' % a in name:
                    lib = 'ulmblas-' + a
            if mode == 'cblas':
                lib = name.split('-')[0]
                order, pack, w[2:5] = '-', '-', ['-', '-', '-']
            key = (run, lib, mode, order, w[2], w[3], w[4], pack,
                   int(w[5]), int(w[6]), int(w[7]))
            yield key, rnd, [float(x) for x in w[8:14]]


def main():
    res, runs = sys.argv[1], sys.argv[2:]
    data = {}
    for run in runs:
        for key, rnd, v in rows(res, run):
            data.setdefault(key, []).append(v)
    print('run,lib,mode,order,L2,KC,L3,pack,m,n,k,rounds,gflops,gflops_min,'
          'gflops_max,t_call,packB,packA,mgemm,other')
    for key in sorted(data, key=lambda k: (k[0], k[8]*k[9]*k[10], k[8:],
                                           k[1:8])):
        vs = data[key]
        med = [statistics.median(v[i] for v in vs) for i in range(6)]
        g = [v[0] for v in vs]
        print(','.join(str(x) for x in key) + ',%d,%.3f,%.3f,%.3f,%.4e,'
              '%.4f,%.4f,%.4f,%.4f' % (len(vs), med[0], min(g), max(g),
                                       med[1], med[2], med[3], med[4],
                                       med[5]))


main()
