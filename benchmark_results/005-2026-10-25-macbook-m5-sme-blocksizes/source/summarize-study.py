#!/usr/bin/env python3
#
#  summarize-study.py STUDY REF OUT PLOTS : the block size study of the
#  SME2 backend.
#
#      STUDY   outputs of study-shapes.sh (STUDY/bMC_KC_NC-ROUND.txt): the
#              12 fixed SME2 configurations, one session
#      REF     outputs of study-ref.sh (REF/V-ROUND.txt): Accelerate, NEON
#              and two SME2 configurations as anchors, a second session
#
#  Writes
#
#      OUT/study.csv, OUT/reference.csv   one line per run and round
#      OUT/summary/square.csv             m = n = k: medians per variant
#      OUT/summary/small_m.csv            n = k = 2048, m varies
#      OUT/summary/small_n.csv            m = k = 2048, n varies
#      OUT/summary/small_k.csv            m = n = 2048, k varies
#      OUT/summary/best.csv               per shape: the best fixed SME2
#                                         configuration, the runner-up, the
#                                         gap and whether the gap is clear
#                                         or within the noise, the
#                                         configurations tied with the best,
#                                         the same comparison against the
#                                         default 384, 1024, 1024, and the
#                                         configurations clearly worse than
#                                         the best
#      OUT/summary/sessions.csv           the anchors in both sessions
#      PLOTS/_selection.gp                the SME2 configurations of the
#                                         readable plots (gnuplot string)
#
#  Criterion: the best configuration (highest median of the rounds) is
#  "clear" if its slowest round is faster than the fastest round of every
#  other configuration and its median is at least 1% higher than that of
#  the runner-up; otherwise "noise", and every configuration whose fastest
#  round reaches the slowest round of the best (or whose median is within
#  1%) is listed as tied.  The same criterion decides whether the best is
#  clearly better than the default of the library (384, 1024, 1024), and
#  which configurations are clearly worse than the best (fastest round
#  below the slowest round of the best and median at least 1% lower).  The SME2 configurations are compared only
#  within the session of the study; Accelerate and NEON come from the
#  second session (sessions.csv shows the difference of the anchors).
#
import csv
import glob
import os
import statistics
import sys

STUDY, REF, OUT, PLOTS = sys.argv[1:5]
os.makedirs(os.path.join(OUT, 'summary'), exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)


def read(directory):
    rows, g = [], {}
    for path in sorted(glob.glob(os.path.join(directory, '*.txt'))):
        v, rnd = os.path.basename(path)[:-4].rsplit('-', 1)
        for line in open(path):
            w = line.split()
            if len(w) >= 9 and w[0].isdigit():
                shape = tuple(int(x) for x in w[:3])
                rows.append([v] + list(shape) + [int(rnd), float(w[4]), w[8]])
                g.setdefault((v, shape), []).append(float(w[4]))
    return rows, g


def write(name, header, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        c = csv.writer(f)
        c.writerow(header)
        c.writerows(rows)


def label(v):
    if v.startswith('b'):
        mc, kc, nc = v[1:].split('_')
        return 'MC=%s KC=%s NC=%s' % (mc, kc, nc)
    return v


srows, sg = read(STUDY)
rrows, rg = read(REF)
header = ['variant', 'm', 'n', 'k', 'round', 'gflops', 'check']
write('study.csv', header, srows)
write('reference.csv', header, rrows)

CONFIGS = sorted({v for v, _ in sg})
REFS = ['accelerate', 'neon']
med = {key: statistics.median(x) for key, x in sg.items()}
rmed = {key: statistics.median(x) for key, x in rg.items()}

FAMILIES = [
    ('square', 'n', lambda s: s[0] == s[1] == s[2], lambda s: s[0]),
    ('small_m', 'm', lambda s: s[1] == s[2] == 2048 and s[0] < 2048,
     lambda s: s[0]),
    ('small_n', 'n', lambda s: s[0] == s[2] == 2048 and s[1] < 2048,
     lambda s: s[1]),
    ('small_k', 'k', lambda s: s[0] == s[1] == 2048 and s[2] < 2048,
     lambda s: s[2]),
]
shapes = sorted({s for _, s in sg})
for fam, xname, member, x in FAMILIES:
    out = []
    for s in sorted((s for s in shapes if member(s)), key=x):
        out.append([x(s)]
                   + [round(rmed[(v, s)], 1) if (v, s) in rmed else 'NaN'
                      for v in REFS]
                   + [round(med[(v, s)], 1) if (v, s) in med else 'NaN'
                      for v in CONFIGS])
    write('summary/%s.csv' % fam, [xname] + REFS + CONFIGS, out)

best_rows, winners = [], {}
for fam, xname, member, x in FAMILIES:
    for s in sorted((s for s in shapes if member(s)), key=x):
        ranked = sorted(CONFIGS, key=lambda v: -med[(v, s)])
        b, r = ranked[0], ranked[1]
        lo_b = min(sg[(b, s)])
        gap = med[(b, s)] / med[(r, s)] - 1
        clear = all(max(sg[(v, s)]) < lo_b for v in ranked[1:]) and gap >= 0.01
        tied = [v for v in ranked[1:]
                if max(sg[(v, s)]) >= lo_b or med[(v, s)] >= 0.99 * med[(b, s)]]
        winners.setdefault(b, []).append((fam, x(s)))
        d = 'b384_1024_1024'
        gain = med[(b, s)] / med[(d, s)] - 1
        clear_d = b != d and max(sg[(d, s)]) < lo_b and gain >= 0.01
        worse = [v for v in ranked[1:] if max(sg[(v, s)]) < lo_b
                 and med[(v, s)] <= 0.99 * med[(b, s)]]
        spread = max((max(sg[(v, s)]) - min(sg[(v, s)])) / med[(v, s)]
                     for v in CONFIGS)
        best_rows.append([fam, '%dx%dx%d' % s, label(b), round(med[(b, s)], 1),
                          label(r), round(med[(r, s)], 1),
                          round(100 * gap, 1), 'clear' if clear else 'noise',
                          ' '.join(label(v) for v in tied),
                          round(100 * spread, 1),
                          round(rmed[('accelerate', s)], 1),
                          round(rmed[('neon', s)], 1),
                          round(med[(d, s)], 1), round(100 * gain, 1),
                          'clear' if clear_d else ('default' if b == d
                                                   else 'noise'),
                          ' '.join(label(v) for v in worse)])
write('summary/best.csv',
      ['family', 'shape', 'best', 'best_gflops', 'runner_up',
       'runner_up_gflops', 'gap_percent', 'verdict', 'tied_with_best',
       'max_spread_percent', 'accelerate_gflops',
       'neon_gflops', 'default_gflops', 'gain_over_default_percent',
       'verdict_vs_default', 'clearly_worse_than_best'], best_rows)

out = []
for s in shapes:
    for v in ('b384_1024_1024', 'b608_2240_3072'):
        if (v, s) in rmed:
            out.append(['%dx%dx%d' % s, v, round(med[(v, s)], 1),
                        round(rmed[(v, s)], 1),
                        round(rmed[(v, s)] / med[(v, s)], 3)])
write('summary/sessions.csv',
      ['shape', 'variant', 'study_gflops', 'reference_gflops',
       'reference_over_study'], out)

#  Selection for the readable plots: every configuration that is the best
#  (highest median) for at least 3 of the shapes, plus the default of the
#  library (384, 1024, 1024) and the block sizes of BLIS PR #921 (608,
#  2240, 3072).  (Being the best for a single shape is mostly noise.)
sel = sorted({v for v, w in winners.items() if len(w) >= 3}
             | {'b384_1024_1024', 'b608_2240_3072'})
with open(os.path.join(PLOTS, '_selection.gp'), 'w') as f:
    f.write('#  SME2 configurations of the readable plots (written by\n'
            '#  summarize-study.py: every configuration that is the best for\n'
            '#  at least 3 shapes, the default 384, 1024, 1024 and the block\n'
            '#  sizes of BLIS PR #921 608, 2240, 3072)\n')
    f.write('SEL = "%s"\n' % ' '.join(sel))
print('study %d, reference %d measurements; best for (number of shapes): %s'
      % (len(srows), len(rrows), ', '.join('%s (%d)' % (label(v), len(w))
                                           for v, w in sorted(winners.items()))))
