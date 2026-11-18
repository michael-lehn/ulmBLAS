#!/usr/bin/env python3
#  table.py REF.txt A.txt [B.txt ...]: Markdown tables of the results of
#  run.sh, GFLOP/s per operation and n for 1 and 6 threads, the others as
#  ratio to REF
import sys

files = sys.argv[1:]
data = []
for f in files:
    d = {}
    for line in open(f):
        p = line.split()
        if len(p) != 5:
            continue
        name, t, op, n, g = p
        d[(int(t), op, int(n))] = float(g)
    data.append(d)
ops = []
for k in data[0]:
    if k[1] not in ops:
        ops.append(k[1])
sizes = sorted({k[2] for k in data[0]})
names = [f.split('/')[-1].replace('.txt', '') for f in files]
for t in sorted({k[0] for k in data[0]}):
    print("\n### %d thread%s: GFLOP/s %s, then %s as ratio to it\n"
          % (t, 's' if t > 1 else '', names[0], ', '.join(names[1:])))
    print("| op | " + " | ".join(str(n) for n in sizes) + " |")
    print("|" + "---|" * (len(sizes) + 1))
    for op in ops:
        cells = []
        for n in sizes:
            ref = data[0].get((t, op, n))
            parts = []
            for i, d in enumerate(data):
                v = d.get((t, op, n))
                if v is None or ref is None:
                    parts.append('-')
                elif i == 0:
                    parts.append("%.1f" % v)
                else:
                    parts.append("%.2f" % (v / ref))
            cells.append(" / ".join(parts))
        print("| %s | " % op + " | ".join(cells) + " |")
