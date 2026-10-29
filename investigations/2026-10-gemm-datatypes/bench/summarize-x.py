#!/usr/bin/env python3
#
#  summarize-x.py RESDIR PREC REF [ROUNDS]: summary of benchx results
#  RESDIR/PREC-VARIANT-ROUND.txt (columns m n k median min max calls err
#  check).  Per variant: GFLOP/s (median over the rounds of the per-round
#  medians) at selected sizes, the median over n >= 1000 and n <= 200 of the
#  per-size ratio to the variant REF (per round, then the median over the
#  rounds), and the number of results that are not "ok".  ROUNDS (e.g. 1)
#  restricts the evaluation to the rounds 1 ... ROUNDS.  Writes a Markdown
#  table to standard output and data/runs.csv style rows with -csv.
#
import glob, os, re, statistics, sys

def load(resdir, prec, maxround):
    data, bad = {}, {}
    for f in glob.glob(os.path.join(resdir, '%s-*-*.txt' % prec)):
        m = re.match(r'%s-(.+)-(\d+)\.txt$' % prec, os.path.basename(f))
        v, r = m.group(1), int(m.group(2))
        if maxround and r > maxround:
            continue
        for line in open(f):
            w = line.split()
            if len(w) < 9:
                continue
            n = int(w[0])
            data.setdefault(v, {}).setdefault(r, {})[n] = float(w[3])
            if w[8] != 'ok':
                bad[v] = bad.get(v, 0) + 1
    return data, bad

def main():
    args = [a for a in sys.argv[1:] if a != '-csv']
    resdir, prec, ref = args[0], args[1], args[2]
    maxround = int(args[3]) if len(args) > 3 else 0
    data, bad = load(resdir, prec, maxround)
    if '-csv' in sys.argv:
        print('prec,variant,round,n,gflops')
        for v in sorted(data):
            for r in sorted(data[v]):
                for n in sorted(data[v][r]):
                    print('%s,%s,%d,%d,%.3f' % (prec, v, r, n, data[v][r][n]))
        return
    sizes = [50, 100, 200, 500, 1000, 2000, 3000, 4000]
    def med(v, n):
        x = [data[v][r][n] for r in data[v] if n in data[v][r]]
        return statistics.median(x) if x else None
    def ratio(v, lo, hi):
        per = []
        for r in data[v]:
            if r not in data.get(ref, {}):
                rr = min(data[ref])        # reference measured in another round
            else:
                rr = r
            q = [data[v][r][n] / data[ref][rr][n] for n in data[v][r]
                 if lo <= n <= hi and n in data[ref][rr]]
            if q:
                per.append(statistics.median(q))
        return statistics.median(per) if per else None
    order = sorted(data, key=lambda v: -(med(v, 2000) or 0))
    print('| variant | rounds | ' + ' | '.join('n=%d' % n for n in sizes)
          + ' | /%s n>=1000 | /%s n<=200 | not ok |' % (ref, ref))
    print('|---|---:|' + '---:|' * (len(sizes) + 3))
    for v in order:
        cells = ['%.1f' % med(v, n) if med(v, n) is not None else '-' for n in sizes]
        a, b = ratio(v, 1000, 4000), ratio(v, 0, 200)
        print('| %s | %d | %s | %s | %s | %d |' % (
            v, len(data[v]), ' | '.join(cells),
            '%.3f' % a if a else '-', '%.3f' % b if b else '-', bad.get(v, 0)))

if __name__ == '__main__':
    main()
