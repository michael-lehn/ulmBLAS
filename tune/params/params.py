"""Parameter tuning (make tune-params, see tune/README.md): fits the
parameters of ulmblas/config/tuned.h from the measurements of
tune/params/measure.cc and generates the header of the platform.

    params.py fit GROUP PLATFORM DATA PARAM RESULTS
        fits the parameters of GROUP from DATA (lines of measure run) with
        the parameters of the build PARAM (output of measure param for the
        types s, d, c, z); stores them in RESULTS/params.json and the report
        in RESULTS/params.md

    params.py verify PLATFORM DATA RESULTS
        adds the comparison of the defaults written by hand with the tuned
        parameters (lines of measure verify) to the report

    params.py merge PLATFORM RESULTS OTHER.json ...
        adds the values and reports of runs on other machines of the
        platform (e.g. one type each) to RESULTS/params.json

    params.py generate PLATFORM CONDITION RESULTS TUNEDDIR
        writes TUNEDDIR/PLATFORM.h from RESULTS/params.json (CONDITION: the
        build under which the parameters apply) and the index
        TUNEDDIR/tuned.h

As for the selection tables of gemm (tune/gemm/gentable.py): the structure of
each decision is fixed, make tune measures candidates and takes the values
that maximize the geometric mean of the performance relative to the best
candidate per problem.  A value replaces the one of the build only if it
gains at least GAIN in the geometric mean and is not slower than the build's
by more than TOL on any problem.
"""

import collections
import datetime
import json
import math
import os
import re
import sys

GAIN = 0.01
#  decisions by size: the build's choice for a size unless another gains
#  more than KEEP
KEEP = 0.05
TOL = 0.04
TYPES = "sdcz"
TYPE_INDEX = {"s": "tunedS", "d": "tunedD", "c": "tunedC", "z": "tunedZ"}

#  candidates (the same as in measure.cc)
MIN_WORK = [1 << 12, 1 << 13, 1 << 14, 3 << 13, 1 << 15, 3 << 14, 1 << 16,
            3 << 15, 1 << 17, 3 << 16, 1 << 18, 3 << 17, 1 << 19, 3 << 18,
            1 << 20, 3 << 19, 1 << 21, 1 << 22]
SHARED_BLOCKS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 1 << 30]
SHARED_PANELS = [2, 4, 8, 16, 32, 64, 1 << 30]
MIN_PANELS = [1, 2, 4, 8, 16]


def geomean(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 1.0


def load(data):
    """rows by (type, op, m, n, k, pt): {setting: [medians], threads,
    MR, NR, KC1}; the median over the passes"""
    rows = collections.OrderedDict()
    for line in open(data):
        x = line.split("\t")
        if len(x) < 15:
            continue
        key = (x[1], x[2], int(x[3]), int(x[4]), int(x[5]), int(x[6]))
        r = rows.setdefault(key, {"perf": collections.defaultdict(list),
                                  "threads": {}, "info": (0, 0, 0)})
        r["perf"][x[8]].append(float(x[9]))
        r["threads"][x[8]] = int(x[7])
        r["info"] = (int(x[12]), int(x[13]), int(x[14]))
    for r in rows.values():
        r["perf"] = {s: sorted(v)[len(v) // 2] for s, v in r["perf"].items()}
    return rows


def load_param(path):
    """per type: {'default': {...}, 'current': {...}} (measure param)"""
    out = {}
    t = None
    for line in open(path):
        x = line.rstrip("\n").split("\t")
        if len(x) == 1 and x[0] in TYPES:
            t = x[0]
            out[t] = {}
            continue
        if t and len(x) > 1:
            out[t][x[0]] = dict(kv.split("=", 1) for kv in x[1:])
    return out


def steps_eval(text, x):
    v = [int(a) for a in text.split(",")]
    i = 0
    while 2 * i + 1 < len(v) and x >= v[2 * i + 1]:
        i += 1
    return v[2 * i]


def choose(names, score, worst, current):
    """the candidate with the best score whose worst ratio to the build's
    value is at least 1 - TOL; the build's (current) unless it gains GAIN"""
    ok = [c for c in names if worst[c] >= 1 - TOL]
    best = max(ok, key=lambda c: score[c]) if ok else current
    if current in score and score[best] < score[current] * (1 + GAIN):
        return current
    return best


def table(header, lines):
    out = ["| " + " | ".join(header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in line) + " |" for line in lines]
    return out


#
#  gemm-threads: gemmMinWork per type
#
def threads_by_work(m, n, k, w, pool):
    by = m * n * k // w
    return 1 if by < 2 else min(by, pool)


def steps_text(steps):
    """[(bound, value)] (first bound unused) as v0,b0,v1,..."""
    text = [str(steps[0][1])]
    for b, v in steps[1:]:
        text += [str(b), str(v)]
    return ",".join(text)


def merge_steps(works, values, maxsteps=8):
    """steps over increasing works with a value each: neighbours with the
    same value merged, bounds at the geometric means of neighbouring
    works; at most maxsteps bounds (the shortest steps join a neighbour)"""
    steps = []
    for i, v in enumerate(values):
        if steps and steps[-1][1] == v:
            continue
        b = 0 if i == 0 else int(math.sqrt(works[i - 1] * works[i]))
        steps.append((b, v))
    while len(steps) > maxsteps + 1:
        i = min(range(1, len(steps)),
                key=lambda j: (steps[j + 1][0] if j + 1 < len(steps)
                               else 1 << 62) / steps[j][0])
        del steps[i]
        #  equal neighbours again
        k = 1
        while k < len(steps):
            if steps[k][1] == steps[k - 1][1]:
                del steps[k]
            else:
                k += 1
    return steps


def fit_gemm_threads(rows, param, pool):
    """gemmMinWork (the rule m n k / MinWork), then gemmThreads: a decision
    by the work m n k with explicit numbers of threads where they gain at
    least STEP_GAIN over the rule (-1: the rule, 0: all threads)"""
    STEP_GAIN = 0.03
    values, report = {}, []
    for t in TYPES:
        probs = [(k, r) for k, r in rows.items() if k[0] == t]
        if not probs:
            continue
        cur = int(param[t]["current"]["gemmMinWork"])
        cur_steps = param[t]["current"].get("gemmThreads", "-1")
        cands = sorted(set(MIN_WORK + [cur]))

        def best_of(r):
            return max(v for c, v in r["perf"].items() if c != "current")

        def perf(r, key, w):
            return r["perf"].get("t%d" % threads_by_work(key[2], key[3],
                                                         key[4], w, pool))

        score, worst = {}, {}
        for w in cands:
            rel, vs = [], []
            for key, r in probs:
                p, pc = perf(r, key, w), perf(r, key, cur)
                if p is None or pc is None:
                    break
                rel.append(p / best_of(r))
                vs.append(p / pc)
            else:
                score[w] = geomean(rel)
                worst[w] = min(vs)
        w = choose(sorted(score), score, worst, cur)
        if w != cur:
            values["gemmMinWork." + t] = w

        #  the decision by work: numbers of threads that do not decrease
        #  with the work (measured for all problems), the one with the best
        #  geometric mean; used if it gains STEP_GAIN over the rule
        by = collections.OrderedDict()
        for key, r in sorted(probs, key=lambda kr: kr[0][2] * kr[0][3] *
                             kr[0][4]):
            by.setdefault(key[2] * key[3] * key[4], []).append((key, r))
        counts = sorted({int(c[1:]) for _, r in probs for c in r["perf"]
                         if c.startswith("t")})
        counts = [n for n in counts
                  if all("t%d" % n in r["perf"] for _, r in probs)]
        works = list(by)
        gain = [[sum(math.log(r["perf"]["t%d" % n] / best_of(r))
                     for _, r in krs) for n in counts]
                for krs in by.values()]
        #  dp[i][j]: the best sum for the works 0..i with counts[j] at i
        dp = [gain[0][:]]
        arg = [[None] * len(counts)]
        for i in range(1, len(works)):
            row, back = [], []
            for jj in range(len(counts)):
                k = max(range(jj + 1), key=lambda q: dp[i - 1][q])
                row.append(dp[i - 1][k] + gain[i][jj])
                back.append(k)
            dp.append(row)
            arg.append(back)
        jj = max(range(len(counts)), key=lambda q: dp[-1][q])
        seq = [0] * len(works)
        for i in range(len(works) - 1, -1, -1):
            seq[i] = counts[jj]
            jj = arg[i][jj] if i > 0 else jj
        choice = [0 if n >= pool else n for n in seq]
        steps = merge_steps(works, choice)
        text = steps_text(steps)
        rule = geomean(perf(r, key, w) / best_of(r) for key, r in probs)
        tree = geomean(r["perf"]["t%d" % (pool if steps_eval(text, key[2] *
                                                             key[3] *
                                                             key[4]) == 0
                                          else steps_eval(text, key[2] *
                                                          key[3] * key[4]))]
                       / best_of(r) for key, r in probs)
        if tree < rule * (1 + STEP_GAIN):
            text = "-1"
        if text != cur_steps:
            values["gemmThreads." + t] = [int(x) for x in text.split(",")]

        def threads_of(key, steps_values, minwork):
            v = steps_eval(steps_values, key[2] * key[3] * key[4])
            if v < 0:
                return threads_by_work(key[2], key[3], key[4], minwork, pool)
            return pool if v == 0 or v > pool else v

        rel_t, rel_c = [], []
        lines = []
        for key, r in probs:
            best = max((c for c in r["perf"] if c != "current"),
                       key=lambda c: r["perf"][c])
            nw = threads_of(key, text, w)
            pw = r["perf"]["t%d" % nw]
            pc = r["perf"]["current"]
            rel_t.append(pw / best_of(r))
            rel_c.append(pc / best_of(r))
            lines.append(["%d x %d x %d" % key[2:5], best[1:],
                          "%.1f" % r["perf"][best], r["threads"]["current"],
                          "%.1f" % pc, nw, "%.1f" % pw, "%.2f" % (pw / pc)])
        report += ["", "Type %s: gemmMinWork %d (build: %d), gemmThreads "
                   "{%s} (build: {%s}); geometric mean relative to the best "
                   "number of threads %.3f (build: %.3f)."
                   % (t, w, cur, text, cur_steps, geomean(rel_t),
                      geomean(rel_c)), ""]
        report += table(["m x n x k", "best threads", "GFLOP/s",
                         "threads (build)", "GFLOP/s", "threads (tuned)",
                         "GFLOP/s", "tuned / build"], lines)
    return values, report


#
#  gemm-partition: gemmRatioM, gemmRatioN, gemmMaxJR for all types
#
def fit_gemm_partition(rows, param):
    names = sorted({s for r in rows.values() for s in r["perf"]
                    if s != "current"})
    score, worst = {}, {}
    for c in names:
        rel, vs = [], []
        for r in rows.values():
            rel.append(r["perf"][c] / max(r["perf"].values()))
            vs.append(r["perf"][c] / r["perf"]["current"])
        score[c], worst[c] = geomean(rel), min(vs)
    rel = [r["perf"]["current"] / max(r["perf"].values())
           for r in rows.values()]
    score["current"], worst["current"] = geomean(rel), 1.0
    c = choose(names + ["current"], score, worst, "current")
    values = {}
    cur = param["d"]["current"]
    if c != "current":
        a, b, j = (int(x) for x in c[1:].split(":"))
        values = {"gemmRatioM": a, "gemmRatioN": b, "gemmMaxJR": j}
    report = ["", "Partition (all types): %s (build: r%s:%s:%s), geometric "
              "mean relative to the best candidate %.3f (build: %.3f)."
              % (c, cur["gemmRatioM"], cur["gemmRatioN"], cur["gemmMaxJR"],
                 score[c], score["current"]), ""]
    lines = []
    for key, r in rows.items():
        best = max(r["perf"], key=lambda s: r["perf"][s])
        lines.append([key[0], "%d x %d x %d" % key[2:5], best,
                      "%.1f" % r["perf"][best],
                      "%.1f" % r["perf"]["current"],
                      "%.1f" % r["perf"][c],
                      "%.2f" % (r["perf"][c] / r["perf"]["current"])])
    report += table(["type", "m x n x k", "best", "GFLOP/s", "build",
                     "tuned", "tuned / build"], lines)
    return values, report


#
#  trsm-kb, trmm-kb: kb (and trsmInvert) per type
#
def fit_kb(op, rows, param):
    values, report = {}, []
    for t in TYPES:
        probs = [(k, r) for k, r in rows.items() if k[0] == t]
        if not probs:
            continue
        names = sorted({s for _, r in probs for s in r["perf"]
                        if s != "current"},
                       key=lambda s: [int(x) for x in re.findall(r"\d+", s)])
        score, worst = {}, {}
        for c in names + ["current"]:
            rel = [r["perf"][c] / max(r["perf"].values()) for _, r in probs]
            vs = [r["perf"][c] / r["perf"]["current"] for _, r in probs]
            score[c], worst[c] = geomean(rel), min(vs)
        c = choose(names + ["current"], score, worst, "current")
        cur = param[t]["current"]
        curname = "kb{%s}" % cur[op + ".kb"]
        if op == "trsm":
            curname += "-inv%s" % cur["trsmInvert"]
        if c != "current":
            x = [int(v) for v in re.findall(r"\d+", c)]
            #  the measured kb up to the cell of the largest m measured,
            #  the build's above (larger m: not measured)
            big = max(key[2] for key, _ in probs)
            above = steps_eval(cur[op + ".kb"], 1 << 40)
            kb = [x[0]] if x[0] == above else [x[0], int(big * 1.5), above]
            values["%s.kb.%s" % (op, t)] = kb
            if op == "trsm":
                values["trsmInvert." + t] = x[1]
        report += ["", "Type %s: %s (build: %s), geometric mean relative to "
                   "the best candidate %.3f (build: %.3f)."
                   % (t, c, curname, score[c], score["current"]), ""]
        lines = []
        for key, r in probs:
            best = max(r["perf"], key=lambda s: r["perf"][s])
            lines.append(["%d x %d" % key[2:4],
                          "1" if key[5] == 1 else "all", best,
                          "%.1f" % r["perf"][best],
                          "%.1f" % r["perf"]["current"],
                          "%.1f" % r["perf"][c],
                          "%.2f" % (r["perf"][c] / r["perf"]["current"])])
        report += table(["m x n", "threads", "best", "GFLOP/s", "build",
                         "tuned", "tuned / build"], lines)
    return values, report


#
#  trsm-variant, trmm-variant: sharedBlocks, sharedPanels, minPanels per
#  type (the rule of TrShared::useShared)
#
def fit_variant(op, rows, param):
    values, report = {}, []
    for t in TYPES:
        probs = [(k, r) for k, r in rows.items() if k[0] == t]
        if not probs:
            continue
        cur = param[t]["current"]
        kbs = cur[op + ".kb"]

        def decide(key, r, B, P, Q):
            MR, NR, KC1 = r["info"]
            nt = r["threads"]["shared"]
            if nt <= 1:
                return "current"
            KBmax = min(KC1, 2048 // MR * MR)
            kb = steps_eval(kbs, key[2])
            KB = max(MR, min((kb + MR - 1) // MR * MR, KBmax))
            np_ = (key[3] + NR - 1) // NR
            return "shared" if key[2] > B * KB and np_ < P * nt else "c%d" % Q

        curv = (int(cur[op + ".sharedBlocks"]), int(cur[op + ".sharedPanels"]),
                int(cur[op + ".minPanels"]))
        cands = [(B, P, Q) for B in SHARED_BLOCKS for P in SHARED_PANELS
                 for Q in MIN_PANELS] + [curv]
        score, worst = {}, {}
        for v in cands:
            rel, vs = [], []
            for key, r in probs:
                p = r["perf"][decide(key, r, *v)]
                rel.append(p / max(r["perf"].values()))
                vs.append(p / r["perf"]["current"])
            score[v], worst[v] = geomean(rel), min(vs)
        v = choose(sorted(set(cands)), score, worst, curv)
        if v != curv:
            for name, x in zip(("sharedBlocks", "sharedPanels", "minPanels"),
                               v):
                values["%s.%s.%s" % (op, name, t)] = x

        def show(x):
            return "never" if x >= 1 << 30 else str(x)

        report += ["", "Type %s: shared if m > %s KB and fewer than %s panels "
                   "per thread, otherwise at least %d panels per thread "
                   "(build: %s, %s, %d); geometric mean relative to the best "
                   "variant %.3f (build: %.3f)."
                   % (t, show(v[0]), show(v[1]), v[2], show(curv[0]),
                      show(curv[1]), curv[2], score[v], score[curv]), ""]
        lines = []
        for key, r in probs:
            best = max(r["perf"], key=lambda s: r["perf"][s])
            d = decide(key, r, *v)
            lines.append(["%d x %d" % key[2:4], r["threads"]["shared"], best,
                          "%.1f" % r["perf"][best],
                          "%.1f" % r["perf"]["current"], d,
                          "%.1f" % r["perf"][d],
                          "%.2f" % (r["perf"][d] / r["perf"]["current"])])
        report += table(["m x n", "threads", "best", "GFLOP/s", "build",
                         "tuned", "GFLOP/s", "tuned / build"], lines)
    return values, report


#
#  level2: gemvNThreads, gemvTThreads (gemv with A column major and row
#  major) and gerThreads per type: a decision by m n
#
def fit_steps(probs, pool, current_text):
    """probs: [(size, [r of the ops])]: the steps as v0,b0,v1,... and the
    choice per size"""
    #  numbers of threads that do not decrease with the size (dynamic
    #  programming), the build's decision unless they gain GAIN in the
    #  geometric mean
    counts = sorted({int(c[1:]) for _, rs in probs for r in rs
                     for c in r["perf"] if c.startswith("t")})
    counts = [n for n in counts
              if all("t%d" % n in r["perf"] for _, rs in probs for r in rs)]
    gain = [[sum(math.log(r["perf"]["t%d" % n] / max(r["perf"].values()))
                 for r in rs) for n in counts] for _, rs in probs]
    dp, arg = [gain[0][:]], [[None] * len(counts)]
    for i in range(1, len(probs)):
        row, back = [], []
        for jj in range(len(counts)):
            k = max(range(jj + 1), key=lambda q: dp[i - 1][q])
            row.append(dp[i - 1][k] + gain[i][jj])
            back.append(k)
        dp.append(row)
        arg.append(back)
    jj = max(range(len(counts)), key=lambda q: dp[-1][q])
    total = dp[-1][jj]
    choice = [None] * len(probs)
    for i in range(len(probs) - 1, -1, -1):
        choice[i] = "t%d" % counts[jj]
        jj = arg[i][jj] if i > 0 else jj
    n = sum(len(rs) for _, rs in probs)
    cur = sum(math.log(r["perf"]["current"] / max(r["perf"].values()))
              for _, rs in probs for r in rs)
    if total < cur + n * math.log(1 + GAIN):
        choice = []
        for s, _ in probs:
            v = steps_eval(current_text, s * s)
            choice.append("t%d" % (pool if v <= 0 or v > pool else v))
        return current_text, choice
    #  merge equal neighbours; bounds between the work of neighbouring sizes
    sizes = [s for s, _ in probs]
    steps = []
    for i, c in enumerate(choice):
        nt = int(c[1:])
        v = 0 if nt >= pool else nt
        if steps and steps[-1][1] == v:
            continue
        b = 0 if i == 0 else int(round(math.sqrt(sizes[i - 1] ** 2 *
                                                 sizes[i] ** 2)))
        steps.append((b, v))
    while len(steps) > 9:
        #  the shortest step joins its left neighbour
        i = min(range(1, len(steps)),
                key=lambda j: (steps[j + 1][0] if j + 1 < len(steps)
                               else 1 << 62) / steps[j][0])
        del steps[i]
    text = [str(steps[0][1])]
    for b, v in steps[1:]:
        text += [str(b), str(v)]
    return ",".join(text), choice


def fit_level2(rows, param, pool):
    values, report = {}, []
    for t in TYPES:
        for name, ops in (("gemvNThreads", ("gemv-n",)),
                          ("gemvTThreads", ("gemv-t",)),
                          ("gerThreads", ("ger",))):
            by = collections.OrderedDict()
            for key, r in rows.items():
                if key[0] == t and key[1] in ops:
                    by.setdefault(key[2], []).append(r)
            if not by:
                continue
            probs = sorted(by.items())
            cur = param[t]["current"][name]
            text, choice = fit_steps(probs, pool, cur)
            if text != cur:
                values["%s.%s" % (name, t)] = [int(x) for x in text.split(",")]
            report += ["", "Type %s, %s: {%s} (build: {%s})."
                       % (t, name, text, cur), ""]
            lines = []
            for (s, rs), c in zip(probs, choice):
                for r, op in zip(rs, ops):
                    best = max(r["perf"], key=lambda x: r["perf"][x])
                    lines.append([op, s, best, "%.1f" % r["perf"][best],
                                  "%.1f" % r["perf"]["current"], c,
                                  "%.1f" % r["perf"][c],
                                  "%.2f" % (r["perf"][c] /
                                            r["perf"]["current"])])
            report += table(["op", "m = n", "best", "GFLOP/s", "build",
                             "tuned", "GFLOP/s", "tuned / build"], lines)
    return values, report


#
#  potrf (OpenMP): potrfSerialN and the tile sizes potrfNb by n, for all
#  types
#
def fit_potrf(rows, param):
    by = collections.OrderedDict()
    for key, r in rows.items():
        by.setdefault(key[2], []).append((key[0], r))
    sizes = sorted(by)
    cur = param["d"]["current"]
    cur_serial = int(cur["potrfSerialN"])
    choice = []
    for s in sizes:
        rs = [r for _, r in by[s]]
        names = [c for c in rs[0]["perf"] if c != "current" and
                 all(c in r["perf"] for r in rs)]

        def score(c):
            return geomean(r["perf"][c] / max(r["perf"].values())
                           for r in rs)

        best = max(names, key=score)
        cur_score = geomean(r["perf"]["current"] / max(r["perf"].values())
                            for r in rs)
        if cur_score >= score(best) * (1 - KEEP):
            nb = steps_eval(cur["potrfNb"], s)
            c = "serial" if s <= cur_serial or s <= nb else "nb%d" % nb
            choice.append(c if c in names else best)
        else:
            choice.append(best)
    #  serial for the sizes up to the first one with tasks
    i = 0
    while i < len(sizes) and choice[i] == "serial":
        i += 1
    if i == 0:
        serial_n = min(cur_serial, sizes[0] - 1)
    elif i == len(sizes):
        serial_n = sizes[-1]
    else:
        serial_n = int(math.sqrt(sizes[i - 1] * sizes[i]))
    #  tile sizes of the other sizes (a serial choice among them: the tile
    #  size of the next size with tasks)
    nbs = [None] * len(sizes)
    nxt = None
    for j in range(len(sizes) - 1, i - 1, -1):
        if choice[j] != "serial":
            nxt = int(choice[j][2:])
        nbs[j] = nxt
    steps = []
    for j in range(i, len(sizes)):
        v = nbs[j] if nbs[j] is not None else 128
        if steps and steps[-1][1] == v:
            continue
        b = 0 if not steps else int(math.sqrt(sizes[j - 1] * sizes[j]))
        steps.append((b, v))
    if not steps:
        text = cur["potrfNb"]
    else:
        text = ",".join([str(steps[0][1])] +
                        ["%d,%d" % (b, v) for b, v in steps[1:]])
    values = {}
    if text != cur["potrfNb"]:
        values["potrfNb"] = [int(x) for x in text.split(",")]
    if serial_n != cur_serial:
        values["potrfSerialN"] = serial_n
    report = ["", "All types: serial for n <= %d, tiles {%s} (build: n <= "
              "%d, {%s})." % (serial_n, text, cur_serial, cur["potrfNb"]),
              ""]
    lines = []
    for s, c in zip(sizes, choice):
        eff = "serial" if s <= serial_n else "nb%d" % steps_eval(text, s)
        for t, r in by[s]:
            best = max(r["perf"], key=lambda x: r["perf"][x])
            p = r["perf"].get(eff, r["perf"][c])
            lines.append([t, s, best, "%.1f" % r["perf"][best],
                          "%.1f" % r["perf"]["current"], eff, "%.1f" % p,
                          "%.2f" % (p / r["perf"]["current"])])
    report += table(["type", "n", "best", "GFLOP/s", "build", "tuned",
                     "GFLOP/s", "tuned / build"], lines)
    return values, report


#
#  params.json and the report
#
def load_json(results):
    path = os.path.join(results, "params.json")
    if os.path.exists(path):
        return json.load(open(path))
    return {"values": {}, "groups": {}}


def write_report(results, platform, state):
    out = ["# Parameter tuning: %s" % platform, "",
           "Generated by `make tune-params` (tune/params/params.py).  "
           "Performance in GFLOP/s (10^9 floating-point operations per "
           "second), median over the samples and passes; *build*: the "
           "parameters of the build before (the defaults written by hand "
           "or an earlier make tune), *tuned*: the fitted ones.", ""]
    order = ["gemm-threads", "gemm-partition", "trsm-kb", "trmm-kb",
             "trsm-variant", "trmm-variant", "level2", "potrf", "verify"]
    for g in sorted(state["groups"], key=lambda g: (order.index(g)
                                                    if g in order else 99)):
        info = state["groups"][g]
        out += ["## %s (%s)" % (g, info["date"])] + info["report"] + [""]
    open(os.path.join(results, "params.md"), "w").write("\n".join(out))


def fit(group, platform, data, param_path, results):
    rows = load(data)
    param = load_param(param_path)
    state = load_json(results)
    pool = state.get("pool", 1)
    if group == "gemm-threads":
        values, report = fit_gemm_threads(rows, param, pool)
    elif group == "gemm-partition":
        values, report = fit_gemm_partition(rows, param)
    elif group in ("trsm-kb", "trmm-kb"):
        values, report = fit_kb(group[:4], rows, param)
    elif group in ("trsm-variant", "trmm-variant"):
        values, report = fit_variant(group[:4], rows, param)
    elif group == "level2":
        values, report = fit_level2(rows, param, pool)
    elif group == "potrf":
        values, report = fit_potrf(rows, param)
    else:
        sys.exit("params.py: unknown group " + group)
    #  the values of the group replace those of an earlier fit
    prefixes = {"gemm-threads": ["gemmMinWork.", "gemmThreads."],
                "gemm-partition": ["gemmRatio", "gemmMaxJR"],
                "trsm-kb": ["trsm.kb.", "trsmInvert."],
                "trmm-kb": ["trmm.kb."],
                "trsm-variant": ["trsm.shared", "trsm.minPanels"],
                "trmm-variant": ["trmm.shared", "trmm.minPanels"],
                "level2": ["gemvNThreads.", "gemvTThreads.",
                           "gerThreads."],
                "potrf": ["potrfNb", "potrfSerialN"]}[group]
    measured = {k for k in values} | {
        k for k in state["values"]
        if any(k.startswith(p) for p in prefixes) and
        (len(k.split(".")) < 2 or k.split(".")[-1] in
         {key[0] for key in rows})}
    for k in measured:
        state["values"].pop(k, None)
    state["values"].update(values)
    state["groups"][group] = {"date": str(datetime.date.today()),
                              "report": report}
    json.dump(state, open(os.path.join(results, "params.json"), "w"),
              indent=1, sort_keys=True)
    write_report(results, platform, state)
    for k, v in sorted(values.items()):
        print("tune-params: %s = %s" % (k, v))
    if not values:
        print("tune-params: %s: the parameters of the build stay" % group)


def merge(platform, results, others):
    """adds the values and reports of other runs (params.json of other
    machines of the platform, e.g. one per type) to RESULTS/params.json"""
    state = load_json(results)
    for path in others:
        other = json.load(open(path))
        state["values"].update(other["values"])
        for g, info in other["groups"].items():
            if g in state["groups"]:
                mine = state["groups"][g]
                mine["report"] = mine["report"] + info["report"]
                mine["date"] = max(mine["date"], info["date"])
            else:
                state["groups"][g] = info
        for k in ("pool", "condition"):
            if k in other and k not in state:
                state[k] = other[k]
    json.dump(state, open(os.path.join(results, "params.json"), "w"),
              indent=1, sort_keys=True)
    write_report(results, platform, state)


def verify(platform, data, results):
    rows = load(data)
    state = load_json(results)
    lines, rel = [], []
    for key, r in rows.items():
        q = r["perf"]["tuned"] / r["perf"]["default"]
        rel.append(q)
        lines.append([key[0], key[1], "%d x %d x %d" % key[2:5],
                      "1" if key[5] == 1 else "all",
                      "%.1f" % r["perf"]["default"],
                      "%.1f" % r["perf"]["tuned"], "%.2f" % q])
    report = ["", "The defaults written by hand (before) and the tuned "
              "parameters (after), measured interleaved in a separate run: "
              "geometric mean of after / before %.3f, minimum %.2f, maximum "
              "%.2f." % (geomean(rel), min(rel), max(rel)), ""]
    report += table(["type", "op", "m x n x k", "threads", "before",
                     "after", "after / before"], lines)
    state["groups"]["verify"] = {"date": str(datetime.date.today()),
                                 "report": report}
    json.dump(state, open(os.path.join(results, "params.json"), "w"),
              indent=1, sort_keys=True)
    write_report(results, platform, state)
    print("tune-params: after / before: geometric mean %.3f, min %.2f"
          % (geomean(rel), min(rel)))


#
#  The header of the platform and the index
#
def cpp_name(platform):
    return re.sub(r"[^A-Za-z0-9]", "_", platform)


def assignment(key, value):
    parts = key.split(".")
    if parts[-1] in TYPE_INDEX:
        lhs = "p.%s[%s]" % (".".join(parts[:-1]), TYPE_INDEX[parts[-1]])
    else:
        lhs = "p." + key
    if isinstance(value, list):
        rhs = "{%s}" % ", ".join(str(v) for v in value)
    else:
        rhs = str(value)
    return "    %s = %s;" % (lhs, rhs)


def generate(platform, condition, results, tuneddir):
    state = load_json(results)
    state["condition"] = condition
    json.dump(state, open(os.path.join(results, "params.json"), "w"),
              indent=1, sort_keys=True)
    name = cpp_name(platform)
    guard = "ULMBLAS_CONFIG_TUNED_%s_H" % name.upper()
    dates = sorted(g["date"] for k, g in state["groups"].items())
    date = dates[-1] if dates else str(datetime.date.today())
    L = ["//",
         "//  Parameters of ulmBLAS for %s (ulmblas/config/tuned.h): measured"
         % platform,
         "//  by make tune-params (tune/params/params.py), last on %s, do not"
         % date,
         "//  edit.  The report: tune/results/%s/params.md." % platform,
         "//  Included by ulmblas/config/tuned/tuned.h (after the definition "
         "of Tuned).",
         "//",
         "#ifndef " + guard,
         "#define %s 1" % guard,
         "",
         "#if %s && !defined(USE_TESTPARAM)" % condition,
         "",
         "#define ULMBLAS_TUNED_%s 1" % name.upper(),
         "",
         "namespace ulmBLAS {",
         "namespace tunedparam {",
         "",
         "inline void",
         "%s(Tuned &p)" % name,
         "{"]
    values = state["values"]
    if not values:
        L.append("    (void)p;")
    L.append("    // clang-format off")
    for k in sorted(values):
        L.append(assignment(k, values[k]))
    L += ["    // clang-format on", "}", "", "} // namespace tunedparam", "} // namespace ulmBLAS", "",
          "#endif // %s && !defined(USE_TESTPARAM)" % condition, "",
          "#endif // " + guard, ""]
    open(os.path.join(tuneddir, platform + ".h"), "w").write("\n".join(L))
    write_index(tuneddir)


def write_index(tuneddir):
    platforms = sorted(f[:-2] for f in os.listdir(tuneddir)
                       if f.endswith(".h") and f != "tuned.h")
    L = ["//",
         "//  Index of the parameters measured by make tune per platform: "
         "generated",
         "//  by make tune (tune/params/params.py), do not edit.",
         "//",
         "//  tunedForPlatform(platform, p) sets the measured parameters of "
         "the",
         "//  platform in p (see ulmblas/config/tuned.h); the others keep "
         "their",
         "//  defaults.",
         "//",
         "#ifndef ULMBLAS_CONFIG_TUNED_TUNED_H",
         "#define ULMBLAS_CONFIG_TUNED_TUNED_H 1",
         "",
         "#include <ulmblas/config/platform.h>"]
    L += ["#include <ulmblas/config/tuned/%s.h>" % p for p in platforms]
    L += ["", "namespace ulmBLAS {", "", "struct Tuned;", "", "inline void",
          "tunedForPlatform(Platform platform, Tuned &p)", "{",
          "    (void)platform;", "    (void)p;"]
    for p in platforms:
        n = cpp_name(p)
        L += ["#if defined(ULMBLAS_TUNED_%s)" % n.upper(),
              "    if (platform == platformFromName(\"%s\")) {" % p,
              "        tunedparam::%s(p);" % n,
              "    }",
              "#endif"]
    L += ["}", "", "} // namespace ulmBLAS", "",
          "#endif // ULMBLAS_CONFIG_TUNED_TUNED_H", ""]
    open(os.path.join(tuneddir, "tuned.h"), "w").write("\n".join(L))


def main(argv):
    if len(argv) == 7 and argv[1] == "fit":
        fit(*argv[2:])
    elif len(argv) == 5 and argv[1] == "verify":
        verify(*argv[2:])
    elif len(argv) == 6 and argv[1] == "generate":
        generate(*argv[2:])
    elif len(argv) >= 4 and argv[1] == "merge":
        merge(argv[2], argv[3], argv[4:])
    elif len(argv) == 4 and argv[1] == "pool":
        state = load_json(argv[2])
        state["pool"] = int(argv[3])
        json.dump(state, open(os.path.join(argv[2], "params.json"), "w"),
                  indent=1, sort_keys=True)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
