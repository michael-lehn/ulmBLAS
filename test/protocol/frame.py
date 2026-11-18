#!/usr/bin/env python3
#
#  Exhaustive check of the synchronization of the parallel gemm frame
#  (GemmFrame::runParallel, ulmblas/level3/gemm/frame.h): all interleavings
#  of the threads of a group, for small numbers of threads (P), parts per
#  share (D; the frame has D = 1, D > 1 is OpenBLAS's DIVIDE_RATE) and
#  blocks (S), and every combination of threads with and without rows of C.
#
#  The model has sequentially consistent counters; the C++ memory model
#  (release stores, acquire loads of the counters) is argued in frame.h.
#  Thread u in block s (buffer s % 2, Q = P D shares, share c packed by
#  thread c / D):
#
#      free   s >= 2: wait until done[v] >= s - 1 for all v
#      no rows: pack the own shares (write, ready[c] = s + 1)
#      rows:  for t = 0, ..., Q-1, c = (u D + t) % Q:
#                 t < D: pack c, read c
#                 else:  wait until ready[c] >= s + 1, read c
#      done   done[u] = s + 1
#
#  Checked: a read of share c in block s finds block s in the buffer (no
#  read before the write, no overwrite before the read), and every state
#  without an enabled step is final (no deadlock).  The rule before the
#  done counters (a sum of uses per share) must fail: a self-test of the
#  check.
#
#      python3 test/protocol/frame.py          quick (a few seconds)
#      python3 test/protocol/frame.py --full   P <= 4, D <= 3, S <= 6
#
import itertools
import sys
from collections import deque


def program(u, P, D, S, rows):
    Q = P * D
    steps = []
    for s in range(S):
        steps.append(('free', s))
        if not rows[u]:
            for c in range(u * D, (u + 1) * D):
                steps.append(('pack', s, c))
        else:
            for t in range(Q):
                c = (u * D + t) % Q
                steps.append(('pack', s, c) if t < D else ('wait', s, c))
                steps.append(('read', s, c))
        steps.append(('done', s))
    return steps


#  rule 'done': counters of finished blocks per thread (frame.h); rule
#  'sum': uses per share, P per block (the rule before)
def check(P, D, S, rows, rule='done'):
    Q = P * D
    progs = [program(u, P, D, S, rows) for u in range(P)]
    counters = P if rule == 'done' else Q
    init = ((0,) * P, (0,) * Q, (0,) * counters, (-1,) * (2 * Q))
    seen = {init}
    queue = deque([init])
    while queue:
        pcs, ready, count, buf = queue.popleft()
        enabled = False
        for u in range(P):
            if pcs[u] == len(progs[u]):
                continue
            step = progs[u][pcs[u]]
            kind, s = step[0], step[1]
            r, k, b = list(ready), list(count), list(buf)
            if kind == 'free':
                if s >= 2:
                    if rule == 'done':
                        if any(k[v] < s - 1 for v in range(P)):
                            continue
                    elif any(k[c] < (s - 1) * P
                             for c in range(u * D, (u + 1) * D)):
                        continue
            elif kind == 'pack':
                c = step[2]
                b[(s % 2) * Q + c] = s
                r[c] = s + 1
            elif kind == 'wait':
                if r[step[2]] < s + 1:
                    continue
            elif kind == 'read':
                c = step[2]
                if b[(s % 2) * Q + c] != s:
                    return ('thread %d reads share %d of block %d, the '
                            'buffer holds block %d'
                            % (u, c, s, b[(s % 2) * Q + c]))
            elif kind == 'done':
                if rule == 'done':
                    k[u] = s + 1
                else:
                    for c in range(Q):
                        k[c] += 1
            enabled = True
            next_pcs = list(pcs)
            next_pcs[u] += 1
            state = (tuple(next_pcs), tuple(r), tuple(k), tuple(b))
            if state not in seen:
                seen.add(state)
                queue.append(state)
        if not enabled and any(pcs[u] < len(progs[u]) for u in range(P)):
            return 'deadlock in state %s' % (pcs,)
    return None


def cases(full):
    for P in (2, 3, 4) if full else (2, 3):
        for D in (1, 2, 3) if full else (1, 2):
            for S in range(2, 7 if full else 5):
                if P * D * S > (36 if full else 24):
                    continue
                for rows in itertools.product((False, True), repeat=P):
                    yield P, D, S, rows


def name(P, D, S, rows):
    return 'P=%d D=%d S=%d rows=%s' % (
        P, D, S, ''.join('r' if x else '-' for x in rows))


def main():
    full = '--full' in sys.argv[1:]
    all_cases = list(cases(full))
    failures = 0
    for case in all_cases:
        error = check(*case)
        if error:
            failures += 1
            print('FAIL %s: %s' % (name(*case), error))
    print('done counters: %d cases, %d failures'
          % (len(all_cases), failures))
    found = [case for case in all_cases if check(*case, rule='sum')]
    print('sum of uses (self-test, must fail): %d of %d cases fail'
          % (len(found), len(all_cases)))
    return 1 if failures or not found else 0


if __name__ == '__main__':
    sys.exit(main())
