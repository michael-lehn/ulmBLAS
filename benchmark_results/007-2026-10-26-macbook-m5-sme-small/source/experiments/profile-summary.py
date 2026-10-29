#!/usr/bin/env python3
#
#  profile-summary.py FILE... : summary of macOS 'sample' profiles of bench2
#  (sample PID 3 -f FILE while bench2 runs one shape).  Per file: the total
#  number of samples and, inclusive (callees counted), the share of the
#  micro kernel, the packing routines, the buffer management (memory pool
#  or malloc/free with everything they call) and the rest.
#
import re
import sys

GROUPS = [
    ('micro kernel', re.compile(r'ulmBLAS::sme::ugemm')),
    ('pack B', re.compile(r'ulmBLAS::sme::pack_B')),
    ('pack A', re.compile(r'ulmBLAS::sme::pack_A')),
    ('direct path', re.compile(r'ulmBLAS::direct::')),
    ('buffers', re.compile(r'MemoryPool|__hash_table|std::__1::list|operator new'
                           r'|operator delete|\bmalloc\b|\bfree\b|xzm_|_mm_free')),
]
LINE = re.compile(r'^([ +!:|]*?)(\d+) (.*)$')

print('%-28s %7s' % ('profile', 'samples')
      + ''.join('%14s' % g for g, _ in GROUPS) + '%14s' % 'rest')
for path in sys.argv[1:]:
    text = open(path).read()
    graph = text.split('Call graph:', 1)[1].split('Total number in stack', 1)[0]
    total, inside, stack = 0, {g: 0 for g, _ in GROUPS}, []
    for line in graph.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        depth, count = len(m.group(1)), int(m.group(2))
        name = m.group(3).split('  (in ', 1)[0]      # not the binary
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if not stack:
            total = max(total, count)      # the thread
        group = next((g for g, r in GROUPS if r.search(name)), None)
        if group and not any(s[1] for s in stack):
            inside[group] += count
        stack.append((depth, group))
    rest = total - sum(inside.values())
    print('%-28s %7d' % (path.rsplit('/', 1)[-1], total)
          + ''.join('%13.1f%%' % (100.0*inside[g]/total) for g, _ in GROUPS)
          + '%13.1f%%' % (100.0*rest/total))
