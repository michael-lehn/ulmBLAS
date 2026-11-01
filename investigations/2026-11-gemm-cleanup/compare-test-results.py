#!/usr/bin/env python3
"""Compare two test/build/check.log files test by test.

    compare-test-results.py BEFORE.log AFTER.log

Every result of BEFORE is identified by (configuration, test ID).  The output
says what became of it in AFTER, and lists results that are new in AFTER.  The
exit status is 1 if a result of BEFORE has disappeared or a test fails.
"""
import collections
import re
import sys

LINE = re.compile(r'^(\w+)\s+(T[\d-]+) \[([^\]]+)\] (.*)$')
REMOVED = 'micro kernel temporarily removed'


def load(path):
    results = {}
    with open(path, encoding='utf-8', errors='replace') as log:
        for line in log:
            match = LINE.match(line.rstrip('\n'))
            if not match:
                continue
            status, test, config, rest = match.groups()
            if status == 'SKIP':
                # the reason follows the last ': '
                reason = rest.rsplit(': ', 1)[-1]
                if REMOVED in reason:
                    status = 'SKIP(kernel removed)'
                elif 'CBLAS test programs' in reason:
                    status = 'SKIP(64-bit index)'
                elif 'tools not usable' in reason:
                    status = 'SKIP(no Fortran)'
                else:
                    status = 'SKIP(CPU or compiler)'
            results[(config, test)] = status
    return results


def table(title, counter):
    print(title)
    for key in sorted(counter):
        print(f'  {counter[key]:6d}  {key}')
    print()


before = load(sys.argv[1])
after = load(sys.argv[2])

ids_before = {test for _, test in before}
ids_after = {test for _, test in after}
transitions = collections.Counter()
by_config = collections.defaultdict(collections.Counter)
for key, status in before.items():
    new = after.get(key, 'DISAPPEARED')
    transitions[f'{status} -> {new}'] += 1
    by_config[key[0]][f'{status} -> {new}'] += 1
new_results = collections.Counter()
for key, status in after.items():
    if key not in before:
        new_results[f'{key[0]}: {status}'] += 1

print(f'results: before {len(before)}, after {len(after)}')
print(f'test IDs: before {len(ids_before)}, after {len(ids_after)}, '
      f'missing {sorted(ids_before - ids_after)}, '
      f'new {sorted(ids_after - ids_before)}')
print()
table('(configuration, test) of BEFORE, status before -> after:', transitions)
print('per configuration:')
for config in sorted(by_config):
    parts = ', '.join(f'{count} {key}'
                      for key, count in sorted(by_config[config].items()))
    print(f'  {config}: {parts}')
print()
if new_results:
    table('results only in AFTER:', new_results)

gone = sum(n for key, n in transitions.items() if key.endswith('DISAPPEARED'))
failed = sum(1 for status in after.values() if status == 'FAIL')
print(f'disappeared: {gone}, failures after: {failed}')
sys.exit(1 if gone or failed else 0)
