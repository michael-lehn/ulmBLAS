#!/usr/bin/env python3
#
# Checks the benchmark archive and the references to it (make
# check_benchmark_results, see benchmark_results/README.md).
#
#     check_benchmark_results.py TOPDIR
#
# 1. Every campaign directory in benchmark_results/ is named
#    NNN-YYYY-MM-DD-machine-experiment; the sequence numbers NNN are unique
#    and contiguous from 001.  The dates (completion of the measurements)
#    need not increase with the number, which follows the order in which
#    the campaigns were archived.
# 2. campaign.conf: NAME equals the directory name, DATE its date.
# 3. The index (benchmark_results/README.md) has one table per data type
#    (### double, complex double, float, complex float, in this order, only
#    the types measured); each lists the campaigns with this type in TYPES
#    of campaign.conf (default double), in the order of the numbers.
# 4. Local links and images in the Markdown files of the repository
#    (tracked by git, or all files if TOPDIR is not a git checkout) point to
#    existing files; references benchmark_results/NAME and names of the
#    form NNN-YYYY-MM-DD-... in other text files name archived campaigns.
#    The raw data (data/) and the programs (source/) of a campaign are
#    historical records and are not checked.
# 5. No campaign is referred to by its name before the numbering
#    (YYYY-MM-DD-machine-experiment) outside data/ and source/, except in
#    the table of former names in benchmark_results/README.md.
#
# Prints every problem and exits with status 1 if there is one.
#

import os
import re
import subprocess
import sys

top = sys.argv[1] if len(sys.argv) > 1 else '.'
archive = os.path.join(top, 'benchmark_results')
problems = []

def problem(msg):
    problems.append(msg)

def get(conf, key):
    for line in open(conf, encoding='utf-8'):
        m = re.match(r'%s\s*=\s*(.*?)\s*$' % key, line)
        if m:
            return m.group(1)
    return None

# 1., 2. names, numbers, dates, campaign.conf
NAME = re.compile(r'^(\d{3})-(\d{4}-\d{2}-\d{2})-[a-z0-9][a-z0-9._-]*$')
campaigns = []
for d in sorted(os.listdir(archive)):
    if d.startswith('.') or not os.path.isdir(os.path.join(archive, d)):
        continue
    m = NAME.match(d)
    if not m:
        problem('benchmark_results/%s: not named NNN-YYYY-MM-DD-machine-experiment' % d)
        continue
    campaigns.append((int(m.group(1)), m.group(2), d))
    conf = os.path.join(archive, d, 'campaign.conf')
    if not os.path.isfile(conf):
        problem('benchmark_results/%s: campaign.conf not found' % d)
        continue
    if get(conf, 'NAME') != d:
        problem('benchmark_results/%s/campaign.conf: NAME %s differs from the directory name'
                % (d, get(conf, 'NAME')))
    if get(conf, 'DATE') != m.group(2):
        problem('benchmark_results/%s/campaign.conf: DATE %s differs from the date of the name'
                % (d, get(conf, 'DATE')))

numbers = [c[0] for c in campaigns]
for n in sorted(set(numbers)):
    if numbers.count(n) > 1:
        problem('sequence number %03d is used %d times: %s'
                % (n, numbers.count(n), ', '.join(c[2] for c in campaigns if c[0] == n)))
if sorted(set(numbers)) != list(range(1, len(set(numbers)) + 1)):
    problem('sequence numbers are not contiguous from 001: %s'
            % ' '.join('%03d' % n for n in sorted(set(numbers))))
names = {c[2] for c in campaigns}

# 3. index
index = open(os.path.join(archive, 'README.md'), encoding='utf-8').read()
m = re.search(r'<!-- BEGIN CAMPAIGNS.*?-->(.*?)<!-- END CAMPAIGNS -->', index, re.S)
if not m:
    problem('benchmark_results/README.md: no index (<!-- BEGIN CAMPAIGNS -->)')
else:
    order = ['double', 'complex double', 'float', 'complex float']
    types = {}
    for c in campaigns:
        conf = os.path.join(archive, c[2], 'campaign.conf')
        value = get(conf, 'TYPES') if os.path.isfile(conf) else None
        ts = [t.strip() for t in (value or 'double').split(',')]
        for t in ts:
            if t not in order:
                problem('benchmark_results/%s/campaign.conf: TYPES: unknown data type %s'
                        % (c[2], t))
        types[c[2]] = ts
    sections = re.split(r'^### (.+)$', m.group(1), flags=re.M)
    headings = sections[1::2]
    expected = [t for t in order if any(t in ts for ts in types.values())]
    if headings != expected:
        problem('benchmark_results/README.md: the index has the sections %s, expected %s'
                % (headings, expected))
    for t, body in zip(headings, sections[2::2]):
        listed = re.findall(r'^\| \[([^\]]+)\]\(([^)]+)/README\.md\)', body, re.M)
        if [l[0] for l in listed] != [c[2] for c in campaigns if t in types[c[2]]]:
            problem('benchmark_results/README.md: the index section %s does not list exactly '
                    'the campaigns of this type in the order of their numbers' % t)
        for text, link in listed:
            if text != link:
                problem('benchmark_results/README.md: index entry %s links to %s' % (text, link))

# 4., 5. references
try:
    files = subprocess.run(['git', '-C', top, 'ls-files'], capture_output=True,
                           text=True, check=True).stdout.split('\n')
except (OSError, subprocess.CalledProcessError):
    files = [os.path.relpath(os.path.join(r, f), top)
             for r, _, fs in os.walk(top) if '/.git' not in r for f in fs]
files = [f for f in files if f and os.path.isfile(os.path.join(top, f))]

def historical(f):
    p = f.split('/')
    return p[0] == 'benchmark_results' and len(p) > 2 and p[2] in ('data', 'source')

LINK = re.compile(r'!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)')
REF = re.compile(r'benchmark_results/([A-Za-z0-9._-]+)')
NUMBERED = re.compile(r'(?<![A-Za-z0-9_-])(\d{3}-\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9._-]*[a-z0-9])')
OLDNAMES = {c[2][4:] for c in campaigns}
OLD = re.compile(r'(?<![A-Za-z0-9_-])(\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9._-]*[a-z0-9])')
TEXT = ('.md', '.h', '.tcc', '.cc', '.c', '.cpp', '.py', '.sh', '.sbatch', '.gp',
        '.conf', '.txt', '.mk', '')

for f in files:
    if historical(f) or os.path.splitext(f)[1] not in TEXT:
        continue
    path = os.path.join(top, f)
    try:
        text = open(path, encoding='utf-8').read()
    except UnicodeDecodeError:
        continue
    if f.endswith('.md'):
        # links outside code blocks and code spans
        prose = re.sub(r'```.*?```', '', text, flags=re.S)
        prose = re.sub(r'`[^`\n]*`', '', prose)
        for target in LINK.findall(prose):
            if re.match(r'[a-z]+:', target) or target.startswith('#'):
                continue
            t = target.split('#')[0]
            if t and not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(path), t))):
                problem('%s: broken link %s' % (f, target))
    for ref in REF.findall(text):
        ref = ref.rstrip('.')
        if ref and ref != 'README.md' and ref not in names and NAME.match(ref):
            problem('%s: benchmark_results/%s is not an archived campaign' % (f, ref))
    for ref in NUMBERED.findall(text):
        if ref not in names and not any(n.startswith(ref + '/') for n in names):
            if NAME.match(ref):
                problem('%s: %s is not an archived campaign' % (f, ref))
    if f == 'benchmark_results/README.md':
        text = re.sub(r'<!-- BEGIN FORMER NAMES -->.*?<!-- END FORMER NAMES -->', '',
                      text, flags=re.S)
    for ref in OLD.findall(text):
        if ref in OLDNAMES:
            problem('%s: former name %s (now %s)'
                    % (f, ref, [c[2] for c in campaigns if c[2][4:] == ref][0]))

for p in problems:
    print(p)
print('benchmark_results: %d campaigns (%03d ... %03d), %d problems'
      % (len(campaigns), min(numbers or [0]), max(numbers or [0]), len(problems)))
sys.exit(1 if problems else 0)
