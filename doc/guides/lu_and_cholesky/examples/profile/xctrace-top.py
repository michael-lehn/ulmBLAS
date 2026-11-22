#!/usr/bin/env python3
"""Shares of the functions at the top of the stack in a Time Profiler
recording of xctrace (macOS).  Usage:

    xcrun xctrace export --input factor.trace --xpath \\
        '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \\
        > tp.xml
    python3 xctrace-top.py tp.xml | python3 groups.py

Each sample counts for the innermost function of its backtrace.  The XML
refers to repeated elements by id (attribute ref), which is resolved here.
"""
import collections
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
byid = {e.get("id"): e for e in root.iter() if e.get("id")}


def resolve(e):
    return byid[e.get("ref")] if e.get("ref") else e


count = collections.Counter()
for row in root.iter("row"):
    bt = row.find("tagged-backtrace")
    if bt is None:
        bt = row.find("backtrace")
    if bt is None:
        continue
    bt = resolve(bt)
    frame = bt.find("frame")
    if frame is None:
        continue
    count[resolve(frame).get("name", "?")] += 1

total = sum(count.values())
for name, n in count.most_common():
    print(f"{100.0 * n / total:6.2f}%  {name}")
