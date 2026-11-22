#!/usr/bin/env python3
"""Groups the shares of functions in a profile by what they do in a
factorization of ulmBLAS.  Reads lines with a share and a function from
stdin: the output of xctrace-top.py (macOS), of

    perf report --stdio --no-children --sort symbol      (Linux)

or of callgrind_annotate (valgrind; shares of instructions, not of time).

and prints the share of each group.  The first matching pattern decides.
"""
import re
import sys

GROUPS = [
    ("trsm: solve of the diagonal tiles", r"TrsmFrame"),
    # GCC inlines swapRows into getrf2_impl::rec, which does little else
    ("row interchanges", r"swapRows|laswp|getrf2_impl::rec|std::.*swap|is_move_constructible"),
    ("unblocked base case", r"getf2|potf2|rank1|gerSerial|\bger\b|iamax|crout|rightLooking|::rows<|scal"),
    ("memmove, memset (copies, zeroing)", r"memmove|memcpy|memset"),
    ("gemm: micro kernels and packing", r"ugemm|sme::|kernel|pack|MacroKernel|macroBlock|Frame|gemm|dgemm|_mm|avx|haswell|skylake"),
]

total = {}
other = 0.0
for line in sys.stdin:
    m = re.match(r"\s*(?:[0-9,]+\s+\(\s*)?([0-9.]+)%\)?\s+(?:\[\.\]\s+)?(.*)", line)
    if not m or "PROGRAM TOTALS" in line:
        continue
    share, name = float(m.group(1)), m.group(2)
    # callgrind_annotate: "file:function"
    name = re.sub(r"^(?:\?\?\?|\S+\.(?:h|tcc|cc|c|S)):", "", name)
    for group, pattern in GROUPS:
        if re.search(pattern, name):
            total[group] = total.get(group, 0.0) + share
            break
    else:
        other += share

for group, _ in GROUPS:
    print(f"{total.get(group, 0.0):5.1f} %  {group}")
print(f"{other:5.1f} %  other")
