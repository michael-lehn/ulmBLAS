#!/usr/bin/env python3
"""Generates ulmblas/level3/ukernel/kernels.h: one descriptor per vendored
micro kernel of GEMM (BLIS, AOCL-BLAS, OpenBLAS), from the selection tables
ulmblas/level3/ukernel/blis/select{,_s,_c,_z}.h, so that every kernel the
build's instruction set allows is compiled in and the selection tables of
gemm (ulmblas/level3/gemm/select.h, make tune) can choose among them.

    python3 mk/gen_kernel_descriptors.py        (in the source root)

The selection tables remain the source: a kernel is added there (as for
the build-time selection with -DULMBLAS_?GEMM_UKERNEL_...) and this script
is run again.  The output is formatted with clang-format if available.
"""

import os
import re
import subprocess
import sys

TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLIS = os.path.join(TOP, "ulmblas", "level3", "ukernel", "blis")
OUT = os.path.join(TOP, "ulmblas", "level3", "ukernel", "kernels.h")

TABLES = [("d", "double", "select.h"), ("s", "float", "select_s.h"),
          ("c", "std::complex<float>", "select_c.h"),
          ("z", "std::complex<double>", "select_z.h")]


def logical_lines(path):
    """the lines of a file with continuation lines joined"""
    out, cur = [], ""
    for line in open(path).read().split("\n"):
        if line.endswith("\\"):
            cur += line[:-1] + " "
            continue
        out.append(cur + line)
        cur = ""
    return out


def parse(path):
    """the kernels of a selection table: id, macros, ISA condition"""
    kernels, cur, win = [], None, None
    for line in logical_lines(path):
        m = re.match(r"^#(?:el)?if defined\(ULMBLAS_[SDCZ]GEMM_UKERNEL_(\w+)\)",
                     line)
        if m:
            cur = {"id": m.group(1), "macros": {}, "isa": "1"}
            kernels.append(cur)
            continue
        if cur is None:
            continue
        if re.match(r"^#(else|endif|elif|if)\b", line):
            cur = None
            continue
        #  a value for Windows (#if defined(_WIN32) ... #else ... #endif)
        if re.match(r"^#\s+if\s+defined\(_WIN32\)", line):
            win = True
            continue
        if win is not None and re.match(r"^#\s+else\b", line):
            win = False
            continue
        if win is not None and re.match(r"^#\s+endif\b", line):
            win = None
            continue
        m = re.match(r"^#\s+define\s+ULMBLAS_BLIS_[SDCZ]GEMM_(\w+?)\s+(.*?)"
                     r"\s*(//.*)?$", line)
        if m:
            key = m.group(1) + ("_WIN32" if win else "")
            cur["macros"][key] = m.group(2)
            continue
        m = re.match(r"^#\s+if\s+!\s*(.*?)\s*$", line)
        if m:
            cond = re.sub(r"\s+", " ", m.group(1))
            if cond.startswith("(") and cond.endswith(")"):
                cond = cond[1:-1].strip()
            cur["isa"] = cond
    return kernels


def descriptor(ch, ctype, k):
    mm = k["macros"]
    ident = "%s_%s" % (ch, k["id"])
    ns = mm.get("NAMESPACE", "blis::kernel::" + mm["NAME"])
    flag = lambda key: "true" if mm.get(key, "0") not in ("0", "") else "false"
    kmax = "    static constexpr std::ptrdiff_t KMAX = %s;" % mm.get("KMAX", "0")
    if "KMAX_WIN32" in mm:
        kmax = ("#if defined(_WIN32)\n"
                "    static constexpr std::ptrdiff_t KMAX = %s;\n"
                "#else\n%s\n#endif" % (mm["KMAX_WIN32"], kmax))
    panels = mm.get("INTERLEAVE_PANELS", "1")
    return """
#if (%(isa)s) && defined(__x86_64__) && !defined(__ILP32__)
#include %(header)s
#define ULMBLAS_KERNEL_%(ident)s 1
namespace ulmBLAS {
namespace kernels {

struct %(ident)s
{
    typedef %(ctype)s T;
    static constexpr std::ptrdiff_t MR = %(mr)s, NR = %(nr)s;
    //  default block sizes (of the selection table), alignment of the
    //  packed panels in bytes, maximal kc (0: any), panels of A packed
    //  together
    static constexpr std::ptrdiff_t MC = %(mc)s, KC = %(kc)s, NC = %(nc)s;
    static constexpr std::ptrdiff_t ALIGN = %(align)s;
%(kmax)s
    static constexpr std::ptrdiff_t PANELS = %(panels)s;
    //  calling convention and storage of C (ulmblas/level3/ukernel/blis/
    //  xugemm.h)
    static constexpr int API = %(api)s, STORAGE = %(storage)s;
    static constexpr bool rowpref = %(rowpref)s, blockcall = %(blockcall)s;

    static const char *
    name()
    {
        return "%(ident)s";
    }

    template <typename... Args>
    static void
    call(Args... args)
    {
        %(ns)s::%(function)s(args...);
    }
};

} // namespace kernels
} // namespace ulmBLAS
#define ULMBLAS_KERNEL_IF_%(ident)s(F) F(%(ident)s)
#else
#define ULMBLAS_KERNEL_IF_%(ident)s(F)
#endif
""" % {"isa": k["isa"], "header": mm["HEADER"], "ident": ident,
       "ctype": ctype, "mr": mm["MR"], "nr": mm["NR"], "mc": mm["MC_"],
       "kc": mm["KC_"], "nc": mm["NC_"], "align": mm.get("ALIGN", "64"),
       "kmax": kmax, "panels": panels, "api": mm["API"],
       "storage": mm["STORAGE"], "rowpref": flag("ROWPREF"),
       "blockcall": flag("BLOCKCALL"), "ns": ns,
       "function": mm["FUNCTION"]}


def main():
    parts = ["""#ifndef ULMBLAS_LEVEL3_UKERNEL_KERNELS_H
#define ULMBLAS_LEVEL3_UKERNEL_KERNELS_H 1

//
//  Descriptors of the vendored micro kernels of GEMM (BLIS, AOCL-BLAS,
//  OpenBLAS): one struct per kernel in namespace ulmBLAS::kernels, defined
//  if the build's instruction set allows the kernel (then also
//  ULMBLAS_KERNEL_<name>).  Generated by mk/gen_kernel_descriptors.py from
//  ulmblas/level3/ukernel/blis/select{,_s,_c,_z}.h, do not edit.
//
//  A descriptor is the parameter of ulmBLAS::KernelBackend
//  (ulmblas/level3/ukernel/kernelbackend.h): with it the frame algorithm of
//  gemm runs with that kernel, independent of the kernel the build selects
//  for its default (-DULMBLAS_?GEMM_UKERNEL_...).
//

#include <complex>
#include <cstddef>
#include <ulmblas/level3/ukernel/blis/compat.h>
#include <ulmblas/level3/ukernel/blis/compat_end.h>
"""]
    names = []
    lists = []
    for ch, ctype, table in TABLES:
        ids = []
        for k in parse(os.path.join(BLIS, table)):
            parts.append(descriptor(ch, ctype, k))
            ids.append("%s_%s" % (ch, k["id"]))
        names += ids
        lists.append("""
//  ULMBLAS_KERNELS_%s(F): F(name) for every available kernel for %s
#define ULMBLAS_KERNELS_%s(F) %s
""" % (ch, ctype, ch, " ".join("ULMBLAS_KERNEL_IF_%s(F)" % i for i in ids)))
    parts += lists
    parts.append("""
#endif // ULMBLAS_LEVEL3_UKERNEL_KERNELS_H
""")
    open(OUT, "w").write("".join(parts))
    try:
        subprocess.run(["clang-format", "-i", OUT], check=True)
    except (OSError, subprocess.CalledProcessError):
        print("gen_kernel_descriptors.py: clang-format failed", file=sys.stderr)
    print("%s: %d kernels" % (os.path.relpath(OUT, TOP), len(names)))


if __name__ == "__main__":
    main()
