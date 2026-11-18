"""Candidates of the GEMM tuning (make tune): per candidate set the build
in which it is available, the type of the packed blocks and the frame
algorithms (ulmblas/level3/gemm/frame.h) to be measured.

The C++ type of a candidate is used twice: tune/gemm/measure.cc measures
it, tune/gemm/gentable.py writes it into the generated selection table
(ulmblas/level3/gemm/tables/).

    name        identifier of the candidate (also the name of the plan)
    frame       C++ type of the frame algorithm, B is the backend
    blocks      block sizes (MC, KC, NC, MCmax, KCmax, NCmax)
    variant     the rest of the configuration (backend, macro kernel, ...):
                for a problem smaller than the blocks of two candidates with
                the same variant both run the same code
    features    the properties that the selection rules decide (below)
    alias       true: the same candidate again under other features (it is
                measured once; the features do not matter for it)

The selection rules of a set (as in ATLAS: a fixed structure of the decision
whose thresholds are measured, not a map of the problem space):

    "box":    feature that is true for m >= Mmin and n >= Nmin
    "size":   feature that is true for m n >= MNmin and (k >= Kmin or
              m n >= MNbig)
    "global": features that are the same for all problems
    "small":  feature that is true for m k <= MK1, or m <= M2 and m k <= MK2
              (upper bounds), before the other rules

make tune chooses the thresholds and the global features that maximize the
geometric mean of the performance relative to the best candidate over the
grid (tune/gemm/gentable.py).  The candidates of a set are all combinations
of the features.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "mk"))
import gen_kernel_descriptors as kd



def frame(blocks, macro, interleave=0):
    """GemmFrame with GemmBlocks<MC, KC, NC, MCmax, KCmax, NCmax>"""
    return "GemmFrame<B, GemmBlocks<%s>, %d, %s>" % (
        ", ".join(str(b) for b in blocks), interleave, macro)


def sme_double():
    MR, NR = 32, 16
    candidates = []
    for large, (mc, kc, nc) in ((False, (256, 1024, 1024)),
                                (True, (608, 2240, 3072))):
        for extend in (False, True):
            #  maximum block sizes: the rest of a dimension up to two
            #  panels (MC, NC) or a quarter block (KC) larger is taken whole
            if extend:
                blocks = (mc, kc, nc, mc + 2 * MR, kc + kc // 4, nc + 2 * NR)
            else:
                blocks = (mc, kc, nc, mc, kc, nc)
            for prefetch in (False, True):
                name = "sme-%d-%d-%d%s%s" % (mc, kc, nc,
                                             "-max" if extend else "",
                                             "-prefetch" if prefetch else "")
                macro = "sme::MacroKernel<B, %s>" % ("true" if prefetch
                                                     else "false")
                candidates.append({"name": name,
                                   "frame": frame(blocks, macro),
                                   "blocks": blocks,
                                   "variant": macro,
                                   "features": {"large": large,
                                                "max": extend,
                                                "prefetch": prefetch,
                                                "small": False}})
    #  the kernel for small matrices (ulmblas/level3/ukernel/sme/small.h;
    #  for other types than double the frame algorithm without prefetching),
    #  for m, n <= 16 (below the grid) the NEON kernel (neon/small.h, TinyGemm)
    fallback = frame((256, 1024, 1024, 320, 1280, 1056),
                     "sme::MacroKernel<B, false>")
    for extend in (False, True):
        candidates_small = {"name": "sme-small",
                            "frame": "neon::TinyGemm<sme::SmallGemm<%s>>" % fallback,
                            "blocks": None,
                            "variant": "small",
                            "features": {"large": False, "max": extend,
                                         "prefetch": False, "small": True}}
        if not extend:
            candidates.append(candidates_small)
        else:
            #  the same kernel: max does not matter for it
            alias = dict(candidates_small)
            alias["alias"] = True
            candidates.append(alias)
    return candidates


def skylakex_double():
    """OpenBLAS SKYLAKEX DGEMM kernel (16 x 12, AVX-512F; the avx512 backend
    on x86-64): OpenBLAS's blocks 192/384/8640 or 144/512/8640 (the same
    size of the block of A, longer panels), with and without interleaved
    packing of B (packed panel by panel while the first block of A is
    computed, see gemm/default.h), with and without maximum block sizes"""
    MR, NR = 16, 12
    candidates = []
    for large, (mc, kc, nc) in ((False, (192, 384, 8640)),
                                (True, (144, 512, 8640))):
        for extend in (False, True):
            if extend:
                blocks = (mc, kc, nc, mc + 2 * MR, kc + kc // 4, nc + 2 * NR)
            else:
                blocks = (mc, kc, nc, mc, kc, nc)
            for interleave in (False, True):
                name = "skx-%d-%d-%d%s%s" % (mc, kc, nc,
                                             "-max" if extend else "",
                                             "-interleave" if interleave
                                             else "")
                macro = "MacroKernel<B>"
                candidates.append({"name": name,
                                   "frame": frame(blocks, macro,
                                                  1 if interleave else 0),
                                   "blocks": blocks,
                                   "variant": macro + (" interleave"
                                                       if interleave else ""),
                                   "features": {"large": large,
                                                "max": extend,
                                                "interleave": interleave,
                                                "small": False}})
    #  OpenBLAS's kernels for small matrices (SKYLAKEX, AVX-512F,
    #  ulmblas/level3/ukernel/openblas/small.h), with OpenBLAS's permit
    #  function; otherwise the frame algorithm of OpenBLAS's blocks
    fallback = frame((192, 384, 8640, 192, 384, 8640), "MacroKernel<B>", 1)
    for extend in (False, True):
        c = {"name": "openblas-small",
             "frame": "openblas::SmallGemm<%s>" % fallback,
             "blocks": None,
             "variant": "openblas-small",
             "features": {"large": False, "max": extend,
                          "interleave": False, "small": True}}
        if extend:
            c["alias"] = True
        candidates.append(c)
    #  the kernel for small matrices of ulmBLAS (ulmblas/level3/ukernel/avx/
    #  small.h), measured for comparison with OpenBLAS's (no rule selects
    #  it: its features match none)
    candidates.append({"name": "avx-small",
                       "frame": "avx::SmallGemm<%s>" % fallback,
                       "blocks": None,
                       "variant": "avx-small",
                       "features": {"large": False, "max": False,
                                    "interleave": False, "small": "own"}})
    return candidates


#
#  All vendored micro kernels with FMA of an x86-64 instruction set level
#  (ulmblas/level3/ukernel/kernels.h, KernelBackend): per kernel its default
#  block sizes with maximum sizes (MC + 2 MR, KC + KC/4 (at most KMAX),
#  NC + 2 NR), and the vector kernel for small matrices (avx/small.h); for
#  double with AVX-512 also OpenBLAS's small kernels.  The table: the best
#  candidate per cell of the grid, merged into boxes (rules "boxes").
#
X86_LEVELS = {
    "avx2": {"SSE3", "AVX", "FMA", "AVX2"},
    "avx512": {"SSE3", "AVX", "FMA", "AVX2", "AVX512F", "AVX512DQ",
               "AVX512VL"},
}

#  short names of the types (sets, files of the tables)
TYPE_NAMES = {"float": "float", "double": "double",
              "std::complex<float>": "cfloat",
              "std::complex<double>": "cdouble"}


def x86_kernels(ch, level):
    """the candidates of all kernels for type ch (s, d, c, z) at level"""
    table = {"d": "select.h", "s": "select_s.h", "c": "select_c.h",
             "z": "select_z.h"}[ch]
    allowed = X86_LEVELS[level]
    candidates = []
    for k in kd.parse(os.path.join(kd.BLIS, table)):
        isa = set(re.findall(r"defined\(__(\w+)__\)", k["isa"]))
        #  only kernels of the level, not the older ones (without FMA: never
        #  faster on a CPU with AVX2)
        if not isa <= allowed or "FMA" not in isa:
            continue
        mm = k["macros"]
        mr, nr = int(mm["MR"]), int(mm["NR"])
        mc, kc, nc = int(mm["MC_"]), int(mm["KC_"]), int(mm["NC_"])
        kmax = int(mm.get("KMAX", "0"))
        kcmax = kc + kc // 4
        if kmax > 0:
            kcmax = min(kcmax, kmax)
            kc = min(kc, kcmax)
        blocks = (mc, kc, nc, mc + 2 * mr, kcmax, nc + 2 * nr)
        name = "%s_%s" % (ch, k["id"])
        backend = "KernelBackend<kernels::%s>" % name
        candidates.append({
            "name": name.replace("_", "-"),
            "frame": "GemmFrame<%s, GemmBlocks<%s>, 0, MacroKernel<%s>>" % (
                backend, ", ".join(str(b) for b in blocks), backend),
            "blocks": blocks,
            "variant": name,
            "features": {}})
    return candidates


def x86_small(ch, level):
    """the kernels for small matrices (fallback: frame of the first kernel;
    the vector kernel takes every type, OpenBLAS's only double)"""
    first = x86_kernels(ch, level)[0]
    out = [{"name": "avx-small",
            "frame": "avx::SmallGemm<%s>" % first["frame"],
            "blocks": None, "variant": "avx-small", "features": {}}]
    if ch == "d" and level == "avx512":
        out.append({"name": "openblas-small",
                    "frame": "openblas::SmallGemm<%s>" % first["frame"],
                    "blocks": None, "variant": "openblas-small",
                    "features": {}})
    return out


X86_CONDITIONS = {
    "avx2": "defined(__AVX2__) && defined(__FMA__) && "
            "!defined(__AVX512F__) && !defined(USE_TESTPARAM)",
    "avx512": "defined(__AVX512F__) && defined(__AVX512DQ__) && "
              "defined(__AVX512VL__) && !defined(USE_TESTPARAM)",
}


def x86_set(ch, ctype, level):
    return {
        "condition": X86_CONDITIONS[level] + " && defined(__x86_64__)",
        "type": ctype,
        "backend": "DefaultBackend<%s>" % ctype,
        "includes": ["ulmblas/level3/gemm/macrokernel.h",
                     "ulmblas/level3/ukernel/kernelbackend.h",
                     "ulmblas/level3/ukernel/avx/small.h",
                     "ulmblas/level3/ukernel/openblas/small.h"],
        "candidates": x86_kernels(ch, level) + x86_small(ch, level),
        "rules": {"boxes": True},
    }


#
#  key: (type of the packed blocks, kernel set)
#
#  condition: preprocessor condition under which the candidates exist (the
#  kernels of the set are compiled in); test configurations with their own
#  block sizes (USE_TESTPARAM) never use generated tables.
#
SETS = {
    ("double", "sme"): {
        "condition": "defined(USE_SME) && !defined(USE_TESTPARAM)",
        "type": "double",
        "backend": "DefaultBackend<double>",
        "includes": ["ulmblas/level3/ukernel/sme/macrokernel.h",
                     "ulmblas/level3/ukernel/sme/small.h",
                     "ulmblas/level3/ukernel/neon/small.h"],
        "candidates": sme_double(),
        #  large blocks (608/2240/3072) for large m and n, prefetching of C
        #  for large C and k, maximum block sizes for all or none
        "rules": {"box": "large", "size": "prefetch", "global": ["max"],
                  "small": "small"},
    },
    ("double", "skylakex"): {
        "condition": "defined(ULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12) "
                     "&& !defined(USE_TESTPARAM)",
        "type": "double",
        "backend": "DefaultBackend<double>",
        "includes": ["ulmblas/level3/ukernel/openblas/small.h",
                     "ulmblas/level3/ukernel/avx/small.h"],
        "candidates": skylakex_double(),
        #  longer panels (144/512) for large m and n, interleaved packing of
        #  B by the size of C and k, maximum block sizes for all or none
        "rules": {"box": "large", "size": "interleave", "global": ["max"],
                  "small": "small"},
    },
}

for _level in ("avx2", "avx512"):
    for _ch, _ctype in (("d", "double"), ("s", "float"),
                        ("c", "std::complex<float>"),
                        ("z", "std::complex<double>")):
        SETS[(TYPE_NAMES[_ctype], "x86-" + _level)] = x86_set(_ch, _ctype,
                                                              _level)
