#!/usr/bin/env python3
#
#  vendor_openblas_kernels.py: generates the OpenBLAS-derived DGEMM micro
#  kernels in ulmblas/level3/ukernel/openblas/kernel/ (C kernels with inline
#  assembly) and ulmblas/level3/ukernel/openblas/asm/ (assembly kernels)
#  from an unmodified OpenBLAS source tree.
#
#      python3 mk/vendor_openblas_kernels.py OPENBLAS_DIR
#
#  OPENBLAS_DIR is an unpacked https://github.com/OpenMathLib/OpenBLAS tag
#  v0.3.34 (commit e0166008be8e466242aa76b2ff75ce3f0fbf574a); it is only
#  read.  For every kernel in KERNELS the script writes kernel/NAME.h:
#
#    1. the license of OpenBLAS (LICENSE of the source tree, unchanged; the
#       kernel file itself has no license comment);
#    2. a provenance note (repository, tag, commit, path, SHA-256 of the
#       file) and the list of changes;
#    3. the upstream file without #include "common.h" (the few definitions
#       it needs come from openblas/compat.h), inside namespace
#       ulmBLAS::openblas::kernel::NAME, with the kernel function (CNAME)
#       declared inline, and with the PATCHES of the kernel applied: exact
#       text replacements, each of which must occur exactly once;
#    4. #undef for every macro defined by the file.
#
#  For every kernel in ASM_KERNELS (assembly files *.S) it writes asm/NAME.S:
#
#    1. the upstream file with its license comment, unchanged except that
#       #include "common.h" is replaced by the definition of the symbol
#       name (ULMBLAS_OPENBLAS_KERNEL) and #include of asm/compat.h, which
#       defines the few macros of OpenBLAS's common headers the file uses
#       (DOUBLE, x86-64, without TRMMKERNEL: the GEMM kernel only);
#    2. a provenance note (repository, tag, commit, path, SHA-256) after the
#       license comment.
#
import hashlib, os, re, sys

TOP = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(TOP, 'ulmblas', 'level3', 'ukernel', 'openblas')

REPO = ('OpenBLAS', 'https://github.com/OpenMathLib/OpenBLAS', 'v0.3.34',
        'e0166008be8e466242aa76b2ff75ce3f0fbf574a')

#  name: description in the generated header (default: the DGEMM kernel for
#  SKYLAKEX), macros defined before the code
WHAT = {}
DEFINES = {}

#  name: (path, function name for CNAME, patches, description of patches)
KERNELS = {
    'skylakex_16x12': (
        'kernel/x86_64/dgemm_kernel_16x2_skylakex.c',
        'dgemm_kernel_skylakex',
        [
            #  B addressing for the packed B panels of ulmBLAS: one panel of
            #  12 columns stored k-major (12 consecutive elements per k)
            #  instead of 6 panels of 2 columns (2 elements per k, panels
            #  16*K bytes apart).  The kernel reads B at %1, %1+r12,
            #  %1+2*r12, r15 = %1+3*r12, r15+r12, r15+2*r12: with r12 = 16
            #  bytes these are the 6 column pairs of a row of 12 elements.
            ('movq %10,%%r12; salq $4,%%r12;',
             'movq $16,%%r12;'),
            #  ... and %1, r15 advance by one row of 12 elements (96 bytes)
            #  per k instead of one row of 2 elements (16 bytes); the 16 x 12
            #  path advances %1 in KERNEL_k1m16n6 and r15 in KERNEL_k1m16n12.
            ('#define KERNEL_k1m16n6 KERNEL_h_k1m16n4 unit_acc_m16n2(16,17,18,19,%1,%%r12,2) "addq $16,%1;"',
             '#define KERNEL_k1m16n6 KERNEL_h_k1m16n4 unit_acc_m16n2(16,17,18,19,%1,%%r12,2) "addq $96,%1;"'),
            ('#define KERNEL_k1m16n12 KERNEL_h_k1m16n12 "addq $16,%%r15;"',
             '#define KERNEL_k1m16n12 KERNEL_h_k1m16n12 "addq $96,%%r15;"'),
        ],
        '''//  - B addressing (3 replacements, see PATCHES in the script): the kernel
//    reads B as 6 panels of 2 columns, 16*K bytes apart (OpenBLAS packs B
//    with gemm_ncopy_2); ulmBLAS packs a panel of 12 columns k-major.  r12
//    (the distance of the column pairs) is set to 16 bytes instead of
//    16*K, and the B pointers %1 and r15 advance by 96 bytes (12 doubles)
//    instead of 16 bytes per k.  The computational instructions, the
//    register allocation, the loop structure and the prefetching are
//    unchanged.  After this change only the path for m = 16, n = 12
//    (COMPUTE_m16n12) addresses B correctly; the adapter calls the kernel
//    with full 16 x 12 tiles only.'''),
}

#  ZGEMM and CGEMM kernels (complex double, complex float): unchanged,
#  compiled as the variant NN (no conjugation of A and B; ulmBLAS conjugates
#  while packing).  With NR = 2 ulmBLAS packs B as OpenBLAS does
#  (zgemm_ncopy_2: consecutive panels of 2 columns), and A in panels of 4
#  (ZGEMM, zgemm_ncopy_4) or 8 rows (CGEMM, zgemm_ncopy_8).
for _name, _file, _cname, _target in (
        ('z_haswell_4x2', 'zgemm_kernel_4x2_haswell.c', 'zgemm_kernel_haswell',
         'HASWELL (AVX2/FMA)'),
        ('z_skylakex_4x2', 'zgemm_kernel_4x2_skylakex.c', 'zgemm_kernel_skylakex',
         'SKYLAKEX (AVX-512)'),
        ('c_haswell_8x2', 'cgemm_kernel_8x2_haswell.c', 'cgemm_kernel_haswell',
         'HASWELL (AVX2/FMA)'),
        ('c_skylakex_8x2', 'cgemm_kernel_8x2_skylakex.c', 'cgemm_kernel_skylakex',
         'SKYLAKEX (AVX-512)')):
    KERNELS[_name] = ('kernel/x86_64/' + _file, _cname, [],
        '''//  - NN is defined before the code: the kernel is compiled as its variant
//    without conjugation (OpenBLAS builds it for NN, NT, TN, TT the same
//    way); ulmBLAS conjugates A and B while packing.''')
    WHAT[_name] = ('the %sGEMM kernel of %%(label)s for ' % _name[0].upper()) + _target
    DEFINES[_name] = ('NN',)

#  SGEMM kernels (float): unchanged.  With MR = 8 (HASWELL) or 16
#  (SKYLAKEX) and NR = 4 ulmBLAS packs A and B as OpenBLAS (panels of
#  SGEMM_UNROLL_M rows and of SGEMM_UNROLL_N = 4 columns, consecutive).
for _name, _file, _cname, _target in (
        ('s_haswell_8x4', 'sgemm_kernel_8x4_haswell_2.c', 'sgemm_kernel_haswell',
         'HASWELL (AVX2/FMA)'),
        ('s_skylakex_16x4', 'sgemm_kernel_16x4_skylakex_3.c', 'sgemm_kernel_skylakex',
         'SKYLAKEX (AVX-512)')):
    KERNELS[_name] = ('kernel/x86_64/' + _file, _cname, [],
        '//  - No change of the code.')
    WHAT[_name] = 'the SGEMM kernel of %(label)s for ' + _target


#  name: (path, symbol, description).  The kernels are assembled unchanged
#  (see asm/compat.h for the definitions of OpenBLAS's common headers).
ASM_KERNELS = {
    'haswell_4x8': (
        'kernel/x86_64/dgemm_kernel_4x8_haswell.S',
        'ulmblas_openblas_dgemm_kernel_haswell_4x8',
        'the DGEMM kernel of OpenBLAS for HASWELL (AVX2/FMA)'),
}


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def macros(text):
    return sorted(set(re.findall(r'^\s*#\s*define\s+([A-Za-z_]\w*)', text,
                                 re.M)))


def vendor(name, path, cname, patches, patchdoc, src):
    full = os.path.join(src, path)
    body = open(full).read()
    license = open(os.path.join(src, 'LICENSE')).read().strip('\n')
    body, k = re.subn(r'^#include "common\.h"\s*\n', '', body, flags=re.M)
    if k != 1:
        sys.exit('%s: expected one #include "common.h"' % path)
    body = re.sub(r'^#include <stdint\.h>\s*\n', '', body, flags=re.M)
    #  other system headers (e.g. <immintrin.h> after the kernel) are
    #  included before the namespace
    sysinc = re.findall(r'^#include <[^>]+>[ \t]*$', body, flags=re.M)
    body = re.sub(r'^#include <[^>]+>[ \t]*\n', '', body, flags=re.M)
    for old, new in patches:
        if body.count(old) != 1:
            sys.exit('%s: patch target occurs %d times: %s'
                     % (path, body.count(old), old))
        body = body.replace(old, new)
    body, k = re.subn(r'^int __attribute__ \(\(noinline\)\)\nCNAME\(',
                      'inline int __attribute__ ((noinline))\nCNAME(',
                      body, flags=re.M)
    if k != 1:
        sys.exit('%s: kernel function not found' % path)
    label, url, tag, commit = REPO
    guard = 'ULMBLAS_LEVEL3_UKERNEL_OPENBLAS_KERNEL_%s_H' % name.upper()
    undef = macros(body)
    out = ['/*\n\n' + '\n'.join(('   ' + l).rstrip() for l in license.split('\n'))
           + '\n\n*/\n']
    out.append('''
//
//  ulmBLAS: %(what)s, as an
//  ulmBLAS micro kernel (see ../README.md).
//
//  Generated by mk/vendor_openblas_kernels.py, do not edit.  Source:
//
//      repository  %(url)s
//      tag         %(tag)s (commit %(commit)s)
//      file        %(path)s
//      SHA-256     %(sha)s
//
//  The file has no license comment of its own; the license above is the
//  LICENSE file of the OpenBLAS repository.  Changes:
//
//  - #include "common.h" and #include <stdint.h> removed: the definitions
//    the kernel uses (BLASLONG, CNAME) come from
//    ulmblas/level3/ukernel/openblas/compat.h.%(sysincdoc)s
%(patchdoc)s
//  - The code is wrapped into namespace ulmBLAS::openblas::kernel::%(name)s,
//    the kernel function CNAME (%(cname)s) is declared inline, and all
//    macros of this file are undefined at the end.
//
#ifndef %(guard)s
#define %(guard)s 1

#include <cstdint>
%(sysinc)s#include <ulmblas/level3/ukernel/openblas/compat.h>

namespace ulmBLAS { namespace openblas { namespace kernel { namespace %(name)s {

using namespace ulmBLAS::openblas::compat;
using std::int64_t;

#define CNAME %(cname)s
%(defines)s
''' % dict(label=label, url=url, tag=tag, commit=commit, path=path,
           sha=sha256(full), patchdoc=patchdoc, name=name, cname=cname,
           guard=guard,
           what=WHAT.get(name, 'the DGEMM kernel of %(label)s for SKYLAKEX (AVX-512)')
                % dict(label=label),
           defines=''.join('#define %s\n' % d for d in DEFINES.get(name, ())),
           sysinc=''.join(l.strip() + '\n' for l in sysinc),
           sysincdoc='' if not sysinc else
                     '\n//  - The system #include lines (%s) are moved before the'
                     '\n//    namespace.' % ', '.join(l.strip()[9:] for l in sysinc)))
    out.append(body.strip('\n') + '\n')
    out.append('\n#undef CNAME\n')
    out.append(''.join('#undef %s\n' % d for d in DEFINES.get(name, ())))
    out.append('\n} } } } // namespace %s, kernel, openblas, ulmBLAS\n\n' % name)
    out.append(''.join('#undef %s\n' % m for m in undef if m != 'CNAME'))
    out.append('\n#endif // %s\n' % guard)
    os.makedirs(os.path.join(OUT, 'kernel'), exist_ok=True)
    open(os.path.join(OUT, 'kernel', name + '.h'), 'w').write(''.join(out))
    return sha256(full)


def vendor_asm(name, path, symbol, what, src):
    full = os.path.join(src, path)
    body = open(full).read()
    m = re.match(r'(/\*.*?\*/\n)', body, re.S)
    if not m or 'Copyright' not in m.group(1):
        sys.exit('%s: no license comment at the beginning' % path)
    head, body = m.group(1), body[m.end():]
    include = ('#define ULMBLAS_OPENBLAS_KERNEL %s\n'
               '#include <ulmblas/level3/ukernel/openblas/asm/compat.h>\n'
               % symbol)
    body, k = re.subn(r'^#include "common\.h"[ \t]*\n', include, body,
                      flags=re.M)
    if k != 1:
        sys.exit('%s: expected one #include "common.h"' % path)
    label, url, tag, commit = REPO
    note = '''
/*
   ulmBLAS: %(what)s, assembled as the
   function %(symbol)s
   (see ../README.md).

   Generated by mk/vendor_openblas_kernels.py, do not edit.  Source:

       repository  %(url)s
       tag         %(tag)s (commit %(commit)s)
       file        %(path)s
       SHA-256     %(sha)s

   Changes: #include "common.h" is replaced by the definition of the
   function name (ULMBLAS_OPENBLAS_KERNEL) and #include of
   ulmblas/level3/ukernel/openblas/asm/compat.h, which defines the macros
   of OpenBLAS's common headers this file uses (PROLOGUE, EPILOGUE,
   PROFCODE, ALIGN_4, ALIGN_5, SIZE, BASE_SHIFT) as OpenBLAS defines them
   for double precision on x86-64.  The code itself is unchanged.
*/
''' % dict(what=what, symbol=symbol, url=url, tag=tag, commit=commit,
           path=path, sha=sha256(full))
    os.makedirs(os.path.join(OUT, 'asm'), exist_ok=True)
    open(os.path.join(OUT, 'asm', name + '.S'), 'w').write(head + note + body)
    return sha256(full)


def main():
    if len(sys.argv) != 2:
        sys.exit('usage: %s OPENBLAS_DIR' % sys.argv[0])
    for name, (path, cname, patches, doc) in sorted(KERNELS.items()):
        print('%-16s %s %s' % (name, REPO[2],
                                vendor(name, path, cname, patches, doc,
                                       sys.argv[1])))
    for name, (path, symbol, what) in sorted(ASM_KERNELS.items()):
        print('%-16s %s %s' % (name, REPO[2],
                                vendor_asm(name, path, symbol, what,
                                           sys.argv[1])))


if __name__ == '__main__':
    main()
