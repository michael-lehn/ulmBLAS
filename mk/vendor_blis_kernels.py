#!/usr/bin/env python3
#
#  vendor_blis_kernels.py: generates the BLIS-derived DGEMM micro kernels in
#  ulmblas/level3/ukernel/blis/ from unmodified upstream source trees.
#
#      python3 mk/vendor_blis_kernels.py BLIS_DIR AOCL_DIR AOCL51_DIR
#
#  BLIS_DIR is an unpacked https://github.com/flame/blis tag 2.0 (commit
#  e8566eb3e773fb54d11b33e371d13f22d2941e50), AOCL_DIR an unpacked
#  https://github.com/amd/blis tag 5.0 (AOCL-BLAS 5.0, commit
#  34d4bbade33a4384cfaff2208c833fc33a311c5d), AOCL51_DIR tag 5.1 (AOCL-BLAS
#  5.1, commit 16f852a065e76e824d77bc39e2baa82ac19ed419).  The upstream trees are only
#  read.  For every kernel in KERNELS the script writes
#  ulmblas/level3/ukernel/blis/kernel/NAME.h, which contains
#
#    1. the license comment of the upstream file, unchanged;
#    2. a provenance note (repository, tag, commit, path, SHA-256 of the
#       file) and the list of the mechanical changes below;
#    3. the upstream file without its #include lines and without the
#       micro kernels for other data types (functions other than the DGEMM
#       kernel are removed, everything else is kept: macros, static data,
#       #if 0 blocks), inside namespace ulmBLAS::blis::kernel::NAME, with
#       the kernel function declared 'inline' (so that the header can be
#       included in several translation units);
#    4. #undef for every macro defined by the file and by the assembly macro
#       header it uses, so that the macros (e.g. ADD, MOV, add, sub) do not
#       leak into code that includes ulmBLAS.
#
#  The assembly macro headers frame/include/bli_x86_asm_macros.h of both
#  trees are copied unchanged to ulmblas/level3/ukernel/blis/asm/.
#
import hashlib, os, re, sys

TOP = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(TOP, 'ulmblas', 'level3', 'ukernel', 'blis')

REPO = {
    'blis': ('BLIS', 'https://github.com/flame/blis', '2.0',
             'e8566eb3e773fb54d11b33e371d13f22d2941e50'),
    'aocl': ('AOCL-BLAS', 'https://github.com/amd/blis', '5.0',
             '34d4bbade33a4384cfaff2208c833fc33a311c5d'),
    'aocl51': ('AOCL-BLAS', 'https://github.com/amd/blis', '5.1',
               '16f852a065e76e824d77bc39e2baa82ac19ed419'),
}

#  name: (tree, path, function)
KERNELS = {
    'haswell_6x8':     ('blis', 'kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c',
                        'bli_dgemm_haswell_asm_6x8'),
    'haswell_8x6':     ('blis', 'kernels/haswell/3/bli_gemm_haswell_asm_d8x6.c',
                        'bli_dgemm_haswell_asm_8x6'),
    'sandybridge_8x4': ('blis', 'kernels/sandybridge/3/bli_gemm_sandybridge_asm_d8x4.c',
                        'bli_dgemm_sandybridge_asm_8x4'),
    'penryn_4x4':      ('blis', 'kernels/penryn/3/bli_gemm_penryn_asm_d4x4.c',
                        'bli_dgemm_penryn_asm_4x4'),
    #  BLIS 0.x kernels in kernels/haswell/3/old/ (not built by BLIS 2.0, not
    #  used by any configuration): raw inline assembly without the macro
    #  header, with named labels; see LABELS.
    'haswell_12x4':    ('blis', 'kernels/haswell/3/old/bli_gemm_haswell_asm_d12x4.c',
                        'bli_dgemm_haswell_asm_12x4'),
    'haswell_4x12':    ('blis', 'kernels/haswell/3/old/bli_gemm_haswell_asm_d4x12.c',
                        'bli_dgemm_haswell_asm_4x12'),
    #  C with intrinsics (no assembly macro header); see PATCHES.
    'sandybridge_int_8x4': ('blis', 'kernels/sandybridge/3/bli_gemm_sandybridge_int_d8x4.c',
                        'bli_dgemm_sandybridge_int_8x4'),
    'skx_16x14':       ('blis', 'kernels/skx/3/bli_dgemm_skx_asm_16x14.c',
                        'bli_dgemm_skx_asm_16x14'),
    'skx_16x12_l2':    ('blis', 'kernels/skx/3/bli_dgemm_skx_asm_16x12_l2.c',
                        'bli_dgemm_skx_asm_16x12_l2'),
    'zen4_32x6':       ('aocl', 'kernels/zen4/3/bli_dgemm_zen4_asm_32x6.c',
                        'bli_dgemm_zen4_asm_32x6'),
    'zen4_8x24':       ('aocl', 'kernels/zen4/3/bli_dgemm_zen4_asm_8x24.c',
                        'bli_dgemm_zen4_asm_8x24'),
    #  AOCL-BLAS 5.1: the 5.0 version of this kernel compares beta with
    #  zmm2 before zmm2 is cleared in its column-major and general-stride
    #  paths; 5.1 clears zmm2 before all paths.
    'zen5_8x24':       ('aocl51', 'kernels/zen4/3/bli_dgemm_avx512_asm_8x24.c',
                        'bli_dgemm_avx512_asm_8x24'),
    #  ZGEMM kernels (complex double, selected with
    #  ULMBLAS_ZGEMM_UKERNEL_blis_NAME without the prefix z_; see select_z.h)
    'z_haswell_3x4':   ('blis', 'kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c',
                        'bli_zgemm_haswell_asm_3x4'),
    'z_haswell_4x3':   ('blis', 'kernels/haswell/3/bli_gemm_haswell_asm_d8x6.c',
                        'bli_zgemm_haswell_asm_4x3'),
    'z_sandybridge_4x4': ('blis', 'kernels/sandybridge/3/bli_gemm_sandybridge_asm_d8x4.c',
                        'bli_zgemm_sandybridge_asm_4x4'),
    'z_zen4_12x4':     ('aocl51', 'kernels/zen4/3/bli_zgemm_zen4_asm_12x4.c',
                        'bli_zgemm_zen4_asm_12x4'),
    'z_zen4_4x12':     ('aocl51', 'kernels/zen4/3/bli_zgemm_zen4_asm_4x12.c',
                        'bli_zgemm_zen4_asm_4x12'),
    #  AOCL-BLAS's own version of the haswell kernel (BLIS 0.x interface,
    #  special cases for alpha and beta), used by its zen, zen2, zen3
    #  configurations
    'z_aocl_haswell_3x4': ('aocl51', 'kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c',
                        'bli_zgemm_haswell_asm_3x4'),
    #  SGEMM kernels (float, selected with ULMBLAS_SGEMM_UKERNEL_blis_NAME
    #  without the prefix s_; see select_s.h)
    's_haswell_6x16':  ('blis', 'kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c',
                        'bli_sgemm_haswell_asm_6x16'),
    's_haswell_16x6':  ('blis', 'kernels/haswell/3/bli_gemm_haswell_asm_d8x6.c',
                        'bli_sgemm_haswell_asm_16x6'),
    's_haswell_24x4':  ('blis', 'kernels/haswell/3/old/bli_gemm_haswell_asm_d12x4.c',
                        'bli_sgemm_haswell_asm_24x4'),
    's_haswell_4x24':  ('blis', 'kernels/haswell/3/old/bli_gemm_haswell_asm_d4x12.c',
                        'bli_sgemm_haswell_asm_4x24'),
    's_sandybridge_8x8': ('blis', 'kernels/sandybridge/3/bli_gemm_sandybridge_asm_d8x4.c',
                        'bli_sgemm_sandybridge_asm_8x8'),
    's_penryn_8x4':    ('blis', 'kernels/penryn/3/bli_gemm_penryn_asm_d4x4.c',
                        'bli_sgemm_penryn_asm_8x4'),
    's_skx_32x12_l2':  ('blis', 'kernels/skx/3/bli_sgemm_skx_asm_32x12_l2.c',
                        'bli_sgemm_skx_asm_32x12_l2'),
    #  AOCL-BLAS's own version of the haswell kernel (BLIS 0.x interface),
    #  used by its zen, zen2, zen3 configurations
    's_aocl_haswell_6x16': ('aocl51', 'kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c',
                        'bli_sgemm_haswell_asm_6x16'),
    #  CGEMM kernels (complex float, selected with
    #  ULMBLAS_CGEMM_UKERNEL_blis_NAME without the prefix c_; see select_c.h)
    'c_haswell_3x8':   ('blis', 'kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c',
                        'bli_cgemm_haswell_asm_3x8'),
    'c_haswell_8x3':   ('blis', 'kernels/haswell/3/bli_gemm_haswell_asm_d8x6.c',
                        'bli_cgemm_haswell_asm_8x3'),
    'c_sandybridge_8x4': ('blis', 'kernels/sandybridge/3/bli_gemm_sandybridge_asm_d8x4.c',
                        'bli_cgemm_sandybridge_asm_8x4'),
    #  AOCL-BLAS's own version of the haswell kernel (BLIS 0.x interface),
    #  used by its zen, zen2, zen3 and zen4 configurations
    'c_aocl_haswell_3x8': ('aocl51', 'kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c',
                        'bli_cgemm_haswell_asm_3x8'),
}

#  Kernels that use the lowercase alias and() of the assembly macro header
#  (an operator name in C++, see CXX_OPERATORS): the script replaces and( by
#  the uppercase macro AND( of the same header inside the kernel function.
CXX_AND = ('z_haswell_3x4', 'z_haswell_4x3', 'z_sandybridge_4x4', 'c_haswell_3x8',
           'c_haswell_8x3', 'c_sandybridge_8x4', 'c_aocl_haswell_3x8')

ASM_HEADER = 'frame/include/bli_x86_asm_macros.h'

#  Kernels whose inline assembly uses named labels (".DLOOPKITER:"): they
#  are not unique if the compiler emits the asm statement more than once
#  in a translation unit (e.g. inlined), and not local on Mach-O (see issue
#  #0007 in test/known_issues.md).  The script replaces every label .DNAME
#  (and .SNAME of the SGEMM kernels) in their asm strings by
#  ULMBLAS_BLIS_LABEL "DNAME%=", i.e. ".LDNAME%=" on
#  ELF and "LDNAME%=" on Mach-O (as BLIS's assembly macro header does).
LABELS = ('haswell_12x4', 'haswell_4x12', 's_haswell_24x4', 's_haswell_4x24')

#  Kernels written in C without the assembly macro header.
NOASM = LABELS + ('sandybridge_int_8x4',)

#  Exact text replacements in the kernel function (each must occur exactly
#  once): implicit conversions from void pointers, valid in C, need a cast
#  in C++.
PATCHES = {
    'sandybridge_int_8x4': [
        ('const double* alpha = alpha0;', 'const double* alpha = (const double*)alpha0;'),
        ('const double* a     = a0;', 'const double* a     = (const double*)a0;'),
        ('const double* b     = b0;', 'const double* b     = (const double*)b0;'),
        ('const double* beta  = beta0;', 'const double* beta  = (const double*)beta0;'),
        ('      double* c     = c0;', '      double* c     = (double*)c0;'),
    ],
}


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def macros(text):
    """names of all macros defined in text"""
    return sorted(set(re.findall(r'^\s*#\s*define\s+([A-Za-z_]\w*)', text,
                                 re.M)))


def split_license(text):
    """the first comment of the file and the rest"""
    m = re.match(r'\s*(/\*.*?\*/)\n?', text, re.S)
    if not m:
        sys.exit('no license comment')
    return m.group(1), text[m.end():]


def mask(text):
    """text with comments, the contents of string and character literals and
    preprocessor lines replaced by blanks (positions and newlines kept)"""
    out, i, n = list(text), 0, len(text)
    def blank(a, b):
        for j in range(a, b):
            if out[j] != '\n':
                out[j] = ' '
    while i < n:
        if text.startswith('//', i):
            j = text.find('\n', i); j = n if j < 0 else j
            blank(i, j); i = j
        elif text.startswith('/*', i):
            j = text.index('*/', i) + 2
            blank(i, j); i = j
        elif text[i] in '"\'':
            q, j = text[i], i + 1
            while text[j] != q:
                j += 2 if text[j] == '\\' else 1
            blank(i + 1, j); i = j + 1
        else:
            i += 1
    text = ''.join(out)
    # preprocessor lines (with continuation lines)
    lines, cont = text.split('\n'), False
    for k, line in enumerate(lines):
        if cont or line.lstrip().startswith('#'):
            cont = line.rstrip().endswith('\\')
            lines[k] = ' ' * len(line)
    return '\n'.join(lines)


def functions(text):
    """(start, end, name) of every top-level function definition"""
    m_text = mask(text)
    out, depth, start = [], 0, None
    for i, c in enumerate(m_text):
        if c == '{':
            if depth == 0 and m_text[:i].rstrip().endswith(')'):
                m = None
                for m in re.finditer(r'^(?:BLIS_INLINE\s+|static\s+|inline\s+)*void\s+(\w+)\s*\(',
                                     m_text[:i], re.M):
                    pass
                start = (m.start(), m.group(1)) if m else None
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start:
                out.append((start[0], i + 1, start[1]))
                start = None
    return out


def vendor(name, tree, path, func, src):
    full = os.path.join(src, path)
    text = open(full).read()
    lic, body = split_license(text)
    asmh = asm_name(tree)
    body, k = re.subn(r'^\s*#\s*include\s*"bli_x86_asm_macros\.h"\s*$',
                      '#include <ulmblas/level3/ukernel/blis/asm/%s>' % asmh,
                      body, flags=re.M)
    if k != 1 and name not in NOASM:
        sys.exit('%s: expected one #include "bli_x86_asm_macros.h"' % path)
    sysinc = re.findall(r'^\s*#\s*include\s*<(?!ulmblas/)[^>]*>\s*$', body, flags=re.M)
    body = re.sub(r'^\s*#\s*include\s*[<"](?!ulmblas/)[^>"]*[>"]\s*\n', '', body, flags=re.M)
    for old, new in PATCHES.get(name, []):
        if body.count(old) != 1:
            sys.exit('%s: patch target occurs %d times: %s' % (path, body.count(old), old))
        body = body.replace(old, new)
    removed = []
    found = False
    nlabels = 0
    nand = 0
    for s, e, f in reversed(functions(body)):
        if f == func:
            found = True
            head = body[s:e]
            head = re.sub(r'^void\s+' + func, 'inline void ' + func, head, count=1)
            if name in LABELS:
                head, nlabels = re.subn(r'\.([DS][A-Z][A-Z0-9_]*)',
                                        r'" ULMBLAS_BLIS_LABEL "\1%=', head)
            if name in CXX_AND:
                head, nand = re.subn(r'\band\(', 'AND(', head)
                if nand == 0:
                    sys.exit('%s: no and( in %s' % (path, func))
            body = body[:s] + head + body[e:]
        else:
            removed.append(f)
            body = body[:s] + '// (' + f + ' removed)\n' + body[e:]
    if not found:
        sys.exit('%s: %s not found' % (path, func))
    label, url, tag, commit = REPO[tree]
    asm = open(os.path.join(src, ASM_HEADER)).read()
    undef = [m for m in sorted(set(macros(body)) | set(macros(asm)))
             if m not in CXX_OPERATORS]
    guard = 'ULMBLAS_LEVEL3_UKERNEL_BLIS_KERNEL_%s_H' % name.upper()
    out = []
    out.append(lic + '\n')
    out.append('''
//
//  ulmBLAS: %(func)s, vendored from %(label)s.
//
//  Generated by mk/vendor_blis_kernels.py, do not edit.  Source:
//
//      repository  %(url)s
//      tag         %(tag)s (commit %(commit)s)
//      file        %(path)s
//      SHA-256     %(sha)s
//
//  Mechanical changes (the code of %(func)s is unchanged%(anddoc0)s):
//
%(incdoc)s
//  - Functions for other data types removed: %(removed)s.
//  - The code is wrapped into namespace ulmBLAS::blis::kernel::%(name)s and
//    the kernel is declared inline.%(labeldoc)s%(patchdoc)s%(anddoc)s
//  - At the end all macros of this file and of the assembly macro header
//    are undefined.
//
#ifndef %(guard)s
#define %(guard)s 1

#include <ulmblas/level3/ukernel/blis/compat.h>
%(sysinc)s
namespace ulmBLAS { namespace blis { namespace kernel { namespace %(name)s {

using namespace ulmBLAS::blis::compat;

''' % dict(func=func, label=label, url=url, tag=tag, commit=commit, path=path,
           sha=sha256(full), removed=', '.join(sorted(removed)) or 'none',
           name=name, guard=guard, asmh=asm_name(tree),
           sysinc=''.join(l.strip() + '\n' for l in sysinc),
           incdoc=('''//  - #include "blis.h" (and other #include lines) removed: the definitions
//    BLIS provides for the kernel (types, edge case macros) come from
//    ulmblas/level3/ukernel/blis/compat.h; #include "bli_x86_asm_macros.h"
//    replaced (at the same place) by the unchanged copy asm/%s.''' % asm_name(tree))
                  if name not in NOASM else
                  '''//  - #include "blis.h" removed: the definitions BLIS provides for the
//    kernel (types, edge case macros) come from
//    ulmblas/level3/ukernel/blis/compat.h.''',
           patchdoc='' if name not in PATCHES else '''
//  - %d explicit casts of void pointers (valid in C, not in C++): see
//    PATCHES in mk/vendor_blis_kernels.py.  System #include lines are moved
//    before the namespace.''' % len(PATCHES[name]),
           anddoc0='' if name not in CXX_AND else
                   ' except for the\n//  renamed macro and( below',
           anddoc='' if name not in CXX_AND else '''
//  - %d uses of the macro and( (the lowercase alias of AND( in the assembly
//    macro header; "and" is an operator name in C++ and cannot be a macro)
//    are replaced by AND(, which emits the same instruction.''' % nand,
           labeldoc='' if name not in LABELS else '''
//  - The %d named labels .DNAME of the assembly are replaced by
//    ULMBLAS_BLIS_LABEL "DNAME%%=" (local and unique labels, compat.h).
//    The kernel has no #include "bli_x86_asm_macros.h".''' % nlabels))
    out.append(body.strip('\n') + '\n')
    out.append('\n} } } } // namespace %s, kernel, blis, ulmBLAS\n\n' % name)
    out.append('#include <ulmblas/level3/ukernel/blis/compat_end.h>\n')
    out.append(''.join('#undef %s\n' % m for m in undef))
    out.append('\n#endif // %s\n' % guard)
    os.makedirs(os.path.join(OUT, 'kernel'), exist_ok=True)
    open(os.path.join(OUT, 'kernel', name + '.h'), 'w').write(''.join(out))
    return sha256(full)


CXX_OPERATORS = ('and', 'or', 'xor', 'not', 'bitand', 'bitor', 'compl',
                 'and_eq', 'or_eq', 'xor_eq', 'not_eq')


def asm_copy(tree, path):
    """the assembly macro header with the only change needed for C++: the
    lowercase aliases and(), or(), xor() are not defined (these names are
    operators in C++ and cannot be macros; no vendored kernel uses them)"""
    label, url, tag, commit = REPO[tree]
    text = open(path).read()
    names = '|'.join(CXX_OPERATORS)
    text, k = re.subn(r'^(#define (?:%s)\(.*\n)' % names,
                      r'#if !defined(__cplusplus)\n\1#endif\n', text, flags=re.M)
    note = ('//\n'
            '//  ulmBLAS: frame/include/bli_x86_asm_macros.h of %s, tag %s (commit\n'
            '//  %s, SHA-256 %s), copied by\n'
            '//  mk/vendor_blis_kernels.py.  Only change: the %d definitions of the\n'
            '//  lowercase aliases and(), or(), xor() are enclosed in\n'
            '//  "#if !defined(__cplusplus)" (operator names in C++).\n'
            '//\n' % (label, tag, commit, sha256(path), k))
    return note + text


def asm_name(tree):
    #  the header of AOCL-BLAS 5.1 is identical to the one of 5.0 (checked
    #  in main)
    return 'bli_x86_asm_macros_%s.h' % {'blis': 'blis-2.0', 'aocl': 'aocl-5.0',
                                        'aocl51': 'aocl-5.0'}[tree]


def main():
    if len(sys.argv) != 4:
        sys.exit('usage: %s BLIS_DIR AOCL_DIR AOCL51_DIR' % sys.argv[0])
    src = {'blis': sys.argv[1], 'aocl': sys.argv[2], 'aocl51': sys.argv[3]}
    if sha256(os.path.join(src['aocl'], ASM_HEADER)) \
            != sha256(os.path.join(src['aocl51'], ASM_HEADER)):
        sys.exit('the assembly macro headers of AOCL-BLAS 5.0 and 5.1 differ')
    os.makedirs(os.path.join(OUT, 'asm'), exist_ok=True)
    for tree in ('blis', 'aocl'):
        a = os.path.join(src[tree], ASM_HEADER)
        open(os.path.join(OUT, 'asm', asm_name(tree)), 'w').write(asm_copy(tree, a))
        print('%-16s %s %s' % (asm_name(tree), REPO[tree][2], sha256(a)))
    for name, (tree, path, func) in sorted(KERNELS.items()):
        print('%-16s %s %s' % (name, REPO[tree][2],
                                vendor(name, tree, path, func, src[tree])))


if __name__ == '__main__':
    main()
