#ifndef ULMBLAS_LEVEL3_UKERNEL_SSE_UTRLSM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SSE_UTRLSM_TCC 1

#include <iostream>
#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level1extensions/gecopy.h>
#include <ulmblas/level3/ukernel/ref/utrlsm.h>

namespace ulmBLAS { namespace sse {

template <typename IndexType, typename T>
void
utrlsm(const T     *A,
       const T     *B,
       T           *C,
       IndexType   incRowC_,
       IndexType   incColC_)
{
    long incRowC = incRowC_;
    long incColC = incColC_;

    __asm__ volatile
    (
    "movq    %0,               %%rax    \n\t"  // Address of A stored in %rax
    "movq    %1,               %%rbx    \n\t"  // Address of B stored in %rbx
    "movq    %2,               %%rcx    \n\t"  // Address of C stored in %rcx
    "movq    %3,               %%rsi    \n\t"  // load incRowC
    "movq    %4,               %%rdi    \n\t"  // load incColC

    "leaq    (,%%rsi,8),       %%rsi    \n\t"  // incRowC *= sizeof(double)
    "leaq    (,%%rdi,8),       %%rdi    \n\t"  // incColC *= sizeof(double)

    "leaq    (%%rcx,%%rdi,2),  %%rdx    \n\t"  // load &C[2*incColC] in %rdx

    "movapd  0 * 16(%%rbx),    %%xmm8   \n\t"  // load xmm8  = (b00, b01 )
    "movapd  1 * 16(%%rbx),    %%xmm12  \n\t"  // load xmm12 = (b02, b03 )
    "movapd  2 * 16(%%rbx),    %%xmm9   \n\t"  // load xmm9  = (b10, b11 )
    "movapd  3 * 16(%%rbx),    %%xmm13  \n\t"  // load xmm13 = (b12, b13 )
    "movapd  4 * 16(%%rbx),    %%xmm10  \n\t"  // load xmm10 = (b20, b21 )
    "movapd  5 * 16(%%rbx),    %%xmm14  \n\t"  // load xmm14 = (b22, b23 )
    "movapd  6 * 16(%%rbx),    %%xmm11  \n\t"  // load xmm11 = (b30, b31 )
    "movapd  7 * 16(%%rbx),    %%xmm15  \n\t"  // load xmm15 = (b32, b33 )

    //
    // Compute (c00, c01, c02, c03)
    //
    "movddup (0+0*4)*8(%%rax), %%xmm0   \n\t"  // load xmm0 = (a00, a00)
    "mulpd   %%xmm0,           %%xmm8   \n\t"  // (c00,c01) = (b00*a00, b01*a00)
    "mulpd   %%xmm0,           %%xmm12  \n\t"  // (c02,c03) = (b02*a00, b03*a00)


  //"movaps  %%xmm8,     (%%rcx)        \n\t"  // store c00
  //"movaps  %%xmm12,    (%%rdx)        \n\t"  // store c02
    "movlpd  %%xmm8,     (%%rcx)        \n\t"  // store c00
    "movhpd  %%xmm8,     (%%rcx,%%rdi)  \n\t"  // store c01
    "movlpd  %%xmm12,    (%%rdx)        \n\t"  // store c02
    "movhpd  %%xmm12,    (%%rdx,%%rdi)  \n\t"  // store c03

    "leaq    (%%rcx,%%rsi),    %%rcx    \n\t"  // Move %rcx to next row of C
    "leaq    (%%rdx,%%rsi),    %%rdx    \n\t"  // Move %rdx to next row of C

    //
    // Compute (c10, c11, c12, c13)
    //
    "movddup (1+0*4)*8(%%rax), %%xmm0   \n\t"  // load xmm0 = (a10, a10)
    "movapd  %%xmm0,           %%xmm4   \n\t"  // xmm4 = xmm0
    "movddup (1+1*4)*8(%%rax), %%xmm1   \n\t"  // load xmm1 = (a11, a11)

    "mulpd   %%xmm8,           %%xmm0   \n\t"  // (a10*c00, a10*c01)
    "subpd   %%xmm0,           %%xmm9   \n\t"  // (b10-a10*c00, b11-a10*c01)
    "mulpd   %%xmm12,          %%xmm4   \n\t"  // (a10*c02, a10*c03)
    "subpd   %%xmm4,           %%xmm13  \n\t"  // (b12-a10*c02, b13-a10*c03)
    "mulpd   %%xmm1,           %%xmm9   \n\t"  // (c10, c11)
    "mulpd   %%xmm1,           %%xmm13  \n\t"  // (c12, c13)

  //"movaps  %%xmm9,     (%%rcx)        \n\t"  // store c00
  //"movaps  %%xmm13,    (%%rdx)        \n\t"  // store c02
    "movlpd  %%xmm9,     (%%rcx)        \n\t"  // store c10
    "movhpd  %%xmm9,     (%%rcx,%%rdi)  \n\t"  // store c11
    "movlpd  %%xmm13,    (%%rdx)        \n\t"  // store c12
    "movhpd  %%xmm13,    (%%rdx,%%rdi)  \n\t"  // store c13

    "leaq    (%%rcx,%%rsi),    %%rcx    \n\t"  // Move %rcx to next row of C
    "leaq    (%%rdx,%%rsi),    %%rdx    \n\t"  // Move %rdx to next row of C

    //
    // Compute (c20, c21, c22, c23)
    //
    "movddup (2+0*4)*8(%%rax), %%xmm0   \n\t"  // load xmm0 = (a20, a20)
    "movapd  %%xmm0,           %%xmm4   \n\t"  // xmm4 = xmm0
    "movddup (2+1*4)*8(%%rax), %%xmm1   \n\t"  // load xmm1 = (a21, a21)
    "movapd  %%xmm1,           %%xmm5   \n\t"  // xmm5 = xmm1
    "movddup (2+2*4)*8(%%rax), %%xmm2   \n\t"  // load xmm2 = (a22, a22)

    "mulpd   %%xmm8,           %%xmm0   \n\t"  // (a20*c00, a20*c01)
    "subpd   %%xmm0,           %%xmm10  \n\t"  // (b20,b21) -= (a20*c00,a20*c01)
    "mulpd   %%xmm12,          %%xmm4   \n\t"  // (a20*c02, a20*c03)
    "subpd   %%xmm4,           %%xmm14  \n\t"  // (b22,b23) -= (a20*c02,a20*c03)
    "mulpd   %%xmm9,           %%xmm1   \n\t"  // (a21*c10, a21*c11)
    "subpd   %%xmm1,           %%xmm10  \n\t"  // (b20,b21) -= (a21*c10,a21*c11)
    "mulpd   %%xmm13,          %%xmm5   \n\t"  // (a21*c12, a21*c13)
    "subpd   %%xmm5,           %%xmm14  \n\t"  // (b22,b23) -= (a21*c12,a21*c13)
    "mulpd   %%xmm2,           %%xmm10  \n\t"  // (c20, c21)
    "mulpd   %%xmm2,           %%xmm14  \n\t"  // (c22, c23)

  //"movaps  %%xmm10,    (%%rcx)        \n\t"  // store c00
  //"movaps  %%xmm14,    (%%rdx)        \n\t"  // store c02
    "movlpd  %%xmm10,    (%%rcx)        \n\t"  // store c20
    "movhpd  %%xmm10,    (%%rcx,%%rdi)  \n\t"  // store c21
    "movlpd  %%xmm14,    (%%rdx)        \n\t"  // store c22
    "movhpd  %%xmm14,    (%%rdx,%%rdi)  \n\t"  // store c23

    "leaq    (%%rcx,%%rsi),    %%rcx    \n\t"  // Move %rcx to next row of C
    "leaq    (%%rdx,%%rsi),    %%rdx    \n\t"  // Move %rdx to next row of C

    //
    // Compute (c30, c31, c32, c33)
    //
    "movddup (3+0*4)*8(%%rax), %%xmm0   \n\t"  // load xmm0 = (a30, a30)
    "movapd  %%xmm0,           %%xmm4   \n\t"  // xmm4 = xmm0
    "movddup (3+1*4)*8(%%rax), %%xmm1   \n\t"  // load xmm1 = (a31, a31)
    "movapd  %%xmm1,           %%xmm5   \n\t"  // xmm5 = xmm1
    "movddup (3+2*4)*8(%%rax), %%xmm2   \n\t"  // load xmm2 = (a32, a32)
    "movapd  %%xmm2,           %%xmm6   \n\t"  // xmm6 = xmm2
    "movddup (3+3*4)*8(%%rax), %%xmm3   \n\t"  // load xmm3 = (a33, a33)

    "mulpd   %%xmm8,           %%xmm0   \n\t"  // (a30*c00, a30*c01)
    "subpd   %%xmm0,           %%xmm11  \n\t"  // (b30,b31) -= (a30*c00,a30*c01)
    "mulpd   %%xmm12,          %%xmm4   \n\t"  // (a30*c02, a30*c03)
    "subpd   %%xmm4,           %%xmm15  \n\t"  // (b32,b33) -= (a30*c02,a30*c03)
    "mulpd   %%xmm9,           %%xmm1   \n\t"  // (a31*c10, a31*c11)
    "subpd   %%xmm1,           %%xmm11  \n\t"  // (b30,b31) -= (a31*c10,a31*c11)
    "mulpd   %%xmm13,          %%xmm5   \n\t"  // (a31*c12, a31*c13)
    "subpd   %%xmm5,           %%xmm15  \n\t"  // (b32,b33) -= (a31*c12,a31*c13)
    "mulpd   %%xmm10,          %%xmm2   \n\t"  // (a31*c20, a31*c21)
    "subpd   %%xmm2,           %%xmm11  \n\t"  // (b30,b31) -= (a31*c20,a31*c21)
    "mulpd   %%xmm14,          %%xmm6   \n\t"  // (a31*c22, a31*c23)
    "subpd   %%xmm6,           %%xmm15  \n\t"  // (b32,b33) -= (a31*c22,a31*c23)
    "mulpd   %%xmm3,           %%xmm11  \n\t"  // (c30, c31)
    "mulpd   %%xmm3,           %%xmm15  \n\t"  // (c32, c33)

  //"movaps  %%xmm11,    (%%rcx)        \n\t"  // store c00
  //"movaps  %%xmm15,    (%%rdx)        \n\t"  // store c02
    "movlpd  %%xmm11,    (%%rcx)        \n\t"  // store c30
    "movhpd  %%xmm11,    (%%rcx,%%rdi)  \n\t"  // store c31
    "movlpd  %%xmm15,    (%%rdx)        \n\t"  // store c32
    "movhpd  %%xmm15,    (%%rdx,%%rdi)  \n\t"  // store c33

    : // output
    : // input
        "m" (A),        // 0
        "m" (B),        // 1
        "m" (C),        // 2
        "m" (incRowC),  // 3
        "m" (incColC)   // 4
    : // register clobber list
        "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
        "xmm0", "xmm1", "xmm2", "xmm3",
        "xmm4", "xmm5", "xmm6",
        "xmm8", "xmm9", "xmm10", "xmm11",
        "xmm12", "xmm13", "xmm14", "xmm15"
    );
}

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SSE_UTRLSM_TCC 1
