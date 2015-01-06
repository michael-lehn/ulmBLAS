#ifndef ULMBLAS_LEVEL3_UKERNEL_SSE_UTRUSM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SSE_UTRUSM_TCC 1

#include <iostream>
#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level1extensions/gecopy.h>
#include <ulmblas/level3/ukernel/sse/utrusm.h>

namespace ulmBLAS { namespace sse {

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value,
void>::type
utrusm(const double  *A,
       const double  *B,
       double        *C,
       IndexType     incRowC_,
       IndexType     incColC_)
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

    "leaq    (%%rcx,%%rsi,4),  %%rcx    \n\t"  // %rcx = &C[4*incRowC]
    "leaq    (%%rcx,%%rdi,2),  %%rdx    \n\t"  // %rdx = &C[4*incRowC+2*incColC]

    "subq    $1,               %%rsi    \n\t"
    "notq    %%rsi                      \n\t"  // negate incRow


    "movaps  0 * 16(%%rbx),    %%xmm8   \n\t"  // load xmm8  = (b00, b01 )
    "movaps  1 * 16(%%rbx),    %%xmm12  \n\t"  // load xmm12 = (b02, b03 )
    "movaps  2 * 16(%%rbx),    %%xmm9   \n\t"  // load xmm9  = (b10, b11 )
    "movaps  3 * 16(%%rbx),    %%xmm13  \n\t"  // load xmm13 = (b12, b13 )
    "movaps  4 * 16(%%rbx),    %%xmm10  \n\t"  // load xmm10 = (b20, b21 )
    "movaps  5 * 16(%%rbx),    %%xmm14  \n\t"  // load xmm14 = (b22, b23 )
    "movaps  6 * 16(%%rbx),    %%xmm11  \n\t"  // load xmm11 = (b30, b31 )
    "movaps  7 * 16(%%rbx),    %%xmm15  \n\t"  // load xmm15 = (b32, b33 )

    "leaq    (%%rcx,%%rsi),    %%rcx    \n\t"  // Move %rcx to previous row of C
    "leaq    (%%rdx,%%rsi),    %%rdx    \n\t"  // Move %rdx to previous row of C

    //
    // Compute (c30, c31, c32, c33)
    //
    "movddup (3+3*4)*8(%%rax), %%xmm3   \n\t"  // load xmm3 = (a33, a33)
    "mulpd   %%xmm3,           %%xmm11  \n\t"  // (c30,c31) = (b30*a33, b31*a33)
    "mulpd   %%xmm3,           %%xmm15  \n\t"  // (c32,c33) = (b32*a33, b33*a33)
    "movlpd  %%xmm11,    (%%rcx)        \n\t"  // store c30
    "movhpd  %%xmm11,    (%%rcx,%%rdi)  \n\t"  // store c31
    "movlpd  %%xmm15,    (%%rdx)        \n\t"  // store c32
    "movhpd  %%xmm15,    (%%rdx,%%rdi)  \n\t"  // store c33

    "leaq    (%%rcx,%%rsi),    %%rcx    \n\t"  // Move %rcx to previous row of C
    "leaq    (%%rdx,%%rsi),    %%rdx    \n\t"  // Move %rdx to previous row of C

    //
    // Compute (c20, c21, c22, c23)
    //
    "movddup (2+2*4)*8(%%rax), %%xmm2   \n\t"  // load xmm3 = (a22, a22)
    "movddup (2+3*4)*8(%%rax), %%xmm3   \n\t"  // load xmm2 = (a23, a23)

    "movaps  %%xmm3,           %%xmm7   \n\t"  // xmm7 = xmm3
    "mulpd   %%xmm11,          %%xmm3   \n\t"  // (a23*c30, a23*c31)
    "mulpd   %%xmm15,          %%xmm7   \n\t"  // (a23*c32, a23*c33)
    "subpd   %%xmm3,           %%xmm10  \n\t"  // (b20-a23*c30, b21-a23*c31)
    "subpd   %%xmm7,           %%xmm14  \n\t"  // (b22-a23*c32, b23-a23*c33)
    "mulpd   %%xmm2,           %%xmm10  \n\t"  // (c20, c21)
    "mulpd   %%xmm2,           %%xmm14  \n\t"  // (c22, c23)

    "movlpd  %%xmm10,    (%%rcx)        \n\t"  // store c20
    "movhpd  %%xmm10,    (%%rcx,%%rdi)  \n\t"  // store c21
    "movlpd  %%xmm14,    (%%rdx)        \n\t"  // store c22
    "movhpd  %%xmm14,    (%%rdx,%%rdi)  \n\t"  // store c23

    "leaq    (%%rcx,%%rsi),    %%rcx    \n\t"  // Move %rcx to previous row of C
    "leaq    (%%rdx,%%rsi),    %%rdx    \n\t"  // Move %rdx to previous row of C

    //
    // Compute (c10, c11, c12, c13)
    //
    "movddup (1+1*4)*8(%%rax), %%xmm1   \n\t"  // load xmm1 = (a11, a11)
    "movddup (1+2*4)*8(%%rax), %%xmm2   \n\t"  // load xmm2 = (a12, a12)
    "movddup (1+3*4)*8(%%rax), %%xmm3   \n\t"  // load xmm3 = (a13, a13)

    "movaps  %%xmm2,           %%xmm6   \n\t"  // xmm6 = xmm2
    "movaps  %%xmm3,           %%xmm7   \n\t"  // xmm7 = xmm3
    "mulpd   %%xmm11,          %%xmm3   \n\t"  // (a13*c30, a13*c31)
    "mulpd   %%xmm15,          %%xmm7   \n\t"  // (a13*c32, a13*c33)
    "mulpd   %%xmm10,          %%xmm2   \n\t"  // (a12*c20, a12*c21)
    "mulpd   %%xmm14,          %%xmm6   \n\t"  // (a12*c22, a12*c23)
    "subpd   %%xmm3,           %%xmm9   \n\t"  // (b10,b11) -= (a13*c30,a13*c31)
    "subpd   %%xmm7,           %%xmm13  \n\t"  // (b12,b13) -= (a13*c32,a13*c33)
    "subpd   %%xmm2,           %%xmm9   \n\t"  // (b10,b11) -= (a12*c20,a12*c21)
    "subpd   %%xmm6,           %%xmm13  \n\t"  // (b12,b13) -= (a12*c22,a12*c23)
    "mulpd   %%xmm1,           %%xmm9   \n\t"  // (c10, c11)
    "mulpd   %%xmm1,           %%xmm13  \n\t"  // (c12, c13)

    "movlpd  %%xmm9,     (%%rcx)        \n\t"  // store c10
    "movhpd  %%xmm9,     (%%rcx,%%rdi)  \n\t"  // store c11
    "movlpd  %%xmm13,    (%%rdx)        \n\t"  // store c12
    "movhpd  %%xmm13,    (%%rdx,%%rdi)  \n\t"  // store c13

    "leaq    (%%rcx,%%rsi),    %%rcx    \n\t"  // Move %rcx to next row of C
    "leaq    (%%rdx,%%rsi),    %%rdx    \n\t"  // Move %rdx to next row of C

    //
    // Compute (c00, c01, c02, c03)
    //
    "movddup (0+0*4)*8(%%rax), %%xmm0   \n\t"  // load xmm0 = (a00, a00)
    "movddup (0+1*4)*8(%%rax), %%xmm1   \n\t"  // load xmm1 = (a01, a01)
    "movddup (0+2*4)*8(%%rax), %%xmm2   \n\t"  // load xmm2 = (a02, a02)
    "movddup (0+3*4)*8(%%rax), %%xmm3   \n\t"  // load xmm3 = (a03, a03)

    "movaps  %%xmm1,           %%xmm5   \n\t"  // xmm5 = xmm1
    "movaps  %%xmm2,           %%xmm6   \n\t"  // xmm6 = xmm2
    "movaps  %%xmm3,           %%xmm7   \n\t"  // xmm7 = xmm3
    "mulpd   %%xmm11,          %%xmm3   \n\t"  // (a03*c30, a03*c31)
    "mulpd   %%xmm15,          %%xmm7   \n\t"  // (a03*c32, a03*c33)
    "mulpd   %%xmm10,          %%xmm2   \n\t"  // (a02*c10, a03*c11)
    "mulpd   %%xmm14,          %%xmm6   \n\t"  // (a02*c12, a03*c13)
    "mulpd   %%xmm9,           %%xmm1   \n\t"  // (a01*c10, a01*c11)
    "mulpd   %%xmm13,          %%xmm5   \n\t"  // (a01*c12, a01*c13)
    "subpd   %%xmm3,           %%xmm8   \n\t"  // (b00,b01) -= (a03*c30,a03*c31)
    "subpd   %%xmm7,           %%xmm12  \n\t"  // (b02,b03) -= (a03*c32,a03*c33)
    "subpd   %%xmm2,           %%xmm8   \n\t"  // (b00,b01) -= (a02*c20,a02*c21)
    "subpd   %%xmm6,           %%xmm12  \n\t"  // (b02,b03) -= (a02*c22,a02*c23)
    "subpd   %%xmm1,           %%xmm8   \n\t"  // (b00,b01) -= (a01*c10,a01*c11)
    "subpd   %%xmm5,           %%xmm12  \n\t"  // (b02,b03) -= (a01*c12,a01*c13)
    "mulpd   %%xmm0,           %%xmm8   \n\t"  // (c00, c01)
    "mulpd   %%xmm0,           %%xmm12  \n\t"  // (c02, c03)

    "movlpd  %%xmm8,     (%%rcx)        \n\t"  // store c20
    "movhpd  %%xmm8,     (%%rcx,%%rdi)  \n\t"  // store c21
    "movlpd  %%xmm12,    (%%rdx)        \n\t"  // store c22
    "movhpd  %%xmm12,    (%%rdx,%%rdi)  \n\t"  // store c23

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
        "xmm5", "xmm6", "xmm7",
        "xmm8", "xmm9", "xmm10", "xmm11",
        "xmm12", "xmm13", "xmm14", "xmm15"
    );
}

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SSE_UTRUSM_TCC 1
