/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

//
//  ulmBLAS: AVX/FMA micro kernel for double (MR = 6, NR = 8).
//
//  Provenance: the inline assembly is the micro kernel
//  bli_dgemm_haswell_asm_6x8 of BLIS (https://github.com/flame/blis, file
//  kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c, tag 2.0, commit
//  e8566eb3e773fb54d11b33e371d13f22d2941e50; the assembly is the same since
//  BLIS 0.9.0), as adapted for the course "High Performance Computing I",
//  Ulm University, winter term 2025/26 (session19/step03/dugemm_asm_6x8.hpp:
//  the BLIS macros expanded to AT&T syntax, C++ interface).
//
//  Modifications for ulmBLAS (the computational instructions are those of
//  BLIS):
//
//  - Interface of the ulmBLAS micro kernels (alpha and beta by reference,
//    strides of C as IndexType, nextA and nextB are not used).
//  - Numeric local labels (1: ... with 1b, 1f) instead of the named labels:
//    named labels become symbols on Mach-O (macOS), which the linker may
//    separate from the code before them (see issue #0007 in
//    test/known_issues.md).
//  - beta = NaN: "jp" before the test for beta == 0 (an unordered comparison
//    sets ZF), so that C <- beta*C + alpha*A*B is computed as for any other
//    nonzero beta.
//  - The assembly code supports C with unit row or unit column stride (and
//    full MR x NR tiles, as all ulmBLAS micro kernels).  For general strides
//    (BLIS: GEMM_UKR_SETUP_CT_AMBI) alpha*A*B is computed in a local column
//    major tile and C is updated in C++.
//  - Only the general purpose registers used by the assembly code are in the
//    clobber list (BLIS also lists r8 to r12 and r15).  With all of them
//    clobbered GCC cannot address the memory operands in some builds
//    ("impossible constraints", e.g. with -fsanitize=address).
//
//  Requirements: x86-64 with AVX and FMA (selected with USE_AVX, see
//  ulmblas/config/simd.h); B aligned to 32 bytes (vmovapd); A and B are read
//  up to NR elements beyond the panels (the packing buffers are allocated
//  with this padding).
//

#ifndef ULMBLAS_LEVEL3_UKERNEL_AVX_UGEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_AVX_UGEMM_TCC 1

#include <cstdint>
#include <ulmblas/level3/ukernel/avx/ugemm.h>

namespace ulmBLAS { namespace avx {

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value,
void>::type
ugemm(IndexType      kc,
      const double   &alpha,
      const double   *A,
      const double   *B,
      const double   &beta,
      double         *C,
      IndexType      incRowC,
      IndexType      incColC,
      const double   *nextA,
      const double   *nextB)
{
    const IndexType MR = BlockSizeUGemm<double>::MR;
    const IndexType NR = BlockSizeUGemm<double>::NR;

    if (incRowC!=IndexType(1) && incColC!=IndexType(1)) {
        double C_[MR*NR];

        ugemm(kc, alpha, A, B, 0.0, C_, IndexType(1), MR, nextA, nextB);
        if (beta==0.0) {
            for (IndexType j=0; j<NR; ++j) {
                for (IndexType i=0; i<MR; ++i) {
                    C[i*incRowC+j*incColC] = C_[i+j*MR];
                }
            }
        } else {
            for (IndexType j=0; j<NR; ++j) {
                for (IndexType i=0; i<MR; ++i) {
                    C[i*incRowC+j*incColC] = beta*C[i*incRowC+j*incColC]
                                           + C_[i+j*MR];
                }
            }
        }
        return;
    }

    uint64_t k_iter = kc / 4;
    uint64_t k_left = kc % 4;
    uint64_t rs_c   = incRowC;
    uint64_t cs_c   = incColC;
    const double *alpha_ = &alpha;
    const double *beta_  = &beta;

    __asm__ volatile (
    "vzeroall	    \n\t"
    "mov	    %[a], %%rax\n\t"
    "mov	    %[b], %%rbx\n\t"
    "add	    $32*4, %%rbx\n\t"
    "vmovapd	    -4*32(%%rbx),%%Ymm0\n\t"
    "vmovapd	    -3*32(%%rbx),%%Ymm1\n\t"
    "mov	    %[c], %%rcx\n\t"
    "mov	    %[rs_c], %%rdi\n\t"
    "lea	    (,%%rdi,8), %%rdi\n\t"
    "mov	    %[cs_c], %%rsi\n\t"
    "lea	    (,%%rsi,8), %%rsi\n\t"
    "cmp	    $8, %%rdi\n\t"
    "je 1f\n\t"
    "lea	    (%%rdi,%%rdi,2),%%r13\n\t"
    "lea	    (%%rcx,%%r13,1),%%rdx\n\t"
    "prefetcht0	    7*8(%%rcx)\n\t"
    "prefetcht0	    7*8(%%rcx,%%rdi,1)\n\t"
    "prefetcht0	    7*8(%%rcx,%%rdi,2)\n\t"
    "prefetcht0	    7*8(%%rdx)\n\t"
    "prefetcht0	    7*8(%%rdx,%%rdi,1)\n\t"
    "prefetcht0	    7*8(%%rdx,%%rdi,2)\n\t"
    "jmp 2f\n\t"
    "1:\n\t"
    "lea	    (%%rsi,%%rsi,2),%%r13\n\t"
    "lea	    (%%rcx,%%r13,1),%%rdx\n\t"
    "prefetcht0	    7*8(%%rcx)\n\t"
    "prefetcht0	    7*8(%%rcx,%%rsi,1)\n\t"
    "prefetcht0	    7*8(%%rcx,%%rsi,2)\n\t"
    "prefetcht0	    7*8(%%rdx)\n\t"
    "prefetcht0	    7*8(%%rdx,%%rsi,1)\n\t"
    "prefetcht0	    7*8(%%rdx,%%rsi,2)\n\t"
    "prefetcht0	    7*8(%%rdx,%%r13,1)\n\t"
    "prefetcht0	    7*8(%%rdx,%%rsi,4)\n\t"
    "2:\n\t"
    "mov	    %[k_iter],%%rsi\n\t"
    "test	    %%rsi,%%rsi\n\t"
    "je 4f\n\t"
    "3:\n\t"
    "prefetcht0	    64*8(%%rax)\n\t"

    "vbroadcastsd   0*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   1*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm4\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm5\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm6\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm7\n\t"

    "vbroadcastsd   2*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   3*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm8\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm9\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm10\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm11\n\t"

    "vbroadcastsd   4*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   5*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm12\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm13\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm14\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm15\n\t"

    "vmovapd	    -2*32(%%rbx),%%Ymm0\n\t"
    "vmovapd	    -1*32(%%rbx),%%Ymm1\n\t"

    "prefetcht0	    72*8(%%rax)\n\t"

    "vbroadcastsd   6*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   7*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm4\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm5\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm6\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm7\n\t"

    "vbroadcastsd   8*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   9*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm8\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm9\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm10\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm11\n\t"

    "vbroadcastsd   10*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   11*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm12\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm13\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm14\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm15\n\t"

    "vmovapd	    0*32(%%rbx),%%Ymm0\n\t"
    "vmovapd	    1*32(%%rbx),%%Ymm1\n\t"

    "prefetcht0	    80*8(%%rax)\n\t"

    "vbroadcastsd   12*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   13*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm4\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm5\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm6\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm7\n\t"

    "vbroadcastsd   14*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   15*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm8\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm9\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm10\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm11\n\t"

    "vbroadcastsd   16*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   17*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm12\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm13\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm14\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm15\n\t"

    "vmovapd	    2*32(%%rbx),%%Ymm0\n\t"
    "vmovapd	    3*32(%%rbx),%%Ymm1\n\t"

    "vbroadcastsd   18*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   19*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm4\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm5\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm6\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm7\n\t"

    "vbroadcastsd   20*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   21*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm8\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm9\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm10\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm11\n\t"

    "vbroadcastsd   22*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   23*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm12\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm13\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm14\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm15\n\t"

    "add	    $4*6*8,%%rax\n\t"
    "add	    $4*8*8,%%rbx\n\t"

    "vmovapd	    -4*32(%%rbx),%%Ymm0\n\t"
    "vmovapd	    -3*32(%%rbx),%%Ymm1\n\t"

    "dec	    %%rsi\n\t"
    "jne	    3b\n\t"

    "4:\n\t"

    "mov	    %[k_left],%%rsi\n\t"
    "test	    %%rsi,%%rsi\n\t"
    "je		    6f\n\t"

    "5:\n\t"

    "prefetcht0	    64*8(%%rax)\n\t"

    "vbroadcastsd   0*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   1*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm4\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm5\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm6\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm7\n\t"

    "vbroadcastsd   2*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   3*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm8\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm9\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm10\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm11\n\t"

    "vbroadcastsd   4*8(%%rax),%%Ymm2\n\t"
    "vbroadcastsd   5*8(%%rax),%%Ymm3\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm2,%%Ymm12\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm2,%%Ymm13\n\t"
    "vfmadd231pd    %%Ymm0,%%Ymm3,%%Ymm14\n\t"
    "vfmadd231pd    %%Ymm1,%%Ymm3,%%Ymm15\n\t"

    "add	    $1*6*8,%%rax\n\t"
    "add	    $1*8*8,%%rbx\n\t"

    "vmovapd	    -4*32(%%rbx),%%Ymm0\n\t"
    "vmovapd	    -3*32(%%rbx),%%Ymm1\n\t"

    "dec	    %%rsi\n\t"
    "jne 5b\n\t"

    "6:\n\t"

    "mov	    %[alpha],%%rax\n\t"
    "mov	    %[beta],%%rbx\n\t"
    "vbroadcastsd   (%%rax),%%Ymm0\n\t"
    "vbroadcastsd   (%%rbx),%%Ymm3\n\t"

    "vmulpd	    %%Ymm0,%%Ymm4,%%Ymm4\n\t"
    "vmulpd	    %%Ymm0,%%Ymm5,%%Ymm5\n\t"
    "vmulpd 	    %%Ymm0,%%Ymm6,%%Ymm6\n\t"
    "vmulpd 	    %%Ymm0,%%Ymm7,%%Ymm7\n\t"
    "vmulpd 	    %%Ymm0,%%Ymm8,%%Ymm8\n\t"
    "vmulpd 	    %%Ymm0,%%Ymm9,%%Ymm9\n\t"
    "vmulpd 	    %%Ymm0,%%Ymm10,%%Ymm10\n\t"
    "vmulpd 	    %%Ymm0,%%Ymm11,%%Ymm11\n\t"
    "vmulpd 	    %%Ymm0,%%Ymm12,%%Ymm12\n\t"
    "vmulpd 	    %%Ymm0,%%Ymm13,%%Ymm13\n\t"
    "vmulpd 	    %%Ymm0,%%Ymm14,%%Ymm14\n\t"
    "vmulpd 	    %%Ymm0,%%Ymm15,%%Ymm15\n\t"

    "mov	    %[cs_c],%%rsi\n\t"
    "lea	    (,%%rsi,8),%%rsi\n\t"

    "lea	    (%%rcx,%%rsi,4),%%rdx\n\t"
    "lea	    (%%rcx,%%rdi,4),%%r14\n\t"

    "lea	    (%%rsi,%%rsi,2),%%r13\n\t"

    "vxorpd	    %%Ymm0,%%Ymm0,%%Ymm0\n\t"
    "vucomisd	    %%Xmm0,%%Xmm3\n\t"
    "jp 11f\n\t"
    "je 7f\n\t"
    "11:\n\t"

    "cmp $8,%%rdi\n\t"
    "je 8f\n\t"

    "vfmadd231pd    (%%rcx),%%Ymm3,%%Ymm4\n\t"
    "vmovupd	    %%Ymm4,(%%rcx)\n\t"
    "vfmadd231pd    32(%%rcx),%%Ymm3,%%Ymm5\n\t"
    "vmovupd	    %%Ymm5,32(%%rcx)\n\t"
    "add	    %%rdi,%%rcx\n\t"

    "vfmadd231pd    (%%rcx),%%Ymm3,%%Ymm6\n\t"
    "vmovupd	    %%Ymm6,(%%rcx)\n\t"
    "vfmadd231pd    32(%%rcx),%%Ymm3,%%Ymm7\n\t"
    "vmovupd	    %%Ymm7,32(%%rcx)\n\t"
    "add	    %%rdi,%%rcx\n\t"

    "vfmadd231pd (%%rcx),%%Ymm3,%%Ymm8\n\t"
    "vmovupd %%Ymm8,(%%rcx)\n\t"
    "vfmadd231pd 32(%%rcx),%%Ymm3,%%Ymm9\n\t"
    "vmovupd %%Ymm9,32(%%rcx)\n\t"
    "add %%rdi,%%rcx\n\t"

    "vfmadd231pd (%%rcx),%%Ymm3,%%Ymm10\n\t"
    "vmovupd %%Ymm10,(%%rcx)\n\t"
    "vfmadd231pd 32(%%rcx),%%Ymm3,%%Ymm11\n\t"
    "vmovupd %%Ymm11,32(%%rcx)\n\t"
    "add %%rdi,%%rcx\n\t"

    "vfmadd231pd (%%rcx),%%Ymm3,%%Ymm12\n\t"
    "vmovupd %%Ymm12,(%%rcx)\n\t"
    "vfmadd231pd 32(%%rcx),%%Ymm3,%%Ymm13\n\t"
    "vmovupd %%Ymm13,32(%%rcx)\n\t"
    "add %%rdi,%%rcx\n\t"

    "vfmadd231pd (%%rcx),%%Ymm3,%%Ymm14\n\t"
    "vmovupd %%Ymm14,(%%rcx)\n\t"
    "vfmadd231pd 32(%%rcx),%%Ymm3,%%Ymm15\n\t"
    "vmovupd %%Ymm15,32(%%rcx)\n\t"

    "jmp 10f\n\t"

    "8:\n\t"

    "vunpcklpd %%Ymm6,%%Ymm4,%%Ymm0\n\t"
    "vunpckhpd %%Ymm6,%%Ymm4,%%Ymm1\n\t"
    "vunpcklpd %%Ymm10,%%Ymm8,%%Ymm2\n\t"
    "vunpckhpd %%Ymm10,%%Ymm8,%%Ymm3\n\t"
    "vinsertf128 $0x1,%%Xmm2,%%Ymm0,%%Ymm4\n\t"
    "vinsertf128 $0x1,%%Xmm3,%%Ymm1,%%Ymm6\n\t"
    "vperm2f128 $0x31,%%Ymm2,%%Ymm0,%%Ymm8\n\t"
    "vperm2f128 $0x31,%%Ymm3,%%Ymm1,%%Ymm10\n\t"

    "vbroadcastsd (%%rbx),%%Ymm3\n\t"

    "vfmadd231pd (%%rcx),%%Ymm3,%%Ymm4\n\t"
    "vfmadd231pd (%%rcx,%%rsi,1),%%Ymm3,%%Ymm6\n\t"
    "vfmadd231pd (%%rcx,%%rsi,2),%%Ymm3,%%Ymm8\n\t"
    "vfmadd231pd (%%rcx,%%r13,1),%%Ymm3,%%Ymm10\n\t"
    "vmovupd %%Ymm4,(%%rcx)\n\t"
    "vmovupd %%Ymm6,(%%rcx,%%rsi,1)\n\t"
    "vmovupd %%Ymm8,(%%rcx,%%rsi,2)\n\t"
    "vmovupd %%Ymm10,(%%rcx,%%r13,1)\n\t"

    "lea (%%rcx,%%rsi,4),%%rcx\n\t"

    "vunpcklpd %%Ymm14,%%Ymm12,%%Ymm0\n\t"
    "vunpckhpd %%Ymm14,%%Ymm12,%%Ymm1\n\t"
    "vextractf128 $0x1,%%Ymm0,%%Xmm2\n\t"
    "vextractf128 $0x1,%%Ymm1,%%Xmm4\n\t"

    "vfmadd231pd (%%r14),%%Xmm3,%%Xmm0\n\t"
    "vfmadd231pd (%%r14,%%rsi,1),%%Xmm3,%%Xmm1\n\t"
    "vfmadd231pd (%%r14,%%rsi,2),%%Xmm3,%%Xmm2\n\t"
    "vfmadd231pd (%%r14,%%r13,1),%%Xmm3,%%Xmm4\n\t"
    "vmovupd %%Xmm0,(%%r14)\n\t"
    "vmovupd %%Xmm1,(%%r14,%%rsi,1)\n\t"
    "vmovupd %%Xmm2,(%%r14,%%rsi,2)\n\t"
    "vmovupd %%Xmm4,(%%r14,%%r13,1)\n\t"

    "lea (%%r14,%%rsi,4),%%r14\n\t"

    "vunpcklpd %%Ymm7,%%Ymm5,%%Ymm0\n\t"
    "vunpckhpd %%Ymm7,%%Ymm5,%%Ymm1\n\t"
    "vunpcklpd %%Ymm11,%%Ymm9,%%Ymm2\n\t"
    "vunpckhpd %%Ymm11,%%Ymm9,%%Ymm3\n\t"
    "vinsertf128 $0x1,%%Xmm2,%%Ymm0,%%Ymm5\n\t"
    "vinsertf128 $0x1,%%Xmm3,%%Ymm1,%%Ymm7\n\t"
    "vperm2f128 $0x31,%%Ymm2,%%Ymm0,%%Ymm9\n\t"
    "vperm2f128 $0x31,%%Ymm3,%%Ymm1,%%Ymm11\n\t"

    "vbroadcastsd (%%rbx),%%Ymm3\n\t"

    "vfmadd231pd (%%rcx),%%Ymm3,%%Ymm5\n\t"
    "vfmadd231pd (%%rcx,%%rsi,1),%%Ymm3,%%Ymm7\n\t"
    "vfmadd231pd (%%rcx,%%rsi,2),%%Ymm3,%%Ymm9\n\t"
    "vfmadd231pd (%%rcx,%%r13,1),%%Ymm3,%%Ymm11\n\t"
    "vmovupd %%Ymm5,(%%rcx)\n\t"
    "vmovupd %%Ymm7,(%%rcx,%%rsi,1)\n\t"
    "vmovupd %%Ymm9,(%%rcx,%%rsi,2)\n\t"
    "vmovupd %%Ymm11,(%%rcx,%%r13,1)\n\t"

    "vunpcklpd %%Ymm15,%%Ymm13,%%Ymm0\n\t"
    "vunpckhpd %%Ymm15,%%Ymm13,%%Ymm1\n\t"
    "vextractf128 $0x1,%%Ymm0,%%Xmm2\n\t"
    "vextractf128 $0x1,%%Ymm1,%%Xmm4\n\t"

    "vfmadd231pd (%%r14),%%Xmm3,%%Xmm0\n\t"
    "vfmadd231pd (%%r14,%%rsi,1),%%Xmm3,%%Xmm1\n\t"
    "vfmadd231pd (%%r14,%%rsi,2),%%Xmm3,%%Xmm2\n\t"
    "vfmadd231pd (%%r14,%%r13,1),%%Xmm3,%%Xmm4\n\t"
    "vmovupd %%Xmm0,(%%r14)\n\t"
    "vmovupd %%Xmm1,(%%r14,%%rsi,1)\n\t"
    "vmovupd %%Xmm2,(%%r14,%%rsi,2)\n\t"
    "vmovupd %%Xmm4,(%%r14,%%r13,1)\n\t"

    "jmp 10f\n\t"

    "7:\n\t"

    "cmp $8,%%rdi\n\t"
    "je 9f\n\t"

    "vmovupd %%Ymm4,(%%rcx)\n\t"
    "vmovupd %%Ymm5,32(%%rcx)\n\t"
    "add %%rdi,%%rcx\n\t"

    "vmovupd %%Ymm6,(%%rcx)\n\t"
    "vmovupd %%Ymm7,32(%%rcx)\n\t"
    "add %%rdi,%%rcx\n\t"

    "vmovupd %%Ymm8,(%%rcx)\n\t"
    "vmovupd %%Ymm9,32(%%rcx)\n\t"
    "add %%rdi,%%rcx\n\t"

    "vmovupd %%Ymm10,(%%rcx)\n\t"
    "vmovupd %%Ymm11,32(%%rcx)\n\t"
    "add %%rdi,%%rcx\n\t"

    "vmovupd %%Ymm12,(%%rcx)\n\t"
    "vmovupd %%Ymm13,32(%%rcx)\n\t"
    "add %%rdi,%%rcx\n\t"

    "vmovupd %%Ymm14,(%%rcx)\n\t"
    "vmovupd %%Ymm15,32(%%rcx)\n\t"

    "jmp 10f\n\t"

    "9:\n\t"

    "vunpcklpd %%Ymm6,%%Ymm4,%%Ymm0\n\t"
    "vunpckhpd %%Ymm6,%%Ymm4,%%Ymm1\n\t"
    "vunpcklpd %%Ymm10,%%Ymm8,%%Ymm2\n\t"
    "vunpckhpd %%Ymm10,%%Ymm8,%%Ymm3\n\t"
    "vinsertf128 $0x1,%%Xmm2,%%Ymm0,%%Ymm4\n\t"
    "vinsertf128 $0x1,%%Xmm3,%%Ymm1,%%Ymm6\n\t"
    "vperm2f128 $0x31,%%Ymm2,%%Ymm0,%%Ymm8\n\t"
    "vperm2f128 $0x31,%%Ymm3,%%Ymm1,%%Ymm10\n\t"

    "vmovupd %%Ymm4,(%%rcx)\n\t"
    "vmovupd %%Ymm6,(%%rcx,%%rsi,1)\n\t"
    "vmovupd %%Ymm8,(%%rcx,%%rsi,2)\n\t"
    "vmovupd %%Ymm10,(%%rcx,%%r13,1)\n\t"

    "lea (%%rcx,%%rsi,4),%%rcx\n\t"

    "vunpcklpd %%Ymm14,%%Ymm12,%%Ymm0\n\t"
    "vunpckhpd %%Ymm14,%%Ymm12,%%Ymm1\n\t"
    "vextractf128 $0x1,%%Ymm0,%%Xmm2\n\t"
    "vextractf128 $0x1,%%Ymm1,%%Xmm4\n\t"

    "vmovupd %%Xmm0,(%%r14)\n\t"
    "vmovupd %%Xmm1,(%%r14,%%rsi,1)\n\t"
    "vmovupd %%Xmm2,(%%r14,%%rsi,2)\n\t"
    "vmovupd %%Xmm4,(%%r14,%%r13,1)\n\t"

    "lea (%%r14,%%rsi,4),%%r14\n\t"

    "vunpcklpd %%Ymm7,%%Ymm5,%%Ymm0\n\t"
    "vunpckhpd %%Ymm7,%%Ymm5,%%Ymm1\n\t"
    "vunpcklpd %%Ymm11,%%Ymm9,%%Ymm2\n\t"
    "vunpckhpd %%Ymm11,%%Ymm9,%%Ymm3\n\t"
    "vinsertf128 $0x1,%%Xmm2,%%Ymm0,%%Ymm5\n\t"
    "vinsertf128 $0x1,%%Xmm3,%%Ymm1,%%Ymm7\n\t"
    "vperm2f128 $0x31,%%Ymm2,%%Ymm0,%%Ymm9\n\t"
    "vperm2f128 $0x31,%%Ymm3,%%Ymm1,%%Ymm11\n\t"

    "vmovupd %%Ymm5,(%%rcx)\n\t"
    "vmovupd %%Ymm7,(%%rcx,%%rsi,1)\n\t"
    "vmovupd %%Ymm9,(%%rcx,%%rsi,2)\n\t"
    "vmovupd %%Ymm11,(%%rcx,%%r13,1)\n\t"

    "vunpcklpd %%Ymm15,%%Ymm13,%%Ymm0\n\t"
    "vunpckhpd %%Ymm15,%%Ymm13,%%Ymm1\n\t"
    "vextractf128 $0x1,%%Ymm0,%%Xmm2\n\t"
    "vextractf128 $0x1,%%Ymm1,%%Xmm4\n\t"

    "vmovupd %%Xmm0,(%%r14)\n\t"
    "vmovupd %%Xmm1,(%%r14,%%rsi,1)\n\t"
    "vmovupd %%Xmm2,(%%r14,%%rsi,2)\n\t"
    "vmovupd %%Xmm4,(%%r14,%%r13,1)\n\t"

    "10:\n\t"

    "vzeroupper \n\t"
    :
    :   [k_iter] "m"(k_iter),
        [k_left] "m"(k_left),
        [a] "m"(A),
        [b] "m"(B),
        [alpha] "m"(alpha_),
        [beta] "m"(beta_),
        [c] "m"(C),
        [rs_c] "m"(rs_c),
        [cs_c] "m"(cs_c)
    :   "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r13", "r14",
        "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
        "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12",
        "xmm13", "xmm14", "xmm15", "memory"
    );
}

} } // namespace avx, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_AVX_UGEMM_TCC
