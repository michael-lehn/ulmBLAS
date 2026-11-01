/*  int xmmcall(void (*f)(void)): calls f with known values in xmm6-xmm15
    (callee-saved in the Windows x64 ABI) and returns the number of these
    registers f did not preserve.  Windows x64, GNU assembler. */
        .text
        .globl  xmmcall
        .def    xmmcall; .scl 2; .type 32; .endef
xmmcall:
        pushq   %rbx
        pushq   %r12
        subq    $200, %rsp
        movdqu  %xmm6, 32(%rsp)
        movdqu  %xmm7, 48(%rsp)
        movdqu  %xmm8, 64(%rsp)
        movdqu  %xmm9, 80(%rsp)
        movdqu  %xmm10, 96(%rsp)
        movdqu  %xmm11, 112(%rsp)
        movdqu  %xmm12, 128(%rsp)
        movdqu  %xmm13, 144(%rsp)
        movdqu  %xmm14, 160(%rsp)
        movdqu  %xmm15, 176(%rsp)
        movq    %rcx, %r12
        movabsq $0x0606060606060606, %rax
        movq    %rax, %xmm6
        punpcklqdq %xmm6, %xmm6
        movabsq $0x0707070707070707, %rax
        movq    %rax, %xmm7
        punpcklqdq %xmm7, %xmm7
        movabsq $0x0808080808080808, %rax
        movq    %rax, %xmm8
        punpcklqdq %xmm8, %xmm8
        movabsq $0x0909090909090909, %rax
        movq    %rax, %xmm9
        punpcklqdq %xmm9, %xmm9
        movabsq $0x0a0a0a0a0a0a0a0a, %rax
        movq    %rax, %xmm10
        punpcklqdq %xmm10, %xmm10
        movabsq $0x0b0b0b0b0b0b0b0b, %rax
        movq    %rax, %xmm11
        punpcklqdq %xmm11, %xmm11
        movabsq $0x0c0c0c0c0c0c0c0c, %rax
        movq    %rax, %xmm12
        punpcklqdq %xmm12, %xmm12
        movabsq $0x0d0d0d0d0d0d0d0d, %rax
        movq    %rax, %xmm13
        punpcklqdq %xmm13, %xmm13
        movabsq $0x0e0e0e0e0e0e0e0e, %rax
        movq    %rax, %xmm14
        punpcklqdq %xmm14, %xmm14
        movabsq $0x0f0f0f0f0f0f0f0f, %rax
        movq    %rax, %xmm15
        punpcklqdq %xmm15, %xmm15
        call    *%r12
        xorl    %ebx, %ebx
        movabsq $0x0606060606060606, %rax
        movq    %rax, %xmm0
        punpcklqdq %xmm0, %xmm0
        pcmpeqb %xmm6, %xmm0
        pmovmskb %xmm0, %eax
        cmpl    $0xffff, %eax
        je      16f
        incl    %ebx
16:
        movabsq $0x0707070707070707, %rax
        movq    %rax, %xmm0
        punpcklqdq %xmm0, %xmm0
        pcmpeqb %xmm7, %xmm0
        pmovmskb %xmm0, %eax
        cmpl    $0xffff, %eax
        je      17f
        incl    %ebx
17:
        movabsq $0x0808080808080808, %rax
        movq    %rax, %xmm0
        punpcklqdq %xmm0, %xmm0
        pcmpeqb %xmm8, %xmm0
        pmovmskb %xmm0, %eax
        cmpl    $0xffff, %eax
        je      18f
        incl    %ebx
18:
        movabsq $0x0909090909090909, %rax
        movq    %rax, %xmm0
        punpcklqdq %xmm0, %xmm0
        pcmpeqb %xmm9, %xmm0
        pmovmskb %xmm0, %eax
        cmpl    $0xffff, %eax
        je      19f
        incl    %ebx
19:
        movabsq $0x0a0a0a0a0a0a0a0a, %rax
        movq    %rax, %xmm0
        punpcklqdq %xmm0, %xmm0
        pcmpeqb %xmm10, %xmm0
        pmovmskb %xmm0, %eax
        cmpl    $0xffff, %eax
        je      110f
        incl    %ebx
110:
        movabsq $0x0b0b0b0b0b0b0b0b, %rax
        movq    %rax, %xmm0
        punpcklqdq %xmm0, %xmm0
        pcmpeqb %xmm11, %xmm0
        pmovmskb %xmm0, %eax
        cmpl    $0xffff, %eax
        je      111f
        incl    %ebx
111:
        movabsq $0x0c0c0c0c0c0c0c0c, %rax
        movq    %rax, %xmm0
        punpcklqdq %xmm0, %xmm0
        pcmpeqb %xmm12, %xmm0
        pmovmskb %xmm0, %eax
        cmpl    $0xffff, %eax
        je      112f
        incl    %ebx
112:
        movabsq $0x0d0d0d0d0d0d0d0d, %rax
        movq    %rax, %xmm0
        punpcklqdq %xmm0, %xmm0
        pcmpeqb %xmm13, %xmm0
        pmovmskb %xmm0, %eax
        cmpl    $0xffff, %eax
        je      113f
        incl    %ebx
113:
        movabsq $0x0e0e0e0e0e0e0e0e, %rax
        movq    %rax, %xmm0
        punpcklqdq %xmm0, %xmm0
        pcmpeqb %xmm14, %xmm0
        pmovmskb %xmm0, %eax
        cmpl    $0xffff, %eax
        je      114f
        incl    %ebx
114:
        movabsq $0x0f0f0f0f0f0f0f0f, %rax
        movq    %rax, %xmm0
        punpcklqdq %xmm0, %xmm0
        pcmpeqb %xmm15, %xmm0
        pmovmskb %xmm0, %eax
        cmpl    $0xffff, %eax
        je      115f
        incl    %ebx
115:
        movl    %ebx, %eax
        movdqu  32(%rsp), %xmm6
        movdqu  48(%rsp), %xmm7
        movdqu  64(%rsp), %xmm8
        movdqu  80(%rsp), %xmm9
        movdqu  96(%rsp), %xmm10
        movdqu  112(%rsp), %xmm11
        movdqu  128(%rsp), %xmm12
        movdqu  144(%rsp), %xmm13
        movdqu  160(%rsp), %xmm14
        movdqu  176(%rsp), %xmm15
        addq    $200, %rsp
        popq    %r12
        popq    %rbx
        ret
