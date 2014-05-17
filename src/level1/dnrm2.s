	.text
	.align 4,0x90
	.globl _dnrm2_
_dnrm2_:
LFB19:
	movl	(%rdi), %ecx
	movl	(%rdx), %edx
	testl	%ecx, %ecx
	jle	L12
	testl	%edx, %edx
	jle	L12
	cmpl	$1, %ecx
	je	L3
	movsd	LC0(%rip), %xmm6
	subl	$1, %ecx
	movslq	%edx, %rdi
	xorpd	%xmm0, %xmm0
	imull	%edx, %ecx
	salq	$3, %rdi
	movapd	%xmm6, %xmm2
	movapd	%xmm0, %xmm3
	xorl	%eax, %eax
	movapd	%xmm0, %xmm5
	movsd	LC2(%rip), %xmm4
	jmp	L4
	.align 4,0x90
L21:
	divsd	%xmm1, %xmm3
	mulsd	%xmm3, %xmm3
	mulsd	%xmm3, %xmm2
	movapd	%xmm1, %xmm3
	addsd	%xmm6, %xmm2
L5:
	addl	%edx, %eax
	addq	%rdi, %rsi
	cmpl	%ecx, %eax
	jg	L20
L4:
	movsd	(%rsi), %xmm1
	ucomisd	%xmm0, %xmm1
	jp	L13
	ucomisd	%xmm5, %xmm1
	je	L5
L13:
	andpd	%xmm4, %xmm1
	ucomisd	%xmm3, %xmm1
	ja	L21
	divsd	%xmm3, %xmm1
	addl	%edx, %eax
	addq	%rdi, %rsi
	cmpl	%ecx, %eax
	mulsd	%xmm1, %xmm1
	addsd	%xmm1, %xmm2
	jle	L4
L20:
	sqrtsd	%xmm2, %xmm0
	ucomisd	%xmm0, %xmm0
	jp	L22
	mulsd	%xmm3, %xmm0
	ret
	.align 4,0x90
L12:
	xorpd	%xmm0, %xmm0
	ret
	.align 4,0x90
L3:
	movsd	(%rsi), %xmm0
	movsd	LC2(%rip), %xmm1
	andpd	%xmm1, %xmm0
	ret
L22:
	subq	$24, %rsp
LCFI0:
	movapd	%xmm2, %xmm0
	movsd	%xmm3, 8(%rsp)
	call	_sqrt
	movsd	8(%rsp), %xmm3
	addq	$24, %rsp
LCFI1:
	mulsd	%xmm3, %xmm0
	ret
LFE19:
	.literal8
	.align 3
LC0:
	.long	0
	.long	1072693248
	.literal16
	.align 4
LC2:
	.long	4294967295
	.long	2147483647
	.long	0
	.long	0
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$0,LECIE1-LSCIE1
	.long L$set$0
LSCIE1:
	.long	0
	.byte	0x1
	.ascii "zR\0"
	.byte	0x1
	.byte	0x78
	.byte	0x10
	.byte	0x1
	.byte	0x10
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x90
	.byte	0x1
	.align 3
LECIE1:
LSFDE1:
	.set L$set$1,LEFDE1-LASFDE1
	.long L$set$1
LASFDE1:
	.long	LASFDE1-EH_frame1
	.quad	LFB19-.
	.set L$set$2,LFE19-LFB19
	.quad L$set$2
	.byte	0
	.byte	0x4
	.set L$set$3,LCFI0-LFB19
	.long L$set$3
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE1:
	.subsections_via_symbols
