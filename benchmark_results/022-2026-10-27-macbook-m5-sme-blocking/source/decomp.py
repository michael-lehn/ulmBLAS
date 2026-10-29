#!/usr/bin/env python3
#
#  decomp.py: the block decomposition of ulmBLAS gemm (ulmblas/level3/
#  gemm.tcc, mkernel/mgemm.tcc) for a shape m x n x k and block sizes MC,
#  KC, NC (MR = 32, NR = 16 of the SME2 kernel).  gemm splits m, n and k
#  into full blocks and one remainder block at the end (no balancing):
#
#    for j (n in blocks of NC): for l (k in blocks of KC):
#        pack B (kc x nc); for i (m in blocks of MC): pack A (mc x kc),
#        macro kernel: tiles of MR x NR, one micro kernel call per tile
#
#  Every micro kernel call loads its tile of C into ZA and stores it back,
#  once per k block.
#
#  As a module: decomp(m, n, k, MC, KC, NC) returns a dict; as a program:
#  decomp.py M_N_K MC_KC_NC ... prints it.
#
import sys

MR, NR = 32, 16

def blocks(x, b):
    q, r = divmod(x, b)
    return [b]*q + ([r] if r else [])

def decomp(m, n, k, MC, KC, NC):
    bm, bn, bk = blocks(m, MC), blocks(n, NC), blocks(k, KC)
    tm = sum((x+MR-1)//MR for x in bm)          # tile rows
    tn = sum((x+NR-1)//NR for x in bn)          # tile columns
    calls = len(bk)*tm*tn
    partial = len(bk)*(tm*tn - sum(x//MR for x in bm)*sum(x//NR for x in bn))
    return dict(
        m_blocks=' '.join(map(str, bm)), n_blocks=' '.join(map(str, bn)),
        k_blocks=' '.join(map(str, bk)),
        mb=len(bm), nb=len(bn), kb=len(bk),
        mc_last=bm[-1], nc_last=bn[-1], kc_last=bk[-1],
        ukernel_calls=calls, partial_tile_calls=partial,
        # elements packed: A is packed once per n block, B once
        packA=len(bn)*m*k, packB=n*k,
        # elements of C moved through ZA (load and store each count once)
        c_through_za=len(bk)*m*n,
        # tile work (multiply-adds incl. zero padding) / useful work
        padding=(tm*MR)*(tn*NR)/(m*n),
    )

if __name__ == '__main__':
    shape = [int(x) for x in sys.argv[1].split('_')]
    for cfg in sys.argv[2:]:
        MC, KC, NC = [int(x) for x in cfg.split('_')]
        print(cfg, decomp(*shape, MC, KC, NC))
