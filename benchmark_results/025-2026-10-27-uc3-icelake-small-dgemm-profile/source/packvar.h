//
//  packvar.h: variants of the packing routines of ulmBLAS (gepack_A,
//  gepack_B in ulmblas/level3/pack/gepack.tcc) for the experiments of
//  prof.cc.  Same interface and same packed format (A: panels of MR rows,
//  column by column; B: panels of NR columns, row by row; partial panels
//  padded with zeros); no conjugation (double).
//
//  A variants:
//    ulm   ulmBLAS::gepack_A
//    unit  the traversal of ulmBLAS (panel by panel, in each panel column
//          by column), with a separate loop for incRowA == 1, so that the
//          compiler knows that the MR elements are contiguous
//    col   column by column of the whole block (for each column all
//          panels): contiguous reads of mc elements for incRowA == 1
//    blkG  (G = 2, 4, 8, 16) groups of G panels, each group column by
//          column: contiguous reads of G*MR elements per column, G
//          panels written at the same time
//    gblkG (G = 4, 8) as blkG, but one loop for all strides (no separate
//          loop for incRowA == 1)
//  B variants:
//    ulm   ulmBLAS::gepack_B
//    unit  the traversal of ulmBLAS with a separate loop for incColB == 1
//          (row major B) and incRowB == 1 (column major B)
//    two   column major B: portable C++, two rows at a time (two
//          consecutive elements of each column)
//    trn   column major B: NEON, two rows at a time with 2 x 2 transposes
//          in registers (as bli_dpackm_armv8a_int_6x8 of BLIS 2.0)
//
#ifndef PACKVAR_H
#define PACKVAR_H 1

#include <string>
#include <ulmblas/level3/level3.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace packvar {

typedef long Index;
static const Index MR = ulmBLAS::BlockSizeUGemm<double>::MR;
static const Index NR = ulmBLAS::BlockSizeUGemm<double>::NR;

//  partial panels (as ulmBLAS, but without the call of bzero per column)
static inline void
packA_edge(Index mr, Index kc, const double *A, Index incRowA, Index incColA,
           double *buffer)
{
    for (Index j=0; j<kc; ++j) {
        for (Index i=0; i<mr; ++i) buffer[i] = A[i*incRowA];
        for (Index i=mr; i<MR; ++i) buffer[i] = 0;
        buffer += MR;
        A      += incColA;
    }
}

static inline void
packB_edge(Index kc, Index nr, const double *B, Index incRowB, Index incColB,
           double *buffer)
{
    for (Index i=0; i<kc; ++i) {
        for (Index j=0; j<nr; ++j) buffer[j] = B[j*incColB];
        for (Index j=nr; j<NR; ++j) buffer[j] = 0;
        buffer += NR;
        B      += incRowB;
    }
}

static void
packA_unit(Index mc, Index kc, const double *A, Index incRowA,
           Index incColA, double *buffer)
{
    const Index mp = mc/MR, mr_ = mc%MR;
    for (Index p=0; p<mp; ++p) {
        if (incRowA==1) {
            for (Index j=0; j<kc; ++j) {
                for (Index i=0; i<MR; ++i) buffer[i] = A[i];
                buffer += MR;
                A      += incColA;
            }
        } else {
            for (Index j=0; j<kc; ++j) {
                for (Index i=0; i<MR; ++i) buffer[i] = A[i*incRowA];
                buffer += MR;
                A      += incColA;
            }
        }
        A += MR*incRowA - kc*incColA;
    }
    if (mr_>0) packA_edge(mr_, kc, A, incRowA, incColA, buffer);
}

static void
packA_col(Index mc, Index kc, const double *A, Index incRowA,
          Index incColA, double *buffer)
{
    const Index mp = mc/MR, mr_ = mc%MR;
    if (incRowA!=1) {
        packA_unit(mc, kc, A, incRowA, incColA, buffer);
        return;
    }
    for (Index j=0; j<kc; ++j) {
        const double *a = &A[j*incColA];
        double       *b = &buffer[j*MR];
        for (Index p=0; p<mp; ++p) {
            for (Index i=0; i<MR; ++i) b[i] = a[i];
            a += MR;
            b += kc*MR;
        }
    }
    if (mr_>0) packA_edge(mr_, kc, &A[mp*MR], 1, incColA, &buffer[mp*kc*MR]);
}

template <Index G>
static void
packA_blk(Index mc, Index kc, const double *A, Index incRowA,
          Index incColA, double *buffer)
{
    const Index mp = mc/MR, mr_ = mc%MR;
    if (incRowA!=1) {
        packA_unit(mc, kc, A, incRowA, incColA, buffer);
        return;
    }
    Index p = 0;
    for (; p+G<=mp; p+=G) {
        const double *a = &A[p*MR];
        double       *b = &buffer[p*kc*MR];
        for (Index j=0; j<kc; ++j) {
            for (Index g=0; g<G; ++g) {
                for (Index i=0; i<MR; ++i) b[g*kc*MR+i] = a[g*MR+i];
            }
            a += incColA;
            b += MR;
        }
    }
    for (; p<mp; ++p) {
        const double *a = &A[p*MR];
        double       *b = &buffer[p*kc*MR];
        for (Index j=0; j<kc; ++j) {
            for (Index i=0; i<MR; ++i) b[i] = a[i];
            a += incColA;
            b += MR;
        }
    }
    if (mr_>0) packA_edge(mr_, kc, &A[mp*MR], 1, incColA, &buffer[mp*kc*MR]);
}

template <Index G>
static void
packA_gblk(Index mc, Index kc, const double *A, Index incRowA,
           Index incColA, double *buffer)
{
    const Index mp = mc/MR, mr_ = mc%MR;
    Index p = 0;
    for (; p+G<=mp; p+=G) {
        const double *a = &A[p*MR*incRowA];
        double       *b = &buffer[p*kc*MR];
        for (Index j=0; j<kc; ++j) {
            for (Index g=0; g<G; ++g) {
                for (Index i=0; i<MR; ++i) {
                    b[g*kc*MR+i] = a[(g*MR+i)*incRowA];
                }
            }
            a += incColA;
            b += MR;
        }
    }
    for (; p<mp; ++p) {
        const double *a = &A[p*MR*incRowA];
        double       *b = &buffer[p*kc*MR];
        for (Index j=0; j<kc; ++j) {
            for (Index i=0; i<MR; ++i) b[i] = a[i*incRowA];
            a += incColA;
            b += MR;
        }
    }
    if (mr_>0) {
        packA_edge(mr_, kc, &A[mp*MR*incRowA], incRowA, incColA,
                   &buffer[mp*kc*MR]);
    }
}

static void
packB_unit(Index kc, Index nc, const double *B, Index incRowB,
           Index incColB, double *buffer)
{
    const Index np = nc/NR, nr_ = nc%NR;
    for (Index q=0; q<np; ++q) {
        if (incColB==1) {
            for (Index i=0; i<kc; ++i) {
                for (Index j=0; j<NR; ++j) buffer[j] = B[j];
                buffer += NR;
                B      += incRowB;
            }
        } else if (incRowB==1) {
            for (Index i=0; i<kc; ++i) {
                for (Index j=0; j<NR; ++j) buffer[j] = B[i+j*incColB];
                buffer += NR;
            }
            B += kc;
        } else {
            for (Index i=0; i<kc; ++i) {
                for (Index j=0; j<NR; ++j) buffer[j] = B[j*incColB];
                buffer += NR;
                B      += incRowB;
            }
        }
        B += NR*incColB - kc*incRowB;
    }
    if (nr_>0) packB_edge(kc, nr_, B, incRowB, incColB, buffer);
}

static void
packB_two(Index kc, Index nc, const double *B, Index incRowB,
          Index incColB, double *buffer)
{
    const Index np = nc/NR, nr_ = nc%NR;
    if (incRowB!=1) {
        packB_unit(kc, nc, B, incRowB, incColB, buffer);
        return;
    }
    const Index ld = incColB;
    for (Index q=0; q<np; ++q) {
        const double *b = &B[q*NR*ld];
        Index i = 0;
        for (; i+1<kc; i+=2) {
            for (Index j=0; j<NR; ++j) {
                buffer[j]    = b[i+j*ld];
                buffer[NR+j] = b[i+1+j*ld];
            }
            buffer += 2*NR;
        }
        for (; i<kc; ++i) {
            for (Index j=0; j<NR; ++j) buffer[j] = b[i+j*ld];
            buffer += NR;
        }
    }
    if (nr_>0) packB_edge(kc, nr_, &B[np*NR*ld], 1, ld, buffer);
}

#if defined(__ARM_NEON)
static void
packB_trn(Index kc, Index nc, const double *B, Index incRowB,
          Index incColB, double *buffer)
{
    const Index np = nc/NR, nr_ = nc%NR;
    if (incRowB!=1 || NR!=8) {
        packB_unit(kc, nc, B, incRowB, incColB, buffer);
        return;
    }
    const Index ld = incColB;
    for (Index q=0; q<np; ++q) {
        const double *b = &B[q*NR*ld];
        Index i = 0;
        for (; i+1<kc; i+=2) {
            float64x2_t v0 = vld1q_f64(b+i+0*ld), v1 = vld1q_f64(b+i+1*ld),
                        v2 = vld1q_f64(b+i+2*ld), v3 = vld1q_f64(b+i+3*ld),
                        v4 = vld1q_f64(b+i+4*ld), v5 = vld1q_f64(b+i+5*ld),
                        v6 = vld1q_f64(b+i+6*ld), v7 = vld1q_f64(b+i+7*ld);
            vst1q_f64(buffer+0,  vtrn1q_f64(v0, v1));
            vst1q_f64(buffer+2,  vtrn1q_f64(v2, v3));
            vst1q_f64(buffer+4,  vtrn1q_f64(v4, v5));
            vst1q_f64(buffer+6,  vtrn1q_f64(v6, v7));
            vst1q_f64(buffer+8,  vtrn2q_f64(v0, v1));
            vst1q_f64(buffer+10, vtrn2q_f64(v2, v3));
            vst1q_f64(buffer+12, vtrn2q_f64(v4, v5));
            vst1q_f64(buffer+14, vtrn2q_f64(v6, v7));
            buffer += 2*NR;
        }
        for (; i<kc; ++i) {
            for (Index j=0; j<NR; ++j) buffer[j] = b[i+j*ld];
            buffer += NR;
        }
    }
    if (nr_>0) packB_edge(kc, nr_, &B[np*NR*ld], 1, ld, buffer);
}
#endif

typedef void (*PackA)(Index, Index, const double *, Index, Index, double *);
typedef void (*PackB)(Index, Index, const double *, Index, Index, double *);

static void
packA_ulm(Index mc, Index kc, const double *A, Index incRowA, Index incColA,
          double *buffer)
{
    ulmBLAS::gepack_A(mc, kc, false, A, incRowA, incColA, buffer);
}

static void
packB_ulm(Index kc, Index nc, const double *B, Index incRowB, Index incColB,
          double *buffer)
{
    ulmBLAS::gepack_B(kc, nc, false, B, incRowB, incColB, buffer);
}

static PackA
selectA(const char *name)
{
    std::string s = name;
    if (s=="ulm")  return packA_ulm;
    if (s=="unit") return packA_unit;
    if (s=="col")  return packA_col;
    if (s=="blk2") return packA_blk<2>;
    if (s=="blk4") return packA_blk<4>;
    if (s=="blk8") return packA_blk<8>;
    if (s=="blk16") return packA_blk<16>;
    if (s=="gblk4") return packA_gblk<4>;
    if (s=="gblk8") return packA_gblk<8>;
    return nullptr;
}

static PackB
selectB(const char *name)
{
    std::string s = name;
    if (s=="ulm")  return packB_ulm;
    if (s=="unit") return packB_unit;
    if (s=="two")  return packB_two;
#if defined(__ARM_NEON)
    if (s=="trn")  return packB_trn;
#endif
    return nullptr;
}

} // namespace packvar

#endif // PACKVAR_H
