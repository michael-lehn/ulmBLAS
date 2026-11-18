#ifndef ULMBLAS_LEVEL3_GEMM_PLAN_H
#define ULMBLAS_LEVEL3_GEMM_PLAN_H 1

#include <cstddef>
#include <limits>

//
//  The types of the run-time selection of gemm (ulmblas/level3/gemm/
//  select.h): the problem, the plan and an entry of a selection table.
//
namespace ulmBLAS {

//
//  A problem C <- beta C + alpha op(A) op(B), C m x n, k the inner
//  dimension; op(X) = X or conj(X):
//
//      m, n, k            dimensions
//      incRowX, incColX   strides of A, B and C (arbitrary, also negative
//                         or zero for A and B)
//      conjA, conjB       conjugation of A and B
//      threads            number of threads that may be used (so far
//                         always 1: gemm is not parallel)
//
template <typename IndexType> struct GemmProblem
{
    IndexType m, n, k;
    IndexType incRowA, incColA;
    IndexType incRowB, incColB;
    IndexType incRowC, incColC;
    bool conjA, conjB;
    std::size_t threads;
};

//
//  A plan: the function that computes the problem (the same arguments as
//  gemm) and a name for tests and benchmarks.
//
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
struct GemmPlan
{
    typedef void (*Run)(IndexType m, IndexType n, IndexType k,
                        const Alpha &alpha, bool conjA, const TA *A,
                        IndexType incRowA, IndexType incColA, bool conjB,
                        const TB *B, IndexType incRowB, IndexType incColB,
                        const Beta &beta, TC *C, IndexType incRowC,
                        IndexType incColC);

    Run run;
    const char *name;
};

//
//  An entry of a selection table: the plan for the problems with
//
//      minM <= m <= maxM,  minN <= n <= maxN,  minK <= k <= maxK,
//      m n sizeC >= minBytesC  (bytes of C)  and
//      m k sizeA <= maxBytesA  (bytes of A)
//
//  GemmAll and GemmBytesAll (the largest values) as upper bounds: no limit.
//
//  The tables are generic over the operation (Op below): the entries of
//  gemm, gemmt, ... with the same kernels and block sizes, Plan the plan of
//  the operation.
//
constexpr std::ptrdiff_t GemmAll = std::numeric_limits<std::ptrdiff_t>::max();
constexpr std::size_t GemmBytesAll = std::numeric_limits<std::size_t>::max();

template <typename Plan> struct Level3Entry
{
    std::ptrdiff_t minM, maxM;
    std::ptrdiff_t minN, maxN;
    std::ptrdiff_t minK, maxK;
    std::size_t minBytesC;
    std::size_t maxBytesA;

    Plan plan;
};

template <typename Plan>
bool
level3EntryApplies(const Level3Entry<Plan> &entry, std::ptrdiff_t m,
                   std::ptrdiff_t n, std::ptrdiff_t k, std::size_t sizeC,
                   std::size_t sizeA)
{
    return entry.minM <= m && m <= entry.maxM && entry.minN <= n &&
           n <= entry.maxN && entry.minK <= k && k <= entry.maxK &&
           std::size_t(m) * std::size_t(n) * sizeC >= entry.minBytesC &&
           std::size_t(m) * std::size_t(k) * sizeA <= entry.maxBytesA;
}

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
using GemmEntry = Level3Entry<GemmPlan<IndexType, Alpha, TA, TB, Beta, TC>>;

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
bool
gemmEntryApplies(const GemmEntry<IndexType, Alpha, TA, TB, Beta, TC> &entry,
                 const GemmProblem<IndexType> &problem)
{
    return level3EntryApplies(
        entry, std::ptrdiff_t(problem.m), std::ptrdiff_t(problem.n),
        std::ptrdiff_t(problem.k), sizeof(TC), sizeof(TA));
}

//
//  An operation of the selection tables: the plan of an entry from the
//  implementation the entry names, F (a frame of gemm, GemmFrame, or a
//  kernel for small matrices with a frame as fallback); sizeC, sizeA the
//  sizes of the elements of C and A for the bounds in bytes.
//
//  GemmOp: gemm, the plan is F::run.
//
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
struct GemmOp
{
    typedef GemmPlan<IndexType, Alpha, TA, TB, Beta, TC> Plan;

    static constexpr std::size_t sizeC = sizeof(TC);
    static constexpr std::size_t sizeA = sizeof(TA);

    template <typename F>
    static Plan
    plan(const char *name)
    {
	Plan p = {&F::template run<IndexType, Alpha, TA, TB, Beta, TC>, name};
	return p;
    }
};

//
//  The frame of F: F if it is a frame, the frame that a kernel for small
//  matrices falls back to (its FallbackType, recursively) otherwise; for
//  the operations that have no kernel for small matrices.
//
template <typename> struct Level3Void
{
    typedef void type;
};

template <typename F, typename = void> struct GemmFrameOf
{
    typedef F type;
};

template <typename F>
struct GemmFrameOf<F, typename Level3Void<typename F::FallbackType>::type>
{
    typedef typename GemmFrameOf<typename F::FallbackType>::type type;
};

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_PLAN_H
