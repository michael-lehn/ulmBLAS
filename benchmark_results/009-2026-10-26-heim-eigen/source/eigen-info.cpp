/*
 *  eigen-info: configuration of Eigen as compiled with the flags of
 *  bench-eigen (version, SIMD instruction sets, threads, product
 *  thresholds and blocking sizes of the matrix product for some sizes).
 */
#include <Eigen/Dense>
#include <cstdio>
#include <initializer_list>

int
main()
{
    printf("Eigen %s\n", EIGEN_VERSION_STRING);
    // lists AVX, not AVX2 or FMA; see the EIGEN_VECTORIZE_* macros below
    printf("SIMD instruction sets in use: %s\n",
           Eigen::SimdInstructionSetsInUse());
#ifdef EIGEN_VECTORIZE_AVX2
    printf("EIGEN_VECTORIZE_AVX2 defined\n");
#endif
#ifdef EIGEN_VECTORIZE_FMA
    printf("EIGEN_VECTORIZE_FMA defined\n");
#endif
#ifdef EIGEN_VECTORIZE_AVX512
    printf("EIGEN_VECTORIZE_AVX512 defined\n");
#endif
#ifdef EIGEN_USE_BLAS
    printf("EIGEN_USE_BLAS defined\n");
#else
    printf("EIGEN_USE_BLAS not defined (Eigen's own product kernels)\n");
#endif
#ifdef _OPENMP
    printf("OpenMP enabled\n");
#else
    printf("OpenMP not enabled\n");
#endif
    printf("Eigen::nbThreads() = %d\n", Eigen::nbThreads());
    printf("packet size (double) = %d\n",
           (int)Eigen::internal::packet_traits<double>::size);
    printf("EIGEN_GEMM_TO_COEFFBASED_THRESHOLD = %d (lazy coefficient-based "
           "product if m+n+k < threshold)\n",
           EIGEN_GEMM_TO_COEFFBASED_THRESHOLD);
    printf("EIGEN_STACK_ALLOCATION_LIMIT = %d bytes\n",
           EIGEN_STACK_ALLOCATION_LIMIT);
    std::ptrdiff_t l1, l2, l3;
    Eigen::internal::manage_caching_sizes(Eigen::GetAction, &l1, &l2, &l3);
    printf("cache sizes used for blocking: L1 %td, L2 %td, L3 %td bytes\n",
           l1, l2, l3);
    for (int n : {8, 64, 256, 1024, 4096}) {
        Eigen::Index kc = n, mc = n, nc = n;
        Eigen::internal::computeProductBlockingSizes<double, double>(kc, mc,
                                                                     nc,
                                                                     Eigen::Index(1));
        printf("blocking for n = %4d: kc %td, mc %td, nc %td\n", n,
               (std::ptrdiff_t)kc, (std::ptrdiff_t)mc, (std::ptrdiff_t)nc);
    }
    return 0;
}
