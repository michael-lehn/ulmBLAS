# GEMM micro kernels

The micro kernels of ulmBLAS.  The kernel is selected at compile time from
the target of the compiler (`ulmblas/config/simd.h`, `ugemm.h`); all level 3
routines of a data type are built on its GEMM micro kernel.  `make check_arch`
shows what is selected on a machine.

Kernels for AVX, AVX-512, NEON and SME2 and the kernels vendored from BLIS,
AOCL-BLAS and OpenBLAS were removed for the time being, see
[CLEANUP_REPORT.md](../../../CLEANUP_REPORT.md).  Their documentation is at
[6c554eec](https://github.com/michael-lehn/ulmBLAS/tree/6c554eec707e22f118bae5dcaea10ad9945aac54/ulmblas/level3/ukernel/README.md).

## Portable reference kernel (all types)

`ref/`: portable C++ with the block sizes of `ulmblas/config/blocksize.h`
(MR x NR = 8 x 4 for float, 4 x 4 for double, 4 x 2 for complex float, 2 x 2
for complex double).  Used on every target without an optimized kernel, for
all data types without one, and by the test configurations with small block
sizes.

## SSE3 kernels (x86-64, double)

`sse/`: inline assembly for GEMM (`ugemm.tcc`, 4 x 4) and TRSM (`utrlsm.tcc`,
`utrusm.tcc`, which require MR = NR = 4; checked with a `static_assert`).
Selected by default on x86-64 if the compiler targets SSE3.  Tested with the
default block sizes by the configuration `testcase04`, with `-mavx -mfma` by
`testcase06` and with small block sizes by `testcase07`.
