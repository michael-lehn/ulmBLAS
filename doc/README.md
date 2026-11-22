# ulmBLAS guides

Guides to the concepts behind ulmBLAS, one document per group of
concepts.  Each guide explains its concepts on small, complete examples
and then shows how ulmBLAS applies them, with excerpts of its code.  They
are written for the course
[High Performance Computing I](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/)
at Ulm University and for students who continue in this field.

| Guide | Contents | Examples | Based on ulmBLAS |
|---|---|---|---|
| [From C to C++: what is relevant in ulmBLAS](guides/cpp_in_ulmblas.pdf) | The C++ features ulmBLAS uses (function templates, namespaces and the C interface, references, complex numbers, computing with types, casts, constructors and destructors), and those it does not use, and why | [examples](guides/cpp_in_ulmblas/examples) | release 0.4.2 |
| GEMM (general matrix-matrix product) and the other level 3 functions | | | planned |
| [The LU (lower-upper) and Cholesky factorizations in ulmBLAS](guides/lu_and_cholesky.pdf) | From the definition to the implementation: the variants, recursion, where the time goes (profiling with `perf`), the base cases, task graphs, solving and inverting, LU with lookahead, choosing the algorithm, testing | [examples](guides/lu_and_cholesky/examples) | release 0.4.2 |
| [Vectorization and multithreading with OpenMP (Open Multi-Processing) and C++ atomics](guides/vectorization_and_multithreading.pdf) | `omp simd` and reductions; fork–join with atomic flags, tasks, the memory model of C++, barriers | [examples](guides/vectorization_and_multithreading/examples) | release 0.4.2 |

The order is the suggested order of reading.

The examples are complete programs; `examples/run.sh` compiles and runs
them with two compilers and writes the outputs shown in the guide to
`examples/out`.
