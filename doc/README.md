# ulmBLAS guides

Guides to the concepts behind ulmBLAS, one document per group of
concepts.  Each guide explains its concepts on small, complete examples
and then shows how ulmBLAS applies them, with excerpts of its code.  They
are written for the course
[High Performance Computing I](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/)
at Ulm University and for students who continue in this field.

| Guide | Examples | Based on ulmBLAS |
|---|---|---|
| [From C to C++: what is relevant in ulmBLAS](guides/cpp_in_ulmblas.pdf) | [examples](guides/cpp_in_ulmblas/examples) | release 0.4.2 |
| GEMM (general matrix-matrix product) and the other level 3 functions | | planned |
| LU (lower-upper) and Cholesky factorizations (`getrf`, `potrf`) | | in progress |
| [Vectorization and multithreading with OpenMP (Open Multi-Processing) and C++ atomics](guides/vectorization_and_multithreading.pdf) | [examples](guides/vectorization_and_multithreading/examples) | release 0.4.2 |

The order is the suggested order of reading.

The examples are complete programs; `examples/run.sh` compiles and runs
them with two compilers and writes the outputs shown in the guide to
`examples/out`.
