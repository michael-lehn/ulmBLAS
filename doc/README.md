# ulmBLAS guides

Guides to the concepts behind ulmBLAS, one document per group of
concepts.  Each guide explains its concepts on small, complete examples
and then shows how ulmBLAS applies them, with excerpts of its code.  They
are written for the course
[High Performance Computing I](https://www.mathematik.uni-ulm.de/numerik/hpc/ws25/)
at Ulm University and for students who continue in this field.

| Guide | Examples | Based on ulmBLAS |
|---|---|---|
| [Vectorization and multithreading with OpenMP (Open Multi-Processing) and C++ atomics](guides/vectorization_and_multithreading.pdf) | [examples](guides/vectorization_and_multithreading/examples) | release 0.4.2 |

More guides will follow, for example on the algorithms of the
factorizations (`getrf`, `potrf`, ...).

The examples are complete programs; `examples/run.sh` compiles and runs
them with two compilers and writes the outputs shown in the guide to
`examples/out`.
