# Tuning: selection tables from measurements

ulmBLAS selects the implementation of GEMM (general matrix-matrix
multiplication) at run time from a selection table
([`ulmblas/level3/gemm/select.h`](../ulmblas/level3/gemm/select.h)).  An
entry of a table is a region of problems (ranges of m, n and k) and the
implementation for it: an instance of the frame algorithm with its backend
(micro kernel and packing), block sizes and macro kernel.  The first entry
that applies to a problem is taken; the last entry applies to all problems.

There is a table per platform (processor) and type of the packed blocks.
At the first call ulmBLAS detects the platform
([`ulmblas/config/platform.h`](../ulmblas/config/platform.h)) and uses the
table generated for it, if there is one; other processors get the default
table of the build, written by hand.

`make tune` generates the table for the machine it runs on, from
measurements, as ATLAS (Automatically Tuned Linear Algebra Software) does
for its parameters:

```
make tune ARCHFLAGS=-mcpu=apple-m4
```

1. It builds a measurement program with the candidates of the build: the
   candidate set of the kernels that `ARCHFLAGS` selects
   ([`gemm/candidates.py`](gemm/candidates.py)).
2. It measures every candidate for every shape of a grid (m, n, k in 32,
   64, ..., 512, 1100, 2300, all combinations; 1100 and 2300 are just above
   the block sizes KC = 1024 and 2240), interleaved, in several passes of short
   program runs.  A run starts only when the machine is idle (load average
   below `LOADMAX`) and, on macOS, not hot (thermal state nominal); a run
   during which the machine got hot is repeated.
3. It generates the table ([`gemm/gentable.py`](gemm/gentable.py)):
   `ulmblas/level3/gemm/tables/PLATFORM-TYPE.h`, the index
   `ulmblas/level3/gemm/tables/tables.h`, and a report with all
   measurements, `tune/results/PLATFORM/SET.md` (and `.tsv`).

The generated files are code and are committed; a normal build does not
tune.  Rebuild ulmBLAS after `make tune` to use a new table.

**Keep the machine idle while `make tune` runs** (close other programs, no
builds, connect a laptop to the power supply): other load changes the
measurements and so the table.

## From measurements to the table

As ATLAS (Automatically Tuned Linear Algebra Software) does, `make tune`
does not map the problem space but fits the parameters of a decision with
a fixed structure.  For the set `double-sme` (see
[`gemm/candidates.py`](gemm/candidates.py)):

```
large blocks (608/2240/3072)   if m >= Mmin and n >= Nmin
prefetching of C               if m n >= MNmin and (k >= Kmin or m n >= MNbig)
maximum block sizes            for all problems or for none
```

Every grid point stands for a cell of problems, bounded by the geometric
means of neighbouring grid values.  The thresholds are lower bounds of
cells; `make tune` tries all of them and takes those that maximize the
geometric mean of the performance relative to the best candidate per cell.
The rules give a table of at most six entries.

The report compares the rules with the best candidate per cell, with boxes
of merged neighbouring cells (a table of many entries) and with a single
candidate for all problems.  On the Apple M5 (grid of 343 shapes):

| Selection | Geometric mean | Worst cell |
|---|---|---|
| best candidate per cell | 1.000 | 1.000 |
| rules (6 entries) | 0.990 | 0.806 |
| boxes of merged cells (23 entries) | 0.992 | 0.928 |
| one candidate | 0.979 | 0.716 |

## Settings

| Variable | Meaning | Default |
|---|---|---|
| `ARCHFLAGS` | architecture flags of the build (select the kernels) | see `make check_arch` |
| `PASSES` | passes through the grid | 3 |
| `ROUNDS` | samples per shape and candidate in a pass | 5 |
| `CHUNK` | shapes per program run | 50 |
| `LOADMAX` | a run starts only below this 1-minute load average | 1.5 |
| `PAUSE` | seconds of pause before a run | 5 |
| `PLATFORM` | name of the platform of the table | detected |

## Candidate sets

| Set | Build | Candidates |
|---|---|---|
| `double-sme` | SME2 (Scalable Matrix Extension 2) kernels (`-mcpu=apple-m4`) | MC/KC/NC 256/1024/1024 and 608/2240/3072, each without and with maximum block sizes (MC + 2 MR, 5/4 KC, NC + 2 NR), and without and with prefetching of C in the streaming macro kernel |

A new set is an entry in [`gemm/candidates.py`](gemm/candidates.py): the
preprocessor condition under which its kernels exist, the type of the
packed blocks, the backend and the frame algorithms.

# Tuning of the other parameters

Besides the selection tables, ulmBLAS has parameters that depend on the
platform: thresholds for the number of threads and for variants, block
sizes.  They are in one place,
[`ulmblas/config/tuned.h`](../ulmblas/config/tuned.h): `tuned()` returns
the values written by hand (`tunedDefaults`) with the values measured on
the platform, if there are any
(`ulmblas/config/tuned/PLATFORM.h`, generated).  The environment
variables of the places that use them (e.g. `ULMBLAS_TRSM_SHARED`) still
override them for experiments.

```
make tune-params ARCHFLAGS=-march=native
```

measures them group by group (the groups in this order; each group is
measured with the values of the groups before):

| Group | Parameters (per type unless noted) | Decision |
|---|---|---|
| `gemm-threads` | `gemmMinWork` | threads of gemm (and of the level 3 operations on its frame): m n k / MinWork, at most the pool |
| `gemm-partition` | `gemmRatioM`, `gemmRatioN`, `gemmMaxJR` (all types) | partition of the threads of gemm (as BLIS (BLAS-like Library Instantiation Software; BLAS: Basic Linear Algebra Subprograms)) |
| `trsm-kb`, `trmm-kb` | `trsm.kb`, `trsmInvert`, `trmm.kb` | rows of a diagonal block; inverted diagonal tiles |
| `trsm-variant`, `trmm-variant` | `sharedBlocks`, `sharedPanels`, `minPanels` | threads share the packed blocks if m > sharedBlocks KB and there are fewer than sharedPanels panels of B per thread, otherwise at least minPanels panels per thread |
| `level2` | `gemvNThreads`, `gemvTThreads`, `gerThreads` | threads of gemv and ger by m n: a decision by size (steps) |

Per group [`params/measure.cc`](params/measure.cc) measures candidates
(values of the parameters, or the decisions they lead to, e.g. the number
of threads) on a fixed set of problems, interleaved, in several passes;
the workers of the thread pool are asleep before every sample.  Then
[`params/params.py`](params/params.py) fits the parameters (for a decision
by size: the best candidate per size, neighbours with the same choice
merged) and generates `ulmblas/config/tuned/PLATFORM.h`, the index
`ulmblas/config/tuned/tuned.h` and the report
`tune/results/PLATFORM/params.md`.  A value replaces the one of the build
only if it gains at least 1 % in the geometric mean over the problems and
is not more than 4 % slower on any problem.  At the end the defaults and
the tuned values are measured against each other (section *verify* of the
report).

Not measured (compile-time constants or not platform specific): the fuse
factors, lanes and blocks of the level 2 kernels
([`ulmblas/config/level2.h`](../ulmblas/config/level2.h), `-D`), the loop
shapes of level 1 per compiler
([`ulmblas/level1/kernel/simd.h`](../ulmblas/level1/kernel/simd.h)), the
affinity of the threads (a setting of the user), the parameters of the
task-based LAPACK (Linear Algebra PACKage) functions (`potrfNb`, `potrfSerialN`, `inverseSerialN`,
`potrfRowsPerThread` are in `tuned()`, but not measured yet; getrf has its
own, `ulmblas/lapack/tasks/param.h`).

| Variable | Meaning | Default |
|---|---|---|
| `TUNEGROUPS` | the groups | all, in the order above |
| `TYPES` | the types | `s d c z` |
| `PASSES` | passes through the problems | 2 |
| `ROUNDS` | samples per problem and setting in a pass | 5 |
| `CHUNK` | problems per program run | 6 |
| `LOADMAX` | a group starts only below this 1-minute load average (at most 10 minutes of waiting) | 1.5 |
| `VERIFY` | 0: no comparison of defaults and tuned values; `only`: the comparison only | 1 |

The measurements use the threads of the pool of ulmBLAS as the
environment sets them (`ULMBLAS_NUM_THREADS`, otherwise all CPUs); the
tables of this repository were measured with one thread per core (heim:
6, Ice Lake: 64, Zen 4: 96).

Machines of the same platform can share the work, e.g. one type each
(`TYPES=d`); `python3 tune/params/params.py merge PLATFORM
tune/results/PLATFORM other/params.json ...` adds their results, `params.py
generate` writes the header again.

With SME (Scalable Matrix Extension, Apple M4, M5) level 3 runs on one
thread: there are only `trsm-kb`, `trmm-kb` and `level2`.  On an Apple M5
(idle, on the power supply):

```
make tune-params ARCHFLAGS=-mcpu=apple-m4
```
