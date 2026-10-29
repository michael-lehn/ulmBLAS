# bwUniCluster 3.0 as a benchmarking environment for ulmBLAS (2026-10)

This report documents how ulmBLAS can be built, benchmarked and compared
with other BLAS implementations on bwUniCluster 3.0 (KIT), which CPU
architectures are available there, which micro kernels the reference
libraries actually use on them, and which BLIS kernels are candidates for
experiments in ulmBLAS.  It covers the existing x86-64 AVX/FMA DGEMM
backend (6 x 8 micro kernel derived from BLIS `haswell`) only; no ulmBLAS
code was changed.

*Historical note: at that time ulmBLAS had optimized kernels for double
only.  The kernels for the other types came later; see
[2026-10-gemm-datatypes](../2026-10-gemm-datatypes/README.md).*

| Directory | Contents |
|---|---|
| [`slurm/`](slurm) | Slurm job scripts: hardware probe, build, smoke test, benchmark campaign |
| [`src/`](src) | benchmark and information programs, summary script (copied into the campaigns) |
| [`config/`](config) | environment of the campaigns: modules, paths, library builds, compiler options |
| [`probes/`](probes) | raw output of the hardware probe jobs on both CPU types |
| [`plot-template/`](plot-template) | Gnuplot scripts used by the campaigns |
| [`plot-template-kernels/`](plot-template-kernels) | Gnuplot scripts for the kernel campaigns (variable set of variants) |
| [`plot-template-full/`](plot-template-full) | Gnuplot scripts for the comprehensive campaign 023 |
| [`kernels/`](kernels) | records of the kernel integration: test results ([`kernels/tests`](kernels/tests/README.md)) |
| [`openblas/`](openblas) | analysis of the OpenBLAS DGEMM kernel and driver on Ice Lake (execution traces) |

The measurements are archived as benchmark campaigns in
`benchmark_results/` (list in [section 5](#5-results)); the DGEMM
baseline on an exclusive Zen 4 node is still queued.

## Status (2026-10-27)

Sections 1-7 are complete; the results grow with the campaigns
([section 5](#5-results)).  ulmBLAS can now select 13 DGEMM micro kernels
of BLIS 2.0, AOCL-BLAS 5.x and OpenBLAS 0.3.34 at build time
([ulmblas/level3/ukernel/blis](../../ulmblas/level3/ukernel/blis/README.md),
[ulmblas/level3/ukernel/openblas](../../ulmblas/level3/ukernel/openblas/README.md));
all pass the test suite on native hardware ([kernels/tests](kernels/tests/README.md)).
The analysis of OpenBLAS on Ice Lake: [openblas](openblas/README.md).

## 1. Cluster access and Slurm usage

### Access

Access from the institute goes through `theon`:

```sh
ssh theon
ssh ul_mlehn@bwunicluster.scc.kit.edu     # plain ssh; X11 forwarding is not needed
```

Non-interactive commands and scripts can be piped through both hops, e.g.
`ssh theon 'ssh ul_mlehn@bwunicluster.scc.kit.edu bash -l -s' < script.sh`
(`bash -l` loads the module system).  No authentication configuration was
changed for this investigation.

The account (`ul_mlehn`, group `ul_math6`) belongs to the Slurm account
`ul` (Ulm University) with QoS `normal`; `sshare` shows a fair-share factor
of 0.92 and no usage before this investigation.  The bwUniCluster usage
policy limits every user to 1920 physical cores in running jobs.

### Rules that matter for benchmarking

- **Login nodes**: AMD EPYC 9454 (the same CPU as the `cpu` partition, but
  shared by all users).  Only short, small tasks are allowed there ("heavy
  computations are not allowed on the login nodes").  Here they were used
  only for downloads, `git`, compiling single files and submitting jobs;
  nothing was measured on them.
- **Users cannot list nodes**: `sinfo` and `scontrol show node` fail with
  "Access/permission denied"; `scontrol show partition`, `squeue --me`,
  `sacctmgr`, `sshare` work.  All hardware details below were therefore
  recorded **inside jobs** (`slurm/hwprobe.sbatch`), not taken from the login
  node or from the documentation alone.
- **Slurm commands must not be run in tight loops** (minimum interval 30 s
  according to the documentation, violations may terminate the account's
  jobs without notice).
- **Development partitions** (`dev_cpu`, `dev_cpu_il`): at most 30 minutes,
  one running job per user, at most 3 queued; meant for tests and
  performance optimization.  They were used for the builds and probes.
- **Exclusive nodes**: `sbatch --exclusive` allocates all CPUs of a node.
  Without it, `--cpus-per-task=2` yields one physical core with both
  hardware threads (e.g. CPUs 0 and 64 on an Ice Lake node), and other jobs
  run on the other cores of the same node.

### Partitions

From `scontrol show partition` (2026-10-26) and the
[bwUniCluster 3.0 documentation](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs):

| Partition | Nodes | CPU | Cores per node | Memory per node | Time limit | Purpose |
|---|---|---|---|---|---|---|
| `cpu` | 80 (uc3n001-068, 091-102) | 2x AMD EPYC 9454 (Zen 4) | 96 (192 threads) | 384 GB | 72 h | standard CPU nodes |
| `dev_cpu` | 2 (uc3n069-070) | 2x AMD EPYC 9454 | 96 | 384 GB | 30 min | development |
| `highmem` | 5 | 2x AMD EPYC 9454 | 96 | 2.3 TB | 72 h | large memory |
| `cpu_il` | 264 (uc2n609-872) | 2x Intel Xeon Platinum 8358 (Ice Lake) | 64 (128 threads) | 256 GB | 72 h | Ice Lake CPU nodes |
| `dev_cpu_il` | 8 (uc2n601-608) | 2x Intel Xeon Platinum 8358 | 64 | 256 GB | 30 min | development |
| `gpu_h100`, `gpu_h100_short`, `dev_gpu_h100` | 12, 7, 1 | 2x AMD EPYC 9454 + 4x H100 | 96 | 768 GB | 72 h / 30 min | GPU |
| `gpu_mi300` | 1 | 4x AMD Instinct MI300A (Zen 4 APU) | 96 | 512 GB HBM3 | 72 h | GPU (APU) |
| `gpu_a100_il`, `gpu_h100_il`, `dev_gpu_a100_il` | 9, 5, 1 | 2x Intel Xeon Platinum 8358 + 4x A100/H100 | 64 | 512 GB | 48 h / 30 min | GPU |
| `gpu_a100_short` | 19 (uc2n561-579) | 2x Intel Xeon Gold 6248R (Cascade Lake) + 4x A100 | 48 | 384 GB | 30 min | GPU |

The partition, and with it the CPU type, is selected with `-p`; there are
no node features to select by (`--constraint` is not needed and node
features are not visible to users).

### Job scripts

| Script | Partition | Resources | Purpose |
|---|---|---|---|
| [`slurm/hwprobe.sbatch`](slurm/hwprobe.sbatch) | `dev_cpu`, `dev_cpu_il` | 1 core, 5 min | records `lscpu`, caches, NUMA, frequency settings, compilers, `-march=native` |
| [`slurm/build.sbatch`](slurm/build.sbatch) | `dev_cpu`, `dev_cpu_il` | 16 cores, 30 min | builds ulmBLAS, BLIS (2 configurations), OpenBLAS, the Eigen programs, links MKL and AOCL; records which kernels each library selects on that node |
| [`slurm/smoke.sbatch`](slurm/smoke.sbatch) | `cpu`, `cpu_il` | 1 core, 8 min | runs every program once (functional test, not a measurement) |
| [`slurm/bench.sbatch`](slurm/bench.sbatch) | `cpu`, `cpu_il` | 1 exclusive node, 75 min | the benchmark campaign |
| [`slurm/full-build.sbatch`](slurm/full-build.sbatch), [`slurm/full-bench.sbatch`](slurm/full-bench.sbatch) | `dev_cpu_il`, `cpu_il` | 16 cores; 1 exclusive node, 10 h | the comprehensive campaign 023: builds the 6 ulmBLAS variants (with `kernels-build.sbatch`), then 11 variants x 80 sizes x 3 rounds and the verification (summary: `src/summarize-full.py`, figures: `plot-template-full/`) |

All scripts write only below the workspace (see [Storage](#storage)).
Example for Ice Lake:

```sh
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench    # ws_find ulmblas-bench
cd $W
sbatch -p dev_cpu_il -J build-icelake build.sbatch icelake
sbatch -p cpu_il     -J smoke-icelake smoke.sbatch icelake
sbatch -p cpu_il     -J bench-icelake bench.sbatch icelake
```

### Storage

| Location | Size, lifetime | Used for |
|---|---|---|
| `$HOME` (`/home/ul/ul_math6/ul_mlehn`, Lustre) | quota 500 GB / 5 M files, backed up | nothing large; scripts could live here |
| Workspace `ulmblas-bench` (`ws_allocate ulmblas-bench 60`, `/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench`, Lustre `pfs7wor9`) | 40 TiB quota, expires 2026-12-25 (3 extensions of up to 60 days possible with `ws_extend`) | sources, library builds, programs, raw results |
| `$TMPDIR` (`/scratch`, node-local NVMe) | job lifetime | not needed (the benchmarks do no I/O) |

The results are copied from the workspace into this repository; the
workspace is not a permanent archive.

## 2. CPU architectures

Recorded in jobs on `uc2n601` (`dev_cpu_il`) and `uc3n069` (`dev_cpu`),
see [`probes/`](probes):

| | Intel Xeon Platinum 8358 | AMD EPYC 9454 |
|---|---|---|
| Partitions | `cpu_il`, `dev_cpu_il` (also the Ice Lake GPU nodes) | `cpu`, `dev_cpu`, `highmem` (also the H100 nodes and login nodes) |
| Microarchitecture | Ice Lake-SP (family 6, model 106, stepping 6), 10 nm | Zen 4 "Genoa" (family 25, model 17, stepping 1), 5 nm |
| Clock | base 2.6 GHz, max. turbo 3.4 GHz | max. boost 3.81 GHz |
| Sockets x cores, SMT | 2 x 32, 2 threads per core (SMT on) | 2 x 48, 2 threads per core (SMT on) |
| L1d / L1i per core | 48 KiB (12-way) / 32 KiB | 32 KiB (8-way) / 32 KiB |
| L2 per core | 1.25 MiB (20-way) | 1 MiB (8-way) |
| L3 | 48 MiB per socket (shared by 32 cores, 12-way) | 32 MiB per CCD (shared by 6 cores, 16-way), 8 CCDs per socket |
| NUMA | 4 nodes (sub-NUMA clustering, 16 cores each), 64 GB each | 8 nodes (NPS4, 12 cores each), 48 GB each |
| SIMD | SSE4.2, AVX, AVX2, FMA, AVX-512 F/CD/BW/DQ/VL/IFMA/VBMI/VBMI2/VNNI/BITALG/VPOPCNTDQ; no AMX, no AVX-512 FP16/BF16 | as Ice Lake plus AVX-512 BF16; no AMX |
| FMA units for double | 2 x 512 bit (ports 0+1 fused and port 5) | 2 x 256 bit; 512-bit instructions take two passes ("double pumped") |
| Measured FMA peak of one core (`peak.c`) | 53.7 GFLOP/s with ymm (2 FMA/cycle at 3.35 GHz), **101.9 GFLOP/s with zmm** (2 FMA/cycle at 3.18 GHz) | 60.0 GFLOP/s with ymm (2 FMA/cycle at 3.75 GHz), 59.9 GFLOP/s with zmm (1 FMA/cycle) |
| Frequency control | `intel_pstate`, governor `performance`, turbo on | `amd-pstate` (passive), governor `performance`, boost on |
| Other | transparent huge pages `always`; `perf_event_paranoid` 2; `perf` not installed, `likwid` 5 installed | same |
| `gcc -march=native` (GCC 14.2) | `icelake-server` | `znver4` |

The peak values are from jobs on shared nodes (other jobs on other cores);
the campaigns repeat the measurement on their exclusive nodes before and
after the rounds.

Consequences for ulmBLAS:

- On **Ice Lake** the AVX2/FMA kernel can reach at most 53% of the core's
  double-precision peak: an AVX-512 kernel can nearly double the attainable
  DGEMM performance (the AVX-512 clock is only 5% lower than the AVX2 clock
  in the measurement above).
- On **Zen 4** AVX-512 does not increase the FMA throughput; a 512-bit
  kernel can only gain through 32 vector registers (larger micro tiles,
  fewer loads per FMA) and masking.  The AVX2/FMA kernel can in principle
  reach the peak.
- Compared with the **Intel Core i5-12500** of the heim campaigns
  (Alder Lake P-core: 2 x 256-bit FMA, no AVX-512, 48 KiB L1d, 1.25 MiB L2,
  measured up to 61 GFLOP/s with OpenBLAS), the Ice Lake core has the same
  L1d and L2 sizes, a lower clock (3.35 instead of about 4.0 GHz) and
  AVX-512; Zen 4 has a smaller L1d (32 KiB) and L2 (1 MiB) and the same
  256-bit FMA throughput per cycle.

### Other CPU types

- **Intel Xeon Gold 6248R (Cascade Lake, 2 x 512-bit FMA)** exists only in
  the GPU partition `gpu_a100_short` (30-minute jobs, at least one A100
  must be requested).  The partition accepts all accounts, so it is
  accessible to us, but using a GPU node for CPU benchmarks would block
  GPUs for other users; it was not used.
- **AMD Instinct MI300A** (Zen 4 cores in an APU) in `gpu_mi300`: a single
  node, GPU allocation required; not used.
- **Other bwHPC clusters** (not verified, access requires registration of
  a project for the respective bwForCluster, see
  [Registration](https://wiki.bwhpc.de/e/Registration)): bwForCluster
  **Helix** (Heidelberg; research areas include mathematics and computer
  science) has AMD EPYC 7513 (**Zen 3**, AVX2 without AVX-512) and EPYC
  9334 (Zen 4) nodes; **JUSTUS 2** (Ulm; theoretical chemistry, condensed
  matter, quantum sciences) has Intel Xeon Gold 6252 (**Cascade Lake**,
  AVX-512) nodes.  HoreKa (KIT, NHR) and Hunter (HLRS) require a project
  application.  No access to any of them has been confirmed.

## 3. Compilers and mathematical libraries

### Modules (2026-10-26)

| Kind | Modules |
|---|---|
| Compilers | `compiler/gnu/11.4`, `14.2`, `15.2`, `16.1`; `compiler/llvm/19.1`, `22.1`; `compiler/intel/2025.0`; `compiler/aocc/5.0.0`; system GCC 11.5 (RHEL 9.6) |
| BLAS/LAPACK | `numlib/mkl/2025.0` (oneMKL 2025.0.1), `lib/aocl/5.0.0` (AOCL 5.0 built with AOCC: AOCL-BLAS = AMD's BLIS fork, `libblis.a` single threaded and `libblis-mt`), `lib/eigen/3.4.0`; system packages OpenBLAS 0.3.26 (OpenMP) and FlexiBLAS 3.0.4 |
| Not available | upstream BLIS, a current OpenBLAS |
| Tools | `vis/gnuplot/6.0`, `devel/valgrind/3.27`, `likwid` (system); no `perf` |

### What was benchmarked

| Variant | Version, build | Selected configuration and DGEMM micro kernel on Ice Lake | on Zen 4 |
|---|---|---|---|
| **ulmblas** | ulmBLAS 21197f2 (library identical to febb852 of the heim campaigns), `make cblas CXX=g++ ARCHFLAGS="-mavx -mfma"` (`-O3`), GCC 14.2 | AVX/FMA 6 x 8 (BLIS `haswell` assembly), MC/KC/NC 288/256/4080 | same |
| **blis-auto** | BLIS 2.0, `./configure --disable-shared --enable-cblas -t no auto`, GCC 14.2, built on the target node | `skx`: `bli_dgemm_skx_asm_16x14` (AVX-512), MR x NR 16 x 14, MC/KC/NC 240/256/3752 | `zen3`: `bli_dgemm_haswell_asm_6x8` (AVX2), 6 x 8, 72/256/4080 |
| **blis-haswell** | BLIS 2.0, configuration `haswell` | `bli_dgemm_haswell_asm_6x8`, 6 x 8, 72/256/4080 | same |
| **openblas** | OpenBLAS 0.3.34, `make USE_THREAD=0 USE_OPENMP=0 NO_SHARED=1 NOFORTRAN=1 NO_LAPACK=1`, target detected on the node | `SKYLAKEX`: `dgemm_kernel_16x2_skylakex.c` (AVX-512) | `COOPERLAKE` (Zen 4 has AVX-512 BF16): the same DGEMM kernel as SKYLAKEX |
| **mkl** | oneMKL 2025.0.1 module, static `mkl_intel_lp64`, `mkl_sequential`, `mkl_core` | `MKL_VERBOSE`: AVX-512 code path | `MKL_VERBOSE`: "Intel(R) Architecture processors" (generic message; the ISA of the code path is not reported) |
| **aocl** | AOCL-BLAS 5.0.0 (Build 20241001) from the `lib/aocl/5.0.0` module, static `libblis.a` (single threaded) | **`zen4` ("Genoa")**: `bli_dgemm_zen4_asm_32x6` (AVX-512), 32 x 6, 128/512/4002; AOCL selects its Zen 4 configuration on every CPU with AVX-512, also on Intel | `zen4` ("Genoa"): same |
| **eigen** | Eigen 5.0.1 (header only), `g++ -std=c++17 -O3 -DNDEBUG -march=native`, `C.noalias() += A*B` | AVX-512 (`EIGEN_VECTORIZE_AVX512`, packet size 8) | AVX-512 (packet size 8) |

The kernel names were determined at run time on the target node: the
programs `src/blis-info.c` (configuration chosen by BLIS and AOCL, block
sizes, address of the native dgemm micro kernel, resolved with `nm` in the
statically linked, non-PIE program), `src/openblas-info.c`
(`openblas_get_corename`) and `src/mkl-info.c` (with `MKL_VERBOSE=1`).  All
BLAS programs are linked statically against the libraries (`ldd` lists only
the C and C++ runtime), single threaded by construction (BLIS `-t no`,
OpenBLAS `USE_THREAD=0`, `mkl_sequential`, AOCL `libblis.a`, Eigen without
OpenMP).

The BLAS programs (`bench2.c`, `verify-large.c`) are compiled with
`gcc -O2` as on heim; only the libraries and Eigen are compiled for the
architecture.  ulmBLAS is built with `-mavx -mfma` as on heim (not
`-march=native`), so that its code is the code of the heim campaigns.

Not benchmarked: the system OpenBLAS 0.3.26 (threaded, older than the
built 0.3.34), Eigen 3.4.0 (the heim campaigns used 5.0.1), the threaded
AOCL and MKL libraries.

<!-- KERNELS -->

## 4. Build and benchmark procedures

Everything is built and measured in Slurm jobs; the login nodes are used
only for downloads, `git` and job submission.  Workspace layout and
settings: [`config/uc3.env`](config/uc3.env).

1. **Reference libraries** (once per CPU type, on a node of that type,
   because BLIS `auto` and OpenBLAS select their kernels on the build
   host): `build.sbatch ARCH` (BLIS 2.0 `auto` and `haswell`, OpenBLAS
   0.3.34, Eigen 5.0.1, linking of MKL and AOCL; records the kernels each
   library selects in `logs/kernels.txt`), `build-blis-extra.sbatch ARCH`
   (BLIS `sandybridge`, `penryn`).
2. **ulmBLAS with each micro kernel**: `kernels-build.sbatch ARCH REV TAG`
   builds one library per kernel from git revision REV (optionally only
   `KERNELS=...`, or with `MC`, `KC`, `NC` for tuning).  ulmBLAS's own
   AVX/FMA kernel is built with `-mavx -mfma` (as on heim), the BLIS/AOCL
   AVX-512 kernels with `-mavx -mfma -mavx512f`.
3. **Correctness**: `kernels-check.sbatch ARCH REV` runs `make check`
   (numeric tests and the Netlib BLAS/CBLAS test programs) for the test
   configurations of all kernels (default and small block sizes) on a node
   of that type.  Benchmarks are only run for kernels that pass.
4. **Smoke test** (optional): `smoke.sbatch ARCH` runs every program once
   on a shared core.
5. **Campaign**: `bench.sbatch ARCH` (baseline) or
   `kernels-bench.sbatch ARCH` (kernels; `VARIANTS=...`) on one exclusive
   node: `bench2.c` (7 samples of at least 0.2 s after a warm-up of at
   least 0.3 s, median, first result checked against a naive product),
   rounds with a rotated order of the variants, one process at a time,
   bound with `numactl` to one core and its NUMA node; FMA peak of the core
   (`peak.c`) and conditions (load, clock, temperature) recorded;
   `verify-large.c` checks the large sizes against long double sums.
   `kernels-bench-socket.sbatch` is the one-socket variant for Zen 4 when
   exclusive nodes are not available.
6. **Archive**: copy `res/` to a campaign directory, summarize
   (`src/summarize-uc3.py`, `src/summarize-kernels.py`), add plots and a
   README, `make add_benchmark_result`.

The time limit of a kernel campaign must allow for the slow kernels
(SSE, AVX without FMA): 16 variants needed more than 90 minutes on Ice
Lake (012-2026-10-26-uc3-icelake-blis-kernels ended in round 3).

## 5. Results

| Campaign | CPU | What |
|---|---|---|
| [010-2026-10-26-uc3-icelake-dgemm](../../benchmark_results/010-2026-10-26-uc3-icelake-dgemm/README.md) | Xeon Platinum 8358 | baseline: ulmBLAS AVX/FMA 48-49 GFLOP/s (0.90 of the 256-bit peak, = BLIS haswell), AVX-512 libraries about twice as fast (OpenBLAS, Eigen 93-97, MKL 91-94, BLIS skx 77-79) |
| [011-2026-10-26-heim-blis-kernels](../../benchmark_results/011-2026-10-26-heim-blis-kernels/README.md) | Core i5-12500 | BLIS AVX2/AVX/SSE kernels inside ulmBLAS = native BLIS (0.996-1.015) |
| [013-2026-10-27-heim-blis-kernels-mc](../../benchmark_results/013-2026-10-27-heim-blis-kernels-mc/README.md) | Core i5-12500 | tuning: MC explains the 1.4% between the vendored haswell 6x8 and ulmBLAS's port of it |
| [017-2026-10-27-heim-blis-kernels-old](../../benchmark_results/017-2026-10-27-heim-blis-kernels-old/README.md) | Core i5-12500 | old BLIS haswell 12x4, 4x12 inside ulmBLAS: 0.95, 0.92 of native BLIS haswell |
| [012-2026-10-26-uc3-icelake-blis-kernels](../../benchmark_results/012-2026-10-26-uc3-icelake-blis-kernels/README.md) | Xeon Platinum 8358 | BLIS skx 16x14 inside ulmBLAS = native BLIS skx (1.007); AOCL zen4 32x6 inside ulmBLAS 90-95 GFLOP/s |
| [014-2026-10-27-uc3-icelake-openblas-kernel](../../benchmark_results/014-2026-10-27-uc3-icelake-openblas-kernel/README.md) | Xeon Platinum 8358 | OpenBLAS's kernel inside ulmBLAS: 0.94 of native OpenBLAS |
| [016-2026-10-27-uc3-icelake-openblas-kernel-prof](../../benchmark_results/016-2026-10-27-uc3-icelake-openblas-kernel-prof/README.md) | Xeon Platinum 8358 | diagnosis: the kernel reaches 100 GFLOP/s inside ulmBLAS; per-tile calls (2-3%) and the separate packing phase cost the rest |
| [020-2026-10-27-uc3-icelake-openblas-kernel-panel](../../benchmark_results/020-2026-10-27-uc3-icelake-openblas-kernel-panel/README.md) | Xeon Platinum 8358 | tuning iteration 2: one kernel call per panel of B, OpenBLAS's kernel inside ulmBLAS 0.962 of native OpenBLAS (before 0.940) |
| [021-2026-10-27-uc3-icelake-openblas-kernel-interleave](../../benchmark_results/021-2026-10-27-uc3-icelake-openblas-kernel-interleave/README.md) | Xeon Platinum 8358 | tuning iteration 3: B packed interleaved, OpenBLAS's kernel inside ulmBLAS 0.970 of native OpenBLAS, 0.992 of oneMKL |
| [023-2026-10-27-uc3-icelake-dgemm-single-thread](../../benchmark_results/023-2026-10-27-uc3-icelake-dgemm-single-thread/README.md) | Xeon Platinum 8358 | **comprehensive single-threaded comparison**, n = 50 ... 4000, 11 variants: ulmBLAS with OpenBLAS's kernel 0.971 of OpenBLAS, 0.989 of oneMKL, 0.995 of Eigen, 1.17 of BLIS for n >= 1000; OpenBLAS fastest at 73 of 80 sizes; ulmBLAS 0.59-0.92 of OpenBLAS for n <= 200 |
| [019-2026-10-27-uc3-icelake-kernels-k4](../../benchmark_results/019-2026-10-27-uc3-icelake-kernels-k4/README.md) | Xeon Platinum 8358 | AOCL 5.1 zen5 8x24 inside ulmBLAS 0.91 of OpenBLAS; old haswell 12x4, 4x12 0.95, 0.91 of native BLIS haswell |
| [018-2026-10-27-uc3-zen4-kernels-k4-socket](../../benchmark_results/018-2026-10-27-uc3-zen4-kernels-k4-socket/README.md) | EPYC 9454 (one socket, preliminary) | zen5 8x24 inside ulmBLAS = native AOCL (0.999); old haswell 12x4, 4x12 0.945, 0.941 of native BLIS haswell |
| [015-2026-10-27-uc3-zen4-kernels-socket](../../benchmark_results/015-2026-10-27-uc3-zen4-kernels-socket/README.md) | EPYC 9454 (one socket, preliminary) | every kernel inside ulmBLAS = its native reference (0.986-1.013); AVX-512 only 0-4% faster than AVX2 (same FMA peak) |

## 6. BLIS micro kernels for ulmBLAS

### How ulmBLAS calls a micro kernel

The ulmBLAS micro kernels have the interface of the BLIS micro kernels
before BLIS 1.0: `ugemm(kc, alpha, A, B, beta, C, incRowC, incColC,
nextA, nextB)` computes C ← beta·C + alpha·A·B for one **full** MR x NR
tile.  A is an MR x kc panel stored k-major (`A[i + l*MR]`), B a kc x NR
panel (`B[j + l*NR]`), both packed by ulmBLAS with zero padding to full
panels and with MR or NR elements of padding at the end for reading ahead;
partial tiles at the edges are computed into a local buffer by the
"buffered variant" in `ulmblas/level3/ukernel/ugemm.h` and copied.  The
macro kernel (`mgemm`) runs over the B panels (j) outside and over the A
panels (i) inside, like BLIS.  Constraints:

- `BlockSize` requires MC to be a multiple of **both** MR and NR (the
  symmetric and triangular routines use MC x MC blocks of A), and NC a
  multiple of NR.
- The packing buffers are allocated with `_mm_malloc(…, 32)`: 32-byte
  alignment, sufficient for `vmovapd ymm` but **not for `vmovapd zmm`**.
- The AVX kernel accepts C with unit row or unit column stride; general
  strides go through a local tile.

### Kernels actually selected on the available CPUs

| Kernel | Used by | ISA | MR x NR | Accumulators | Per k: loads of A / broadcasts of B / FMAs | C storage handled in assembly | Blocking of the selecting library (MC/KC/NC) | License |
|---|---|---|---|---|---|---|---|---|
| `bli_dgemm_haswell_asm_6x8` (BLIS 2.0, `kernels/haswell/3/bli_gemm_haswell_asm_d6x8.c`) | ulmBLAS AVX/FMA (already integrated), BLIS `haswell` and `zen3` (Zen 4!) | AVX2, FMA3 | 6 x 8 | 12 ymm | 6 broadcasts of A / 2 loads of B / 12 | row or column stride 1 (`GEMM_UKR_SETUP_CT_AMBI`) | 72/256/4080 | BSD-3 (UT Austin, AMD) |
| `bli_dgemm_skx_asm_16x14` (BLIS 2.0, `kernels/skx/3/bli_dgemm_skx_asm_16x14.c`) | BLIS `skx` on Ice Lake | AVX-512F | 16 x 14 | 28 zmm (zmm4-31) | 2 aligned loads (`vmovapd`) / 14 `vbroadcastsd` / 28 | column stride 1 only (`GEMM_UKR_SETUP_CT`, column preferring); other strides and all edge tiles via a temporary tile | 240/256/3752 | BSD-3 (UT Austin) |
| `bli_dgemm_zen4_asm_32x6` (AOCL-BLAS 5.0, `kernels/zen4/3/bli_dgemm_zen4_asm_32x6.c`) | AOCL `zen4` on Zen 4 **and** on Ice Lake | AVX-512F | 32 x 6 | 24 zmm (zmm8-31) | 4 aligned loads / 6 broadcasts / 24 | column stride 1, general stride with `vgatherqpd`/`vscatterqpd`; full tiles only (old interface without m, n: the AOCL macro kernel handles edges) | 128/512/4002 (Genoa), 64/512/3600 (Bergamo) | BSD-3 (AMD; UT Austin) |

Other kernels in the source trees, not selected on the available CPUs:
`bli_dgemm_haswell_asm_8x6` (the column-preferring AVX2 twin of 6 x 8),
`bli_dgemm_skx_asm_16x12_l2` (skx variant with L2 prefetching),
AOCL `bli_dgemm_zen4_asm_8x24` (row preferring; registered in AOCL's
`zen4` and `zen5` configurations as `BLIS_GEMM_FOR_TRSM_UKR`, the GEMM
kernel of TRSM, with its own TRSM block sizes 120/512/4008) and AOCL zen5 `bli_dgemm_avx512_asm_8x24` (the
DGEMM kernel of AOCL's `zen5` configuration, 24 zmm accumulators, row
preferring).  OpenBLAS (`dgemm_kernel_16x2_skylakex.c`, inline assembly,
16-row A panels, B packed in panels of 2 columns, up to 12 columns per
pass, P/Q = 512/256) and MKL are not BLIS kernels and are not considered
for integration.

All three BLIS-family kernels are inline assembly written with the BLIS
assembly macros (`bli_x86_asm_macros.h`), not intrinsics and not generated
code.  All have the same structure: k unrolled by 4, a main loop, a loop
that prefetches the C tile into L1 during the last iterations (14 + 2 for
skx, 6 + 5 for zen4), a tail loop for k mod 4, and alpha/beta scaling with
a separate path for beta = 0.  The skx kernel prefetches A and B 4
iterations ahead into L1 and the next B panel (`b + 14·k`, i.e. it assumes
contiguous B panels, as ulmBLAS packs them) into L2.

### Compatibility with ulmBLAS

| Kernel | Class | What has to be adapted |
|---|---|---|
| haswell 6 x 8 | integrated | — |
| haswell 8 x 6 | **compatible** | MR/NR swap, same alignment; interface adaptation as for 6 x 8 |
| skx 16 x 14 | **packing / calling-convention adaptation** | MR = 16, NR = 14 in `BlockSize<double>` (MC a multiple of 112, e.g. 224 or 336; NC a multiple of 14, e.g. 3752); 64-byte alignment of the packing buffers; the ulmBLAS interface (alpha/beta by reference, no m/n, full tiles only, nextA/nextB unused) as for the AVX kernel; row-stored C needs a local tile (the CBLAS row-major interface maps to column-major calls, so this matters only for general strides); AVX-512 must be selected (`__AVX512F__`) |
| AOCL zen4 32 x 6 | **packing / calling-convention adaptation** | MR = 32, NR = 6 (MC a multiple of 96); 64-byte alignment; already uses the pre-1.0 interface (full tiles only), so the adaptation is smaller than for skx; general stride is supported in assembly |
| AOCL zen5 8 x 24 | packing adaptation | MR = 8, NR = 24 (MC a multiple of 24); row preferring like the current kernel; for Zen 5 only (no such CPU available) |
| BLIS 2.0 "sup" kernels (`bli_dgemmsup_rv_haswell_asm_6x8m` etc.) | **substantial changes** | operate on unpacked A and/or B for small and skinny problems; would need a separate path in ulmBLAS's GEMM driver |
| AOCL `bli_dgemm_avx512_asm_8x24_macro_kernel`, small-matrix kernels (`bli_gemm_small`, `trsm_small`) | substantial changes | replace the macro kernel or the whole driver for small sizes |

### Licensing

The BLIS kernels are BSD-3-Clause (copyright The University of Texas at
Austin and AMD); AOCL-BLAS is BSD-3-Clause as well (its `LICENSE`, AMD
copyrights 2017 to 2024).  ulmBLAS already includes a BLIS-derived kernel
with the full BSD-3 notice and a provenance note
(`ulmblas/level3/ukernel/avx/ugemm.tcc`); an adapted skx or zen4 kernel
would need the same: the original copyright lines (for the AOCL kernel
the AMD copyright), the license text, the source file, tag and commit,
and the list of modifications.  No BLIS or AOCL code was copied in this
investigation.

## 7. Plan

In order of priority:

1. **OpenBLAS's kernel inside ulmBLAS on Ice Lake** (the three tuning
   iterations are done): one call per panel of B (968e250) and B packed
   interleaved (ac2d97e) raised ulmBLAS from 0.938 to 0.970 of native
   OpenBLAS.  Open: measure the packing code (OpenBLAS's copy kernels)
   and balanced remainders as causes of the last 3%; apply the panel calls
   and the interleaved packing to the other kernels where they help.
2. **Exclusive Zen 4 measurements** of all kernels (queued) to confirm the
   one-socket results.
3. **Choose default kernels per CPU** from the measurements: on Ice Lake
   AOCL zen4 32x6 or OpenBLAS SKYLAKEX 16x12 (about 1.9 times ulmBLAS's
   AVX/FMA kernel), on Zen 4 AOCL zen4 32x6 (4% over AVX/FMA), and a
   run-time dispatch between them.
4. **More kernels**: OpenBLAS's HASWELL kernel (`dgemm_kernel_4x8_haswell.S`,
   assembly file: needs a build change); later AOCL releases.
5. **Other CPUs**: Cascade Lake (bwUniCluster GPU nodes, JUSTUS 2) and
   Zen 3 (bwForCluster Helix) would add AVX-512 with one or two FMA units
   and AVX2-only AMD cores; they need a GPU allocation or a project
   registration.
