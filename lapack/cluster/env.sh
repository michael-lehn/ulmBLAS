#  Environment of the ulmlapack experiments on bwUniCluster 3.0 (Ice Lake,
#  partition cpu_il / dev_cpu_il, 2 x Xeon Platinum 8358).  Sourced by the
#  job scripts.
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
L=$W/lapack-omp                              # experiment directory
OB=$L/openblas/install                       # OpenBLAS 0.3.34, USE_OPENMP=1
module load compiler/gnu/14.2 >/dev/null 2>&1

#  ulmlapack build: ulmBLAS headers with OpenBLAS's SKYLAKEX micro kernel
#  (the fastest ulmBLAS DGEMM configuration on Ice Lake, campaign 023)
MAKEVARS=(CXX=g++
          "ARCHFLAGS=-mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12"
          OMPFLAGS=-fopenmp OMPLIBS=-fopenmp
          "REFLIBS=$OB/lib/libopenblas.a -lgfortran -lm")

#  Threads: one socket (cores 0-31 = NUMA nodes 0 and 1, sub-NUMA
#  clustering), one thread per core, memory interleaved over the two NUMA
#  nodes of the socket.  The same for ulmlapack and OpenBLAS (both use
#  libgomp of GCC 14.2; OpenBLAS's OpenMP build takes omp_get_max_threads).
THREADS=${THREADS:-32}
export OMP_NUM_THREADS=$THREADS
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_MAX_TASK_PRIORITY=3
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-}
[ -z "$OMP_WAIT_POLICY" ] && unset OMP_WAIT_POLICY
unset OPENBLAS_NUM_THREADS GOTO_NUM_THREADS
NUMA=(numactl --cpunodebind=0,1 --interleave=0,1)
