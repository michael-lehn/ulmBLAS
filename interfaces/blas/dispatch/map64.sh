#! /bin/sh
#
#  map64.sh FUNCTIONS: the functions of libulmblas64, the front with the
#  interfaces with 64-bit integers under the usual names, from the
#  functions of a backend (FUNCTIONS, as for gen.sh).  One line per
#  function: the exported name and the name in the backend:
#
#      dgemm_                     dgemm_64_      (Fortran 77, LAPACK, lsame_,
#                                                 xerbla_array_, ...)
#      cblas_dgemm                cblas_dgemm_64
#      ulmblas_set_f77_xerbla     ulmblas_set_f77_xerbla_64
#      ulmblas_set_cblas_xerbla   ulmblas_set_cblas_xerbla_64
#      __flexiblas_set_xerbla     ulmblas_set_f77_xerbla_64
#      ulmblas_set_num_threads    ulmblas_set_num_threads (no integers of
#                                 the BLAS: the same in both interfaces;
#                                 also ..._get_num_threads, ..._affinity
#                                 and the thread hooks of FlexiBLAS)
#
#  Not exported: the LP64 functions without such a counterpart, e.g. the C
#  functions ulmblas_dgemm, ulmblas_dgemm_strided, ... (their ILP64
#  versions are internal to the backends).
#
set -e
awk '
	{ have[$1] = 1; name[NR] = $1 }
	END {
		for (i = 1; i <= NR; ++i) {
			f = name[i]
			if (f ~ /_64_$/ || f ~ /^cblas_.*_64$/) {
				continue
			}
			if (f ~ /_$/ && (substr(f, 1, length(f) - 1) "_64_") in have) {
				print f, substr(f, 1, length(f) - 1) "_64_"
			} else if (f ~ /^cblas_/ && (f "_64") in have) {
				print f, f "_64"
			} else if (f ~ /^(ulmblas_[sg]et_(num_threads|affinity)|flexiblas_backend_[sg]et_num_threads)$/) {
				print f, f
			} else if (f == "__flexiblas_set_xerbla") {
				print f, "ulmblas_set_f77_xerbla_64"
			}
		}
		print "ulmblas_set_f77_xerbla", "ulmblas_set_f77_xerbla_64"
		print "ulmblas_set_cblas_xerbla", "ulmblas_set_cblas_xerbla_64"
	}' "$1" | sort
