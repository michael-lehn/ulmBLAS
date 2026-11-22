# ATLAS 3.10.3 test sources

The files of ATLAS (Automatically Tuned Linear Algebra Software) 3.10.3
that the testers of `make check-large` need, unmodified: `src/testing/*.c`
(the trusted wrappers around the reference BLAS, norms, matrix
generators), `src/testing/ATL_?f77wrap.f` and `src/auxil/ATL_set.c`.
`s.files`, `d.files`, `c.files` and `z.files` list the C files each type
uses (`../build.sh`).

The testers themselves (`../../l1blastst.c`, `l2blastst.c`, `l3blastst.c`)
and the headers (`../../atlas_*.h`) are those of ATLAS 3.10.3 as well; the
other `ATL_*.c` in `bench/` are ATLAS 3.10.1, used by `make bench`.  The
3.10.3 versions of the norms detect NaN themselves (3.10.1's
`ATL_infnrm` calls `ATL_i?amax`, which in the check is the routine under
test).

ATLAS is distributed under a BSD license; its text is at the top of every
file.
