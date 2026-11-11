"""
Tests of ulmblas.py: every element type against a reference computed by
NumPy (integers in int64 modulo 2^32, floating point in float64 with an
error bound), with views of several layouts (column major, row major,
non-unit and negative strides, broadcasts).  Exit status 1 on failure.

    ULMBLAS_LIBRARY=.../libulmblas.0.dylib python3 test_ulmblas.py
"""

import sys

import numpy as np

import ulmblas

rng = np.random.default_rng(1)
failures = 0
cases = 0


def operand(dtype, layout, r, c):
    """an r x c view of the given layout"""
    def values(shape):
        if np.dtype(dtype).kind in "iu":
            info = np.iinfo(dtype)
            return rng.integers(info.min, info.max, shape,
                                endpoint=True).astype(dtype)
        return rng.uniform(-1, 1, shape).astype(dtype)
    if layout == "row":
        return values((r, c))
    if layout == "col":
        return np.asfortranarray(values((r, c)))
    if layout == "stride2":
        return values((2 * r + 1, 3 * c + 2))[1::2, 2::3][:r, :c]
    if layout == "neg":
        return values((r + 2, c + 1))[r:0:-1, ::-1][:r, :c]
    if layout == "transposed":
        return values((c, r)).T
    # "bcast": every row the same
    return np.broadcast_to(values((1, c)), (r, c))


def check(ta, tb, m, n, k, la, lb, alpha, beta, out_layout="row"):
    global failures, cases
    cases += 1
    A = operand(ta, la, m, k)
    B = operand(tb, lb, k, n)
    integer = np.dtype(ta).kind in "iu"
    tc = np.int32 if integer else (np.float32 if np.dtype(ta).itemsize <= 2
                                   else np.dtype(ta))
    C = operand(tc, out_layout, m, n)      # a view (no broadcast for C)
    Cin = np.array(C)
    ulmblas.matmul(A, B, out=C, alpha=alpha, beta=beta)
    if integer:
        ref = (alpha * (A.astype(np.int64) @ B.astype(np.int64)) +
               (beta * Cin.astype(np.int64) if beta else 0))
        ref = (ref % 2**32).astype(np.uint32).view(np.int32)
        ok = np.array_equal(C, ref)
    else:
        #  16-bit values are exact in float32 and float64
        A64 = A.astype(np.float32 if np.dtype(ta).itemsize <= 2
                       else ta).astype(np.float64)
        B64 = B.astype(np.float32 if np.dtype(tb).itemsize <= 2
                       else tb).astype(np.float64)
        ref = alpha * (A64 @ B64) + (beta * Cin.astype(np.float64) if beta
                                     else 0)
        bound = (abs(alpha) * (abs(A64) @ abs(B64)) +
                 (abs(beta) * abs(Cin.astype(np.float64)) if beta else 0))
        eps = np.finfo(np.float32 if np.dtype(tc) == np.float32
                       else tc).eps
        ok = np.all(abs(C - ref) <= 4 * (k + 2) * eps * bound + 1e-30)
    if not ok:
        failures += 1
        print("FAIL %s x %s m=%d n=%d k=%d A %s B %s C %s alpha=%s beta=%s"
              % (np.dtype(ta), np.dtype(tb), m, n, k, la, lb, out_layout,
                 alpha, beta))


LAYOUTS = ("row", "col", "stride2", "neg", "transposed", "bcast")
types = [(np.int8, np.int8), (np.int8, np.uint8), (np.uint8, np.int8),
         (np.uint8, np.uint8), (np.int16, np.int16), (np.float16, np.float16),
         (np.float32, np.float32), (np.float64, np.float64)]
try:
    import ml_dtypes
    types.append((ml_dtypes.bfloat16, ml_dtypes.bfloat16))
except ImportError:
    print("ml_dtypes not installed: bfloat16 not tested")

for ta, tb in types:
    integer = np.dtype(ta).kind in "iu"
    ab = [(1, 0), (-3, 5), (2, 1)] if integer else [(1, 0), (-1.5, 0.25)]
    for m, n, k in ((1, 1, 1), (3, 2, 5), (17, 13, 29), (70, 33, 131)):
        for i, la in enumerate(LAYOUTS):
            for j, lb in enumerate(LAYOUTS):
                check(ta, tb, m, n, k, la, lb, *ab[(i + j) % len(ab)])
    for lc in ("row", "col", "stride2", "neg"):
        check(ta, tb, 23, 19, 31, "stride2", "neg", *ab[-1], out_layout=lc)
    check(ta, tb, 301, 263, 517, "col", "row", *ab[-1])

print("%d cases, %d failures (library %s)" % (cases, failures,
                                              ulmblas.library))
sys.exit(1 if failures else 0)
