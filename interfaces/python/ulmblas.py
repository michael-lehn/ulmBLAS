"""
ulmblas: matrix products of ulmBLAS for NumPy arrays, also for the element
types NumPy multiplies without BLAS (with its own loops):

    import numpy as np, ulmblas

    A = np.random.randint(-128, 128, (1024, 1024), dtype=np.int8)
    B = np.random.randint(-128, 128, (1024, 1024), dtype=np.int8)
    C = ulmblas.matmul(A, B)          # int32, exact (modulo 2^32)

    X = np.random.rand(1024, 1024).astype(np.float16)
    Z = ulmblas.matmul(X, X.T)        # float32 (products and sums in float32)

Element types of A and B (both the same kind) and of the result:

    int8, uint8 (each its own)   int32    ulmblas_i8i32gemm_strided
    int16                        int32    ulmblas_i16i32gemm_strided
    float16                      float32  ulmblas_shgemm_strided
    bfloat16 (ml_dtypes)         float32  ulmblas_sbgemm_strided
    float32, float64             the same ulmblas_{s,d}gemm_strided

A and B are 2-dimensional arrays with any strides (views, transposes,
negative steps, broadcasts with stride 0), passed without copies.

    ulmblas.matmul(A, B, out=None, alpha=1, beta=0)

computes out = alpha A B + beta out (out allocated if None, then beta
must be 0).  out must not overlap with A or B; integers wrap around modulo
2^32.

The library: the environment variable ULMBLAS_LIBRARY, else libulmblas of
the active conda environment or of PREFIX/lib for PREFIX in
ULMBLAS_PREFIX (interfaces/install-shared.sh), else the system's search
path.
"""

import ctypes
import ctypes.util
import os
import sys

import numpy as np

__all__ = ["matmul", "library"]

_i64 = ctypes.c_int64
_i32 = ctypes.c_int32
_f32 = ctypes.c_float
_f64 = ctypes.c_double
_ptr = ctypes.c_void_p


def _find_library():
    path = os.environ.get("ULMBLAS_LIBRARY")
    if path:
        return path
    if sys.platform == "darwin":
        names = ["libulmblas.dylib", "libulmblas.0.dylib"]
    elif sys.platform == "win32":
        names = ["libulmblas.dll"]
    else:
        names = ["libulmblas.so", "libulmblas.so.0"]
    prefixes = [os.environ.get(v) for v in ("ULMBLAS_PREFIX", "CONDA_PREFIX")]
    for prefix in filter(None, prefixes):
        for sub in ("lib", "bin", os.path.join("Library", "bin")):
            for name in names:
                p = os.path.join(prefix, sub, name)
                if os.path.exists(p):
                    return p
    found = ctypes.util.find_library("ulmblas")
    if found:
        return found
    raise OSError("ulmblas: libulmblas not found; set ULMBLAS_LIBRARY")


library = _find_library()
_lib = ctypes.CDLL(library)


def _function(name, argtypes):
    try:
        f = getattr(_lib, name)
    except AttributeError:
        return None
    f.argtypes = argtypes
    f.restype = None
    return f


_strides = [_i64, _i64]
_i8i32 = _function("ulmblas_i8i32gemm_strided",
                   [_i64] * 3 + [_i32, ctypes.c_int, _ptr] + _strides +
                   [ctypes.c_int, _ptr] + _strides + [_i32, _ptr] + _strides)
_i16i32 = _function("ulmblas_i16i32gemm_strided",
                    [_i64] * 3 + [_i32, _ptr] + _strides + [_ptr] + _strides +
                    [_i32, _ptr] + _strides)
_gemm16 = {
    "sh": _function("ulmblas_shgemm_strided",
                    [_i64] * 3 + [_f32, _ptr] + _strides + [_ptr] + _strides +
                    [_f32, _ptr] + _strides),
    "sb": _function("ulmblas_sbgemm_strided",
                    [_i64] * 3 + [_f32, _ptr] + _strides + [_ptr] + _strides +
                    [_f32, _ptr] + _strides),
}
_gemm = {
    np.dtype(np.float32): _function(
        "ulmblas_sgemm_strided",
        [_i64] * 3 + [_f32, _ptr] + _strides + [_ptr] + _strides +
        [_f32, _ptr] + _strides),
    np.dtype(np.float64): _function(
        "ulmblas_dgemm_strided",
        [_i64] * 3 + [_f64, _ptr] + _strides + [_ptr] + _strides +
        [_f64, _ptr] + _strides),
}

try:
    import ml_dtypes

    _bfloat16 = np.dtype(ml_dtypes.bfloat16)
except ImportError:
    _bfloat16 = None


def _layout(X):
    """pointer to element (0,0) and the strides in elements"""
    s0, s1 = X.strides
    size = X.dtype.itemsize
    if s0 % size or s1 % size:
        raise ValueError("ulmblas: strides must be multiples of the "
                         "element size")
    return X.ctypes.data, s0 // size, s1 // size


def _missing(name):
    raise RuntimeError("ulmblas: the library %s has no %s" % (library, name))


def matmul(A, B, out=None, alpha=1, beta=0):
    """out = alpha A B + beta out; see the module documentation"""
    A = np.asarray(A)
    B = np.asarray(B)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("ulmblas.matmul: A and B must be 2-dimensional")
    m, k = A.shape
    if B.shape[0] != k:
        raise ValueError("ulmblas.matmul: A is %dx%d, B is %dx%d"
                         % (m, k, B.shape[0], B.shape[1]))
    n = B.shape[1]
    ta, tb = A.dtype, B.dtype
    int8 = (np.dtype(np.int8), np.dtype(np.uint8))

    if ta in int8 and tb in int8:
        kind, tc = "i8", np.dtype(np.int32)
    elif ta == tb == np.dtype(np.int16):
        kind, tc = "i16", np.dtype(np.int32)
    elif ta == tb == np.dtype(np.float16):
        kind, tc = "sh", np.dtype(np.float32)
    elif _bfloat16 is not None and ta == tb == _bfloat16:
        kind, tc = "sb", np.dtype(np.float32)
    elif ta == tb and ta in _gemm:
        kind, tc = "fp", ta
    else:
        raise TypeError("ulmblas.matmul: element types %s and %s are not "
                        "supported" % (ta, tb))

    if out is None:
        if beta != 0:
            raise ValueError("ulmblas.matmul: beta != 0 needs out")
        out = np.empty((m, n), dtype=tc)
    elif out.dtype != tc or out.shape != (m, n):
        raise ValueError("ulmblas.matmul: out must be %s with shape %s"
                         % (tc, (m, n)))
    if (m > 1 and out.strides[0] == 0) or (n > 1 and out.strides[1] == 0):
        raise ValueError("ulmblas.matmul: out has a zero stride")

    pa, ra, ca = _layout(A)
    pb, rb, cb = _layout(B)
    pc, rc, cc = _layout(out)
    if kind == "i8":
        if _i8i32 is None:
            _missing("ulmblas_i8i32gemm_strided")
        _i8i32(m, n, k, _wrap32(alpha), int(ta == np.uint8), pa, ra, ca,
               int(tb == np.uint8), pb, rb, cb, _wrap32(beta), pc, rc, cc)
    elif kind == "i16":
        if _i16i32 is None:
            _missing("ulmblas_i16i32gemm_strided")
        _i16i32(m, n, k, _wrap32(alpha), pa, ra, ca, pb, rb, cb,
                _wrap32(beta), pc, rc, cc)
    elif kind in _gemm16:
        f = _gemm16[kind]
        if f is None:
            _missing("ulmblas_%sgemm_strided" % kind)
        f(m, n, k, float(alpha), pa, ra, ca, pb, rb, cb, float(beta), pc, rc,
          cc)
    else:
        f = _gemm[ta]
        if f is None:
            _missing("ulmblas_?gemm_strided")
        f(m, n, k, alpha, pa, ra, ca, pb, rb, cb, beta, pc, rc, cc)
    return out


def _wrap32(x):
    """an integer as int32 modulo 2^32"""
    x = int(x) % 2**32
    return x - 2**32 if x >= 2**31 else x
