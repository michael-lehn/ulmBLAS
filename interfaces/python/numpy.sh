#! /bin/sh
#
#  numpy.sh PREFIX VENV [VERSION]: a Python virtual environment VENV with
#  NumPy (VERSION, default the latest release) built from source against
#  libulmblas and libulmlapack in PREFIX (interfaces/install-shared.sh).
#  Checks the result: NumPy's build configuration names ulmblas/ulmlapack,
#  a matrix product and a linear system are computed correctly.
#
#      sh interfaces/install-shared.sh $HOME/ulmblas
#      sh interfaces/python/numpy.sh $HOME/ulmblas $HOME/venv-ulmblas
#      $HOME/venv-ulmblas/bin/python -c 'import numpy; numpy.show_config()'
#
#  Needs python3 (PYTHON), a C/C++ compiler and pkg-config; the other build
#  dependencies of NumPy (meson, cython, ...) are installed by pip.  Without
#  libulmlapack in PREFIX NumPy uses its own LAPACK (lapack_lite).
#
set -u
[ $# -ge 2 ] || { echo "usage: numpy.sh PREFIX VENV [VERSION]" >&2; exit 2; }
PREFIX=$(cd "$1" && pwd) || exit 2
VENV=$2
PYTHON=${PYTHON:-python3}
SPEC=numpy${3:+==$3}

[ -f "$PREFIX/lib/pkgconfig/ulmblas.pc" ] ||
    { echo "error: $PREFIX/lib/pkgconfig/ulmblas.pc not found" >&2; exit 2; }
if [ -f "$PREFIX/lib/pkgconfig/ulmlapack.pc" ]; then
    LAPACK=ulmlapack
else
    LAPACK=lapack_lite
fi

#  Debian's python3 has no ensurepip without the package python3-venv: then
#  a virtual environment that uses the pip of the system (if there is one)
if [ ! -x "$VENV/bin/python" ]; then
    if ! "$PYTHON" -m venv "$VENV" 2> /dev/null; then
        rm -rf "$VENV"
        "$PYTHON" -m pip --version > /dev/null 2>&1 &&
            "$PYTHON" -m venv --without-pip --system-site-packages "$VENV" ||
            { echo "error: no venv (Debian: apt install python3-venv)" >&2
              exit 1; }
    fi
fi
#  a current pip in the environment: older ones (Debian 12: 23.0) pass
#  repeated --config-settings not on to the build
"$VENV/bin/python" -m pip install -q --upgrade pip || exit 1

#  --no-binary: build from source (a wheel brings its own BLAS);
#  --no-cache-dir: never reuse a wheel built earlier against another BLAS
PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH} \
"$VENV/bin/python" -m pip install --no-cache-dir --force-reinstall \
    --no-binary numpy "$SPEC" \
    --config-settings=setup-args=-Dblas=ulmblas \
    --config-settings=setup-args=-Dlapack=$LAPACK || exit 1

"$VENV/bin/python" - "$LAPACK" << 'EOF' || exit 1
import sys
import numpy as np
cfg = np.show_config(mode="dicts")["Build Dependencies"]
blas, lapack = cfg["blas"]["name"], cfg["lapack"]["name"]
print("numpy", np.__version__, "blas:", blas, "lapack:", lapack)
assert blas == "ulmblas" and lapack == sys.argv[1], "wrong BLAS/LAPACK"
rng = np.random.default_rng(1)
a = rng.standard_normal((200, 300))
b = rng.standard_normal((300, 100))
ref = np.einsum("ik,kj->ij", a, b)  # no BLAS
assert np.allclose(a @ b, ref), "matrix product"
s = rng.standard_normal((300, 300)) + 300 * np.eye(300)
x = np.linalg.solve(s, b)
assert np.allclose(s @ x, b), "linear system"
print("ok")
EOF
