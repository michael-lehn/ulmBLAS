#!/bin/bash
#  build.sh SRC: builds ulmlapack (experiments/lapack/build) in the unpacked source SRC
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$1/experiments/lapack"
make -j 16 "${MAKEVARS[@]}" all > build.log 2>&1 || { tail -30 build.log; exit 1; }
grep -E "warning|error" build.log || true
nm -C build/libulmlapack.a | grep -c "skylakex" | sed 's/^/skylakex kernel symbols: /'
