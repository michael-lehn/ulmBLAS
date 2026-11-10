#! /bin/sh
#
#  fetch.sh DIR: Reference-LAPACK 3.12.1 in DIR/lapack-3.12.1, downloaded
#  (and its checksum verified) unless it is already there; the sources are
#  used unmodified.  LAPACK_TARBALL: use this file instead of downloading.
#  Used by build.sh (this directory) and test/reference-lapack.sh, which
#  extracts only the parts it needs (LAPACK_PARTS, default: all).
#
set -u
VERSION=3.12.1
URL=https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v$VERSION.tar.gz
SHA256=2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422
DIR=$1
SRC=$DIR/lapack-$VERSION
TARBALL=${LAPACK_TARBALL:-$DIR/lapack-$VERSION.tar.gz}
PARTS=${LAPACK_PARTS:-lapack-$VERSION}

sha256()
{
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

mkdir -p "$DIR" || exit 2
[ -d "$SRC" ] && exit 0
if [ ! -f "$TARBALL" ]; then
    echo "downloading $URL"
    if command -v curl > /dev/null 2>&1; then
        curl -fsSL -o "$TARBALL.tmp" "$URL"
    else
        wget -q -O "$TARBALL.tmp" "$URL"
    fi || { echo "error: download failed" >&2; exit 2; }
    mv "$TARBALL.tmp" "$TARBALL"
fi
if [ "`sha256 "$TARBALL"`" != "$SHA256" ]; then
    echo "error: checksum of $TARBALL does not match" >&2
    exit 2
fi
(cd "$DIR" && gzip -dc "$TARBALL" | tar -xf - $PARTS) || exit 2
