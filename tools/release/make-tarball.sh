#! /bin/sh
#
#  make-tarball.sh [-o OUTDIR] [-k KEY] REF [VERSION]: the release tarball
#  of REF (normally the tag vX.Y.Z), reproducible:
#
#      OUTDIR/ulmBLAS-VERSION.tar.gz       git archive of REF, top directory
#                                          ulmBLAS-VERSION/
#      OUTDIR/SHA256SUMS                   its SHA-256
#      OUTDIR/ulmBLAS-VERSION.tar.gz.asc   detached signature
#
#  VERSION: default REF without the leading v.  OUTDIR: default dist.
#  KEY: the signing key (default ULMBLAS_GPG_KEY, else the release key
#  5BD0789A912F5874A1366392D8B6E130B9F67007, see RELEASING.md); without
#  its secret key in the keyring, or with -k none, no signature.
#
#  The tar file is the output of git archive (files in the order of the
#  tree, mtime = commit time, owner root, mode by tar.umask 0002), so it
#  depends only on the commit; the export-ignore and export-subst rules of
#  .gitattributes apply (VERSION.txt gets the tag).  Compressed with GNU
#  gzip -n (no name, no time stamp), the file is the same as the one GitHub
#  serves as archive/refs/tags/vVERSION.tar.gz, which the recipes in
#  packaging/ download.  Without GNU gzip git's own gzip is used (also
#  reproducible, but a different file than GitHub's).
#
#  Nothing is tagged, pushed or uploaded.
#
set -eu
OUT=dist
KEY=${ULMBLAS_GPG_KEY:-5BD0789A912F5874A1366392D8B6E130B9F67007}
while getopts o:k: opt; do
    case $opt in
        o) OUT=$OPTARG ;;
        k) KEY=$OPTARG ;;
        *) echo "usage: make-tarball.sh [-o OUTDIR] [-k KEY] REF [VERSION]" >&2
           exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] && [ $# -le 2 ] || {
    echo "usage: make-tarball.sh [-o OUTDIR] [-k KEY] REF [VERSION]" >&2
    exit 2
}
REF=$1
V=${2:-$(echo "$REF" | sed 's/^v//')}
TOP=$(cd "$(dirname "$0")/../.." && pwd)
NAME=ulmBLAS-$V
TARBALL=$NAME.tar.gz

git -C "$TOP" rev-parse --verify -q "$REF^{commit}" > /dev/null || {
    echo "error: $REF is not a commit of $TOP" >&2
    exit 2
}
if ! git -C "$TOP" rev-parse --verify -q "refs/tags/$REF" > /dev/null; then
    echo "warning: $REF is not a tag: VERSION.txt gets the last tag before it" >&2
fi

#  git archive with fixed settings (no end-of-line conversion of the
#  user's configuration, the umask GitHub uses)
archive()
{
    git -C "$TOP" -c core.autocrlf=false -c core.eol=lf -c tar.umask=0002 \
        archive "$@" --prefix="$NAME/" "$REF"
}

sha256()
{
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "$1"
    else
        shasum -a 256 "$1"
    fi
}

GZIP_CMD=
for g in gzip ggzip; do
    if command -v $g > /dev/null 2>&1 && $g --version 2>&1 | grep -q '^gzip [0-9]'; then
        GZIP_CMD=$g
        break
    fi
done

mkdir -p "$OUT"
cd "$OUT"
rm -f "$TARBALL" "$TARBALL.asc" SHA256SUMS
if [ -n "$GZIP_CMD" ]; then
    archive --format=tar | $GZIP_CMD -n > "$TARBALL.tmp"
else
    echo "warning: no GNU gzip: compressed by git (not the same file as GitHub's)" >&2
    archive --format=tar.gz > "$TARBALL.tmp"
fi
mv "$TARBALL.tmp" "$TARBALL"

#  VERSION.txt of the tarball must name the version
VT=$(gzip -dc "$TARBALL" | tar -xOf - "$NAME/VERSION.txt" 2> /dev/null || true)
if [ "$VT" != "v$V" ]; then
    echo "warning: VERSION.txt of the tarball is '$VT', not 'v$V'" >&2
fi

sha256 "$TARBALL" > SHA256SUMS
if [ "$KEY" = none ]; then
    KEY=
elif ! command -v gpg > /dev/null 2>&1 ||
     ! gpg --batch --list-secret-keys "$KEY" > /dev/null 2>&1; then
    echo "warning: no secret key $KEY: tarball not signed" >&2
    KEY=
fi
if [ -n "$KEY" ]; then
    gpg --batch --yes --armor --detach-sign --local-user "$KEY" \
        --output "$TARBALL.asc" "$TARBALL"
    gpg --batch --verify "$TARBALL.asc" "$TARBALL"
fi
cat SHA256SUMS
echo "$(pwd)/$TARBALL"
