#! /bin/sh
#
#  set-version.sh --citation VERSION [DATE]: see below (CITATION.cff).
#
#  set-version.sh VERSION TARBALL [URL]: puts the version and the checksums
#  of the release tarball into all recipes of packaging/ (per release,
#  after the tag vVERSION exists: TARBALL is the file GitHub serves as
#  https://github.com/michael-lehn/ulmBLAS/archive/refs/tags/vVERSION.tar.gz).
#
#  URL (optional, for test builds, e.g. the packaging dry run in CI):
#  replaces https://github.com/michael-lehn/ulmBLAS/archive/refs/tags/ in
#  the recipes, e.g. file:///tmp/dist/ with the tarball as
#  /tmp/dist/vVERSION.tar.gz.
#
set -eu
#  --citation VERSION [DATE]: version and date-released in CITATION.cff
#  (before the tag; DATE YYYY-MM-DD, default today)
if [ "${1:-}" = --citation ]; then
    [ $# -ge 2 ] || { echo "usage: set-version.sh --citation VERSION [DATE]" >&2; exit 2; }
    C=$(cd "$(dirname "$0")/.." && pwd)/CITATION.cff
    D=${3:-$(date +%Y-%m-%d)}
    sed -e "s/^version: .*/version: \"$2\"/" \
        -e "s/^date-released: .*/date-released: $D/" "$C" > "$C.tmp"
    mv "$C.tmp" "$C"
    grep -E "^(version|date-released):" "$C"
    exit 0
fi
[ $# -ge 2 ] || { echo "usage: set-version.sh VERSION TARBALL [URL]" >&2; exit 2; }
V=$1
T=$2
URL=${3:-}
P=$(cd "$(dirname "$0")" && pwd)

digest()
{
    if command -v "${1}sum" > /dev/null 2>&1; then
        "${1}sum" "$T" | awk '{print $1}'
    else
        shasum -a "${1#sha}" "$T" | awk '{print $1}'
    fi
}
SHA256=$(digest sha256)
SHA512=$(digest sha512)
#  nix: SRI hash (base64 of the binary digest)
if command -v openssl > /dev/null 2>&1; then
    SRI=sha256-$(openssl dgst -sha256 -binary "$T" | base64 | tr -d '\n')
else
    SRI=sha256-$(python3 -c 'import base64, sys
print(base64.b64encode(bytes.fromhex(sys.argv[1])).decode())' "$SHA256")
fi

edit()
{
    f=$1
    shift
    sed "$@" "$f" > "$f.tmp"
    if [ -n "$URL" ]; then
        sed "s|https://github.com/michael-lehn/ulmBLAS/archive/refs/tags/|$URL|g" \
            "$f.tmp" > "$f"
        rm -f "$f.tmp"
    else
        mv "$f.tmp" "$f"
    fi
}

for f in "$P/arch/PKGBUILD" "$P/msys2/PKGBUILD"; do
    edit "$f" -e "s/^pkgver=.*/pkgver=$V/" -e "s/^pkgrel=.*/pkgrel=1/" \
         -e "s/^sha256sums=.*/sha256sums=('$SHA256')/"
done
edit "$P/alpine/APKBUILD" -e "s/^pkgver=.*/pkgver=$V/" \
     -e "s/^sha512sums=.*/sha512sums=\"$SHA512  ulmblas-$V.tar.gz\"/"
for f in "$P/fedora/ulmblas.spec" "$P/opensuse/ulmblas.spec"; do
    edit "$f" -e "s/^Version:\( *\).*/Version:\1$V/"
done
edit "$P/debian/changelog" -e "1s/^ulmblas ([^)]*)/ulmblas ($V-1)/"
edit "$P/nix/package.nix" -e "s/version = \"[^\"]*\";/version = \"$V\";/" \
     -e "s|hash = \"sha256-[^\"]*\";|hash = \"$SRI\";|"
edit "$P/spack/package.py" \
     -e "s|ulmBLAS/archive/refs/tags/v[0-9.]*\.tar\.gz|ulmBLAS/archive/refs/tags/v$V.tar.gz|" \
     -e "s/version(\"[^\"]*\", sha256=\"[0-9a-f]*\")/version(\"$V\", sha256=\"$SHA256\")/"
for f in "$P"/easybuild/ulmBLAS-*.eb; do
    edit "$f" -e "s/^version = .*/version = '$V'/" \
         -e "s/^checksums = .*/checksums = ['$SHA256']/"
    n=$P/easybuild/ulmBLAS-$V-$(echo "$f" | sed 's/.*ulmBLAS-[^-]*-//')
    [ "$f" = "$n" ] || mv "$f" "$n"
done
for f in "$P/homebrew-core/ulmblas.rb" "$P/homebrew/ulmblas.rb"; do
    edit "$f" \
         -e "s|ulmBLAS/archive/refs/tags/v[0-9.]*\.tar\.gz|ulmBLAS/archive/refs/tags/v$V.tar.gz|" \
         -e "s/^  sha256 \"[0-9a-f]*\"$/  sha256 \"$SHA256\"/"
done
edit "$P/conda-forge/ulmblas/recipe.yaml" \
     -e "s/^  version: \".*\"/  version: \"$V\"/" \
     -e "s/^  sha256: [0-9a-f]*/  sha256: $SHA256/"
echo "version $V, sha256 $SHA256"
