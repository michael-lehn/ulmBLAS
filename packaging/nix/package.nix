# nixpkgs: pkgs/by-name/ul/ulmblas/package.nix.  A provider for the blas
# wrapper of nixpkgs:
#
#   blas = prev.blas.override { blasProvider = final.ulmblas; };
#
# The wrapper copies libblas.so.3 into its own output; the dispatching
# library finds its backends through the directory compiled into it
# (ULMBLAS_BACKENDDIR).  LAPACK: lapack-reference on top (lapackProvider).
{
  lib,
  stdenv,
  fetchurl,
  gfortran,
  # the blas wrapper asks for an ILP64 build with blas64 = true; ulmBLAS
  # has no ILP64 library without symbol suffixes yet
  blas64 ? false,
}:

assert !blas64;

stdenv.mkDerivation (finalAttrs: {
  pname = "ulmblas";
  version = "0.4.1";

  src = fetchurl {
    url = "https://github.com/michael-lehn/ulmBLAS/archive/refs/tags/v${finalAttrs.version}.tar.gz";
    hash = "sha256-e4+WFDigDds14lj7pjIqJJ6dydiJgxtFQ3LLDdaIg9I=";
  };

  nativeCheckInputs = [ gfortran ];
  doCheck = true;
  enableParallelBuilding = true;

  passthru = {
    isILP64 = blas64;
  };

  preConfigure = ''
    export ULMBLAS_VERSION=${finalAttrs.version}
    export ULMBLAS_BACKENDDIR=$out/lib/ulmblas
    export ULMBLAS_RPATH=no
    export ULMBLAS_STATIC_LIBSTDCXX=no
    export FC=none
    export MAKE="make -j$NIX_BUILD_CORES"
  ''
  + lib.optionalString stdenv.hostPlatform.isx86_64 ''
    export ULMBLAS_SONAMES=yes
  '';

  buildPhase = ''
    runHook preBuild
    ULMBLAS_STEP=build sh interfaces/install-shared.sh $out
    runHook postBuild
  '';

  checkPhase = ''
    runHook preCheck
    FC=gfortran sh packaging/check.sh
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    sh interfaces/install-shared.sh $out
  ''
  + (
    if stdenv.hostPlatform.isx86_64 then
      ''
        ln -s ulmblas/libblas.so.3 $out/lib/libblas.so.3
        ln -s ulmblas/libcblas.so.3 $out/lib/libcblas.so.3
      ''
    else
      ''
        ln -s libulmblas.so.0 $out/lib/libblas.so.3
        ln -s libulmblas.so.0 $out/lib/libcblas.so.3
      ''
  )
  + ''
    runHook postInstall
  '';

  meta = {
    description = "BLAS with micro kernels selected for the CPU at load time";
    homepage = "https://github.com/michael-lehn/ulmBLAS";
    license = lib.licenses.bsd3;
    platforms = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    maintainers = [ ];
  };
})
