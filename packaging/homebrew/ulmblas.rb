#  Homebrew formula for the tap michael-lehn/homebrew-ulmblas
#  (Formula/ulmblas.rb there):
#
#      brew install michael-lehn/ulmblas/ulmblas
#
#  Installs libulmblas (BLAS: Fortran 77 interface and CBLAS; on Apple
#  Silicon and x86-64 the front that selects the backend for the CPU) and
#  libulmlapack (Reference-LAPACK on top of it), include/ulmblas/cblas.h
#  and the pkg-config files ulmblas.pc and ulmlapack.pc.  The header is in
#  its own directory, beside the cblas.h of other BLAS libraries.
class Ulmblas < Formula
  desc "BLAS and LAPACK with micro kernels selected for the CPU at load time"
  homepage "https://github.com/michael-lehn/ulmBLAS"
  url "https://github.com/michael-lehn/ulmBLAS/archive/refs/tags/v0.4.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "BSD-3-Clause"

  depends_on "cmake" => :build
  depends_on "pkgconf" => :test
  depends_on "gcc" # gfortran for LAPACK, libgfortran at run time

  on_macos do
    depends_on arch: :arm64
  end

  resource "lapack" do
    url "https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz"
    sha256 "2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422"
  end

  def install
    ENV["LAPACK_TARBALL"] = resource("lapack").cached_download
    ENV["FC"] = Formula["gcc"].opt_bin/"gfortran"
    ENV["MAKE"] = "make -j#{ENV.make_jobs}"
    ENV["ULMBLAS_INCLUDEDIR"] = include/"ulmblas"
    system "sh", "interfaces/install-shared.sh", prefix
  end

  test do
    (testpath/"test.c").write <<~C
      #include <stdio.h>
      #include <cblas.h>
      const char *ulmblas_backend(void);
      void dgesv_(const int *n, const int *nrhs, double *A, const int *ldA,
                  int *ipiv, double *B, const int *ldB, int *info);
      int main(void)
      {
          double A[4] = {1, 2, 3, 4}, B[4] = {1, 0, 0, 1}, C[4] = {0};
          cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2,
                      1.0, A, 2, B, 2, 0.0, C, 2);
          double M[4] = {4, 1, 1, 3}, x[2] = {1, 2};
          int n = 2, one = 1, ipiv[2], info;
          dgesv_(&n, &one, M, &n, ipiv, x, &n, &info);
          printf("%s %g %g %g %g %d\\n", ulmblas_backend(), C[0], C[1], C[2], C[3], info);
          return !(C[0] == 1 && C[3] == 4 && info == 0);
      }
    C
    flags = shell_output("pkgconf --cflags --libs ulmlapack").split
    system ENV.cc, "test.c", *flags, "-o", "test"
    system "./test"
  end
end
