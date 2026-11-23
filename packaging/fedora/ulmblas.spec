Name:           ulmblas
Version:        0.4.2
Release:        1%{?dist}
Summary:        BLAS with micro kernels selected for the CPU at load time

License:        BSD-3-Clause
URL:            https://github.com/michael-lehn/ulmBLAS
Source0:        https://github.com/michael-lehn/ulmBLAS/archive/refs/tags/v%{version}.tar.gz#/ulmBLAS-%{version}.tar.gz
Source1:        ulmblas.conf

# x86-64: dispatching library with backends; aarch64: NEON kernels
ExclusiveArch:  x86_64 aarch64

BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  gcc-gfortran
BuildRequires:  make

# the backends in %{_libdir}/ulmblas are loaded by libulmblas.so.0 only
%global __provides_exclude_from ^%{_libdir}/ulmblas/libulmblas_.*\\.so$
# the assembly kernels and the dispatching library are not built for LTO
%global _lto_cflags %{nil}

%description
ulmBLAS implements all routines of the reference BLAS (Basic Linear
Algebra Subprograms) for single, double, complex and double complex
precision with the Fortran 77 and the C (CBLAS) interface.  On x86-64
libulmblas.so.0 selects at load time the build of the kernels for the CPU
(AVX-512, AVX2 and FMA, SSE3 or generic x86-64); on aarch64 it uses NEON
kernels.  libulmblas64.so.0 has the same interfaces with 64-bit integers
under the usual names (as libblas64), for programs compiled with 64-bit
INTEGER.

%package devel
Summary:        Development files for ulmBLAS
Requires:       %{name}%{?_isa} = %{version}-%{release}

%description devel
The links libulmblas.so and libulmblas64.so, the headers cblas.h (and
ulmblas64/cblas.h) and the pkg-config files ulmblas.pc and ulmblas64.pc
of ulmBLAS.

%package -n flexiblas-ulmblas
Summary:        ulmBLAS as a backend of FlexiBLAS
Requires:       %{name}%{?_isa} = %{version}-%{release}
Requires:       flexiblas

%description -n flexiblas-ulmblas
The FlexiBLAS configuration of the backend ULMBLAS (libulmblas.so.0):
FLEXIBLAS=ULMBLAS selects it for a program, "flexiblas default ULMBLAS"
for a user.  LAPACK: the functions of ulmBLAS (?getrf, ?potrf, ...)
from the backend, the others from FlexiBLAS's reference implementation.
The same for FlexiBLAS with 64-bit integers (libulmblas64.so.0,
FLEXIBLAS64=ULMBLAS).

%prep
%autosetup -n ulmBLAS-%{version}

%build
export ULMBLAS_VERSION=%{version}
export ULMBLAS_LIBDIR=%{_libdir}
export ULMBLAS_BACKENDDIR=%{_libdir}/ulmblas
export ULMBLAS_INCLUDEDIR=%{_includedir}/ulmblas
export ULMBLAS_RPATH=no
export ULMBLAS_STATIC_LIBSTDCXX=no
export FC=none
export MAKE="make %{?_smp_mflags}"
ULMBLAS_STEP=build sh interfaces/install-shared.sh %{_prefix}

%install
export ULMBLAS_VERSION=%{version}
export ULMBLAS_LIBDIR=%{_libdir}
export ULMBLAS_BACKENDDIR=%{_libdir}/ulmblas
export ULMBLAS_INCLUDEDIR=%{_includedir}/ulmblas
export ULMBLAS_RPATH=no
export ULMBLAS_STATIC_LIBSTDCXX=no
export FC=none
export MAKE="make %{?_smp_mflags}"
DESTDIR=%{buildroot} sh interfaces/install-shared.sh %{_prefix}
install -d %{buildroot}%{_sysconfdir}/flexiblasrc.d
sed 's|@LIBDIR@|%{_libdir}|' %{SOURCE1} \
    > %{buildroot}%{_sysconfdir}/flexiblasrc.d/ulmblas.conf
install -d %{buildroot}%{_sysconfdir}/flexiblas64rc.d
sed -e 's|@LIBDIR@|%{_libdir}|' -e 's|libulmblas\.so|libulmblas64.so|' \
    -e 's|FLEXIBLAS=|FLEXIBLAS64=|' %{SOURCE1} \
    > %{buildroot}%{_sysconfdir}/flexiblas64rc.d/ulmblas.conf

%check
FC=gfortran sh packaging/check.sh

%files
%license LICENSE NOTICE
%doc README.md
%{_libdir}/libulmblas.so.0
%{_libdir}/libulmblas64.so.0
%ifarch x86_64
%{_libdir}/ulmblas/
%endif

%files devel
%{_libdir}/libulmblas.so
%{_libdir}/libulmblas64.so
%{_includedir}/ulmblas/
%{_libdir}/pkgconfig/ulmblas.pc
%{_libdir}/pkgconfig/ulmblas64.pc

%files -n flexiblas-ulmblas
%config(noreplace) %{_sysconfdir}/flexiblasrc.d/ulmblas.conf
%config(noreplace) %{_sysconfdir}/flexiblas64rc.d/ulmblas.conf

%changelog
* Tue Nov 17 2026 Michael Lehn <michael.lehn@uni-ulm.de> - 0.4.1-1
- Initial package
