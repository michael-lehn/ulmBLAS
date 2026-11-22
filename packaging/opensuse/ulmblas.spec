#
# spec file for package ulmblas
#
# Copyright (c) 2026 Michael Lehn
#
# All modifications and additions to the file contributed by third parties
# remain the property of their copyright owners, unless otherwise agreed
# upon. The license for this file, and modifications and additions to the
# file, is the same license as for the pristine package itself (unless the
# license for the pristine package is not an Open Source License, in which
# case the license is the MIT License).
#

%define sover 0
Name:           ulmblas
Version:        0.4.1
Release:        0
Summary:        BLAS with micro kernels selected for the CPU at load time
License:        BSD-3-Clause
Group:          Productivity/Scientific/Math
URL:            https://github.com/michael-lehn/ulmBLAS
Source0:        https://github.com/michael-lehn/ulmBLAS/archive/refs/tags/v%{version}.tar.gz#/ulmBLAS-%{version}.tar.gz
BuildRequires:  gcc-c++
BuildRequires:  gcc-fortran
BuildRequires:  make
ExclusiveArch:  x86_64 aarch64
# the backends in %%{_libdir}/ulmblas are loaded by libulmblas.so.0 only
%global __provides_exclude_from ^%{_libdir}/ulmblas/libulmblas_.*\\.so$
# the assembly kernels and the dispatching library are not built for LTO
%define _lto_cflags %{nil}

%description
ulmBLAS implements all routines of the reference BLAS (Basic Linear
Algebra Subprograms) for single, double, complex and double complex
precision with the Fortran 77 and the C (CBLAS) interface.

%package -n libulmblas%{sover}
Summary:        BLAS with micro kernels selected for the CPU at load time
Group:          System/Libraries
Requires(post): update-alternatives
Requires(postun): update-alternatives

%description -n libulmblas%{sover}
ulmBLAS implements all routines of the reference BLAS (Basic Linear
Algebra Subprograms) with the Fortran 77 and the C (CBLAS) interface.  On
x86-64 libulmblas.so.0 selects at load time the build of the kernels for
the CPU (AVX-512, AVX2 and FMA, SSE3 or generic x86-64); on aarch64 it
uses NEON kernels.  ulmBLAS is an alternative for libblas.so.3 and
libcblas.so.3.

%package devel
Summary:        Development files for ulmBLAS
Group:          Development/Libraries/C and C++
Requires:       libulmblas%{sover} = %{version}

%description devel
The link libulmblas.so, the header cblas.h and the pkg-config file
ulmblas.pc of ulmBLAS.

%prep
%autosetup -n ulmBLAS-%{version}

%build
export ULMBLAS_VERSION=%{version}
export ULMBLAS_LIBDIR=%{_libdir}
export ULMBLAS_BACKENDDIR=%{_libdir}/ulmblas
export ULMBLAS_INCLUDEDIR=%{_includedir}/ulmblas
export ULMBLAS_RPATH=no
export ULMBLAS_STATIC_LIBSTDCXX=no
# libulmblas64 (64-bit integers under the usual names) is not packaged yet
export ULMBLAS_INTERFACE64=no
%ifarch x86_64
export ULMBLAS_SONAMES=yes
%endif
export FC=none
export MAKE="make %{?_smp_mflags}"
ULMBLAS_STEP=build sh interfaces/install-shared.sh %{_prefix}
%ifarch aarch64
# one library without backends: its objects linked a second time, with
# the SONAMEs (shared object names) libblas.so.3 and libcblas.so.3 of the
# alternatives
for l in blas cblas; do
    $MAKE -C interfaces/blas/shared SHARED_NAME=$l SHARED_VERSION=3 \
        STATIC_LIBSTDCXX= OUTDIR=$PWD/build.alternatives
done
%endif

%install
export ULMBLAS_VERSION=%{version}
export ULMBLAS_LIBDIR=%{_libdir}
export ULMBLAS_BACKENDDIR=%{_libdir}/ulmblas
export ULMBLAS_INCLUDEDIR=%{_includedir}/ulmblas
export ULMBLAS_RPATH=no
export ULMBLAS_STATIC_LIBSTDCXX=no
# libulmblas64 (64-bit integers under the usual names) is not packaged yet
export ULMBLAS_INTERFACE64=no
%ifarch x86_64
export ULMBLAS_SONAMES=yes
%endif
export FC=none
export MAKE="make %{?_smp_mflags}"
DESTDIR=%{buildroot} sh interfaces/install-shared.sh %{_prefix}
%ifarch aarch64
install -d %{buildroot}%{_libdir}/ulmblas
install -m 0755 build.alternatives/libblas.so.3 \
    build.alternatives/libcblas.so.3 %{buildroot}%{_libdir}/ulmblas/
%endif
# the alternatives libblas.so.3_ARCH and libcblas.so.3_ARCH (the groups
# of openSUSE's lapack and openblas)
install -d %{buildroot}%{_sysconfdir}/alternatives
ln -s %{_sysconfdir}/alternatives/libblas.so.3_%{_arch} \
    %{buildroot}%{_libdir}/libblas.so.3
ln -s %{_sysconfdir}/alternatives/libcblas.so.3_%{_arch} \
    %{buildroot}%{_libdir}/libcblas.so.3
touch %{buildroot}%{_sysconfdir}/alternatives/libblas.so.3_%{_arch}
touch %{buildroot}%{_sysconfdir}/alternatives/libcblas.so.3_%{_arch}

%check
FC=gfortran sh packaging/check.sh

%post -n libulmblas%{sover}
%{_sbindir}/update-alternatives --install %{_libdir}/libblas.so.3 libblas.so.3_%{_arch} %{_libdir}/ulmblas/libblas.so.3 15
%{_sbindir}/update-alternatives --install %{_libdir}/libcblas.so.3 libcblas.so.3_%{_arch} %{_libdir}/ulmblas/libcblas.so.3 15
/sbin/ldconfig

%postun -n libulmblas%{sover}
/sbin/ldconfig
if [ ! -f %{_libdir}/ulmblas/libblas.so.3 ]; then
    %{_sbindir}/update-alternatives --remove libblas.so.3_%{_arch} %{_libdir}/ulmblas/libblas.so.3
fi
if [ ! -f %{_libdir}/ulmblas/libcblas.so.3 ]; then
    %{_sbindir}/update-alternatives --remove libcblas.so.3_%{_arch} %{_libdir}/ulmblas/libcblas.so.3
fi

%files -n libulmblas%{sover}
%license LICENSE NOTICE
%{_libdir}/libulmblas.so.%{sover}
%{_libdir}/ulmblas/
%{_libdir}/libblas.so.3
%{_libdir}/libcblas.so.3
%ghost %{_sysconfdir}/alternatives/libblas.so.3_%{_arch}
%ghost %{_sysconfdir}/alternatives/libcblas.so.3_%{_arch}

%files devel
%doc README.md
%{_libdir}/libulmblas.so
%{_includedir}/ulmblas/
%{_libdir}/pkgconfig/ulmblas.pc

%changelog
