#!/bin/sh
#  check-heim.sh WORKSPACE: correctness of ulmBLAS febb852 on heim (x86-64):
#  CPU flags, make check (all configurations), make check with ASan+UBSan
#  (SSE and AVX/FMA configurations), Reference-LAPACK 3.12.1 test programs.
set -u
W=$1; cd $W || exit 1
L=$W/logs; mkdir -p $L
{
  echo "date $(date -Iseconds)"; uname -a; cat /etc/debian_version
  gcc --version | head -1; gfortran --version | head -1
  grep -m1 'model name' /proc/cpuinfo; grep -m1 microcode /proc/cpuinfo
  echo "flags: $(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(sse|sse2|sse3|pni|ssse3|sse4_1|sse4_2|avx|avx2|fma|avx512.*|f16c)$' | tr '\n' ' ')"
  echo "governor $(cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor) no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
  echo "load $(cat /proc/loadavg)"; who
} > $L/system.txt 2>&1

rm -rf chk && cp -R febb852 chk
(cd chk && make -j12 check) > $L/check.log 2>&1; echo "make check exit $?" >> $L/status.txt

rm -rf chk-san && cp -R febb852 chk-san
SAN="-fsanitize=address,undefined -fno-sanitize=enum -fno-sanitize-recover=undefined -fno-omit-frame-pointer -g"
(cd chk-san && ASAN_OPTIONS=detect_leaks=1:alloc_dealloc_mismatch=1:abort_on_error=1 \
   UBSAN_OPTIONS=print_stacktrace=1 \
   make -j12 check CONFIGS="testcase01 testcase04 testcase05 testcase06 testcase07" \
        OPTFLAGS=-O1 CXXFLAGS="-Wall $SAN" CFLAGS="$SAN" LDFLAGS="$SAN") \
   > $L/check-san.log 2>&1; echo "sanitizer check exit $?" >> $L/status.txt
grep -rlE "runtime error|ERROR: AddressSanitizer|ERROR: LeakSanitizer" chk-san/test/build \
   > $L/san-reports.txt 2>&1; echo "sanitizer reports: $(wc -l < $L/san-reports.txt)" >> $L/status.txt

for c in testcase04 testcase06; do
  (cd chk && LAPACK_TARBALL=$W/lapack-3.12.1.tar.gz make check-reference-lapack REFERENCE_CONFIG=$c) \
     > $L/reference-lapack-$c.log 2>&1; echo "reference-lapack $c exit $?" >> $L/status.txt
done
echo done >> $L/status.txt
