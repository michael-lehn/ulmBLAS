#!/bin/sh
#  variants.sh SRC NAME "DEFINES" ...: builds libulmblas.so of SRC with each
#  set of -D flags as lib-NAME.so (in /tmp/l2study), one after the other
#  (each build uses all cores)
src=$1
shift
while [ $# -ge 2 ]; do
    name=$1
    defs=$2
    shift 2
    rm -rf /tmp/l2study/var-$name
    cp -r $src /tmp/l2study/var-$name
    (cd /tmp/l2study/var-$name &&
	make -j12 shared CXX=g++ CC=gcc FC=gfortran ARCHFLAGS=-march=native \
	    CPPFLAGS="$defs" > build.log 2>&1 &&
	cp libulmblas.so /tmp/l2study/lib-$name.so) || echo "build $name failed"
done
