#!/bin/sh
#  jobs-p3.sh : job list of run P3:
#  (a) packing variants (packvar.h) with the ulmBLAS order and block sizes;
#  (b) block sizes and loop order with the packing variant blk4:unit;
#  (c) call overhead for small problems: library, driver and memory pool.
sizes="128,128,128 199,199,199 256,256,256 512,512,512 1024,1024,1024
2048,2048,2048 4096,4096,4096 2000,2000,32 2048,64,2048 64,2048,2048"
for s in $sizes; do
    set -- $(echo $s | tr , ' '); t=$1-$2-$3
    for v in ulm:ulm unit:unit blk4:unit blk8:unit blk4:trn blk8:trn; do
        echo "pk-$(echo $v | tr : _)-$t prof-exp -p $v -m inst $1 $2 $3"
    done
done
sizes="1024,1024,1024 2048,2048,2048 3072,3072,3072 4096,4096,4096
2000,2000,32 4000,4000,64"
for s in $sizes; do
    set -- $(echo $s | tr , ' '); t=$1-$2-$3
    for b in 960,1024,4080 960,1024,8160 480,2048,4080 320,3072,4080; do
        echo "bs-ulm-$(echo $b | tr , _)-$t prof-exp -p blk4:unit -b $b -m inst $1 $2 $3"
    done
    echo "bs-blis-256_3072_8184-$t prof-exp -p blk4:unit -o blis -m inst $1 $2 $3"
done
for n in 8 16 32 64; do
    echo "small-$n-$n-$n prof-exp -m lib,inst,pool $n $n $n"
done
