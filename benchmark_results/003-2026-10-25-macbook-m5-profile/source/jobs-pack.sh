#!/bin/sh
#  jobs-pack.sh : job list of the packing experiment (run P2): the driver
#  (ulmBLAS order and block sizes) with the packing variants of packvar.h,
#  modes inst and pack, one program run per variant and size.
sizes="128,128,128 256,256,256 512,512,512 1024,1024,1024 2048,2048,2048
4096,4096,4096 2000,2000,32 2048,64,2048 64,2048,2048 199,199,199"
for s in $sizes; do
    set -- $(echo $s | tr , ' '); t=$1-$2-$3
    for v in ulm:ulm unit:unit col:unit unit:trn col:trn; do
        echo "$(echo $v | tr : _)-$t prof-exp -p $v -m inst,pack $1 $2 $3"
    done
done
