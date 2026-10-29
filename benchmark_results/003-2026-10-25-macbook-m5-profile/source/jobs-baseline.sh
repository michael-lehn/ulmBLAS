#!/bin/sh
#  jobs-baseline.sh : job list of the baseline breakdown (run P1), one job per
#  line: NAME PROGRAM ARGS.  Small problems: all modes in one program run;
#  large problems (2mnk > 2.2e9): one program run per mode.
sizes="64,64,64 128,128,128 192,192,192 256,256,256 384,384,384 512,512,512
768,768,768 1024,1024,1024 1536,1536,1536 2048,2048,2048 3072,3072,3072
4096,4096,4096 2000,2000,32 2000,2000,128 4000,4000,64 64,2048,2048
2048,64,2048 199,199,199 1000,1000,1000"
for s in $sizes; do
    set -- $(echo $s | tr , ' '); m=$1; n=$2; k=$3; t=$m-$n-$k
    big=$(awk "BEGIN{print (2*$m*$n*$k>2.2e9)}")
    if [ $big = 0 ]; then
        echo "ulm-$t prof-master -m lib,inst,nopack,pack,kern $m $n $k"
        echo "ulmblis-$t prof-master -o blis -m inst,nopack $m $n $k"
    else
        for mo in lib inst nopack pack kern; do
            echo "ulm-$mo-$t prof-master -m $mo $m $n $k"
        done
        echo "ulmblis-inst-$t prof-master -o blis -m inst $m $n $k"
        echo "ulmblis-nopack-$t prof-master -o blis -m nopack $m $n $k"
    fi
    echo "blis-$t prof-blis -m cblas $m $n $k"
    echo "openblas-$t prof-openblas -m cblas $m $n $k"
done
