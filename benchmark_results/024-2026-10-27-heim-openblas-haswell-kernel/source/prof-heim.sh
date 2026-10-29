#!/bin/sh
#  prof-heim.sh W: prof-hsw (prof-ob.cc of campaign 016 for OpenBLAS's
#  HASWELL kernel) on heim, core 3, under the idle gate of obhsw-heim.sh:
#  breakdown of ulmBLAS DGEMM with the kernel into packing and macro kernel
#  (inst), without packing (nopack), with one kernel call per packed block
#  as OpenBLAS's driver (macro, macronopack), the kernel alone (kern) and
#  native OpenBLAS (cblas, prof-hsw-cblas).  Block sizes 512, 256, 13824.
#  3 rounds; macrocheck before.  Programs prof-hsw, prof-hsw-cblas in W.
set -u
W=$1; cd $W || exit 1
mkdir -p res logs
CPU=3; LOADMAX=1.8
idle() { [ "$(awk '$1=="procs_running" {print $2}' /proc/stat)" -le 1 ]; }
gate() {
    until awk -v l=$(cut -d' ' -f1 /proc/loadavg) -v x=$LOADMAX 'BEGIN { exit !(l < x) }' \
          && idle && sleep 1 && idle && sleep 1 && idle; do
        sleep 10
    done
}
echo "start $(date -Iseconds) load $(cat /proc/loadavg)" > logs/run.log
sha256sum prof-hsw prof-hsw-cblas >> logs/run.log
for n in 512 1024 2000 4000; do
    ./prof-hsw -b 512,256,13824 -m macrocheck $n $n $n > res/mcheck-$n.txt 2>&1
done
for r in 1 2 3; do
    for n in 512 1024 2000 4000; do
        gate; taskset -c $CPU ./prof-hsw -b 512,256,13824 -m lib,inst,nopack,macro,macronopack,kern $n $n $n > res/prof-$n-$r.txt 2>&1
        gate; taskset -c $CPU ./prof-hsw-cblas -b 512,256,13824 -m cblas $n $n $n > res/cblas-$n-$r.txt 2>&1
        echo "$(date -Iseconds) round $r n $n done, load $(cut -d' ' -f1 /proc/loadavg)" >> logs/run.log
    done
done
echo "end $(date -Iseconds)" >> logs/run.log
