#!/bin/sh
#  jobs-p4.sh : job list of run P4, before/after comparison of the packing
#  change through the CBLAS interface: master (d25e3eb), p4 and p8 (gepack
#  with 4 or 8 panels packed together) and BLIS 2.0; the variants of one
#  group of sizes are adjacent (interleaved).
SMALL="16 16 16 32 32 32 64 64 64 100 100 100 128 128 128 192 192 192 199 199 199 256 256 256"
MID="384 384 384 512 512 512 768 768 768 1024 1024 1024"
THIN="2000 2000 32 2000 2000 128 4000 4000 64 2048 64 2048 64 2048 2048 4096 96 4096 96 4096 4096"
V="master p4 p8 blis"
for v in $V; do echo "small-$v bench2-$v 7 1 $SMALL"; done
for v in $V; do echo "mid-$v bench2-$v 7 1 $MID"; done
for n in 2048 4096; do
    for v in $V; do echo "sq$n-$v bench2-$v 5 1 $n $n $n"; done
done
for v in $V; do echo "thin-$v bench2-$v 7 1 $THIN"; done
for n in 512 1024 2048; do
    for v in $V; do echo "l3n$n-$v bench3-$v 5 $n"; done
done
