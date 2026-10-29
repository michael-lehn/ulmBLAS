#!/bin/sh
#  chain.sh B : the experiments fine, rect and blis of this campaign, one
#  after the other (run.sh; see README.md for the design).
set -u
B=${1:?}; S=$(cd "$(dirname "$0")" && pwd)
sq() { for n in "$@"; do printf '%s ' ${n}_${n}_${n}; done; }

ROUNDS=5 VARIANTS="b384_1024_1024 b352_1024_1024 b384_1024_1216 b352_1024_1216 b384_1024_2208 b352_1024_2208 b384_1024_4096" \
SHAPES="$(sq 1000 1024 1025 1032 1040 1050 1064 1100 1150 1200 2000 2048 2049 2056 2064 2100 2150 2200)" \
    sh $S/run.sh $B fine || exit 1

L="1000 1016 1024 1025 1026 1028 1032 1040 1050 1064 1088 1100 1150 1200 2000 2048 2049 2050 2100"
R=""
for x in $L; do R="$R 1000_1000_$x"; done
for x in $L; do [ $x = 1000 ] || R="$R 1000_${x}_1000"; done
for x in 1024 1025 1050 1100 1136 1152 1153 1160 1184 1200 2000; do R="$R ${x}_1000_1000"; done
for x in 2000 2048 2049 2050 2100; do R="$R 2000_2000_$x"; done
ROUNDS=3 VARIANTS="b384_1024_1024 b384_1024_4096 b352_1024_1024 b384_1088_1024 b384_2048_1024" \
SHAPES="$R" sh $S/run.sh $B rect || exit 1

ROUNDS=3 VARIANTS="blis-sme b384_1024_1024 b608_2240_3072" \
SHAPES="$(sq 1000 1024 1025 1050 1100 2000 2048 2049 2050 2100 2200 2240 2241 2250 2300 3072 3073 3100) 2000_2000_2240 2000_2000_2241 2000_2000_2250" \
    sh $S/run.sh $B blis || exit 1
