#!/bin/sh
#  chain3.sh B : further rounds, with the load gate LOADMAX = 4:
#  rect2-r4: round 4 of rect2 (replaces its round 1, which ran from 23:42
#  to 06:12 while other jobs started and stopped; see README.md);
#  blis-r4: rounds 4 to 7 of blis (the configurations with KC = 2240 are
#  bimodal from program run to program run).
set -u
B=${1:?}; S=$(cd "$(dirname "$0")" && pwd)
sq() { for n in "$@"; do printf '%s ' ${n}_${n}_${n}; done; }
export LOADMAX=${LOADMAX:-4}

ROUND0=3 ROUNDS=1 VARIANTS="b384_1024_1024 b384_1024_4096 b352_1024_1024 b384_1088_1024 b384_2048_1024" \
SHAPES="1000_2050_1000 1000_2100_1000 1024_1000_1000 1025_1000_1000 1050_1000_1000 1100_1000_1000 1136_1000_1000 1152_1000_1000 1153_1000_1000 1160_1000_1000 1184_1000_1000 1200_1000_1000 2000_1000_1000 2000_2000_2000 2000_2000_2048 2000_2000_2049 2000_2000_2050 2000_2000_2100" \
    sh $S/run.sh $B rect2-r4 || exit 1

ROUND0=3 ROUNDS=4 VARIANTS="blis-sme b384_1024_1024 b608_2240_3072" \
SHAPES="$(sq 1000 1024 1025 1050 1100 2000 2048 2049 2050 2100 2200 2240 2241 2250 2300 3072 3073 3100) 2000_2000_2240 2000_2000_2241 2000_2000_2250" \
    sh $S/run.sh $B blis-r4 || exit 1
