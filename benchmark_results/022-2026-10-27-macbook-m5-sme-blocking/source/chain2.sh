#!/bin/sh
#  chain2.sh B : the experiments rect2 (re-measurement of the 18 shapes of
#  rect that had program runs between 22:23 and 22:34, when other load
#  disturbed the measurements) and blis (from scratch; the first attempt,
#  blis-aborted, was stopped during the second round because of other
#  jobs on the machine), both with the load gate LOADMAX = 4 of guard.sh.
set -u
B=${1:?}; S=$(cd "$(dirname "$0")" && pwd)
sq() { for n in "$@"; do printf '%s ' ${n}_${n}_${n}; done; }
export LOADMAX=${LOADMAX:-4}

ROUNDS=3 VARIANTS="b384_1024_1024 b384_1024_4096 b352_1024_1024 b384_1088_1024 b384_2048_1024" \
SHAPES="1000_2050_1000 1000_2100_1000 1024_1000_1000 1025_1000_1000 1050_1000_1000 1100_1000_1000 1136_1000_1000 1152_1000_1000 1153_1000_1000 1160_1000_1000 1184_1000_1000 1200_1000_1000 2000_1000_1000 2000_2000_2000 2000_2000_2048 2000_2000_2049 2000_2000_2050 2000_2000_2100" \
    sh $S/run.sh $B rect2 || exit 1

ROUNDS=3 VARIANTS="blis-sme b384_1024_1024 b608_2240_3072" \
SHAPES="$(sq 1000 1024 1025 1050 1100 2000 2048 2049 2050 2100 2200 2240 2241 2250 2300 3072 3073 3100) 2000_2000_2240 2000_2000_2241 2000_2000_2250" \
    sh $S/run.sh $B blis || exit 1
