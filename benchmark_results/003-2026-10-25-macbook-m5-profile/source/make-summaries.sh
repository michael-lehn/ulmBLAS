#!/bin/sh
#  make-summaries.sh CAMPAIGN : regenerate the summaries of the campaign
#  from its raw data: medians over the rounds (summarize.py,
#  summarize-bench.py) in CAMPAIGN/data/summary/medians, derived tables
#  (tables.py) in CAMPAIGN/data/summary.
set -u
C=${1:?}; S=$(cd "$(dirname "$0")" && pwd); R=$C/data/raw; M=$C/data/summary/medians
mkdir -p $M
python3 -I $S/summarize.py $R P1 > $M/P1.csv || exit 1
python3 -I $S/summarize.py $R P2 > $M/P2.csv || exit 1
python3 -I $S/summarize.py $R P3 > $M/P3.csv || exit 1
for r in P4 P5 P6; do python3 -I $S/summarize-bench.py $R/$r > $M/$r.csv || exit 1; done
python3 -I $S/summarize-bench.py $R/heim/pack  > $M/heim_pack.csv || exit 1
python3 -I $S/summarize.py $R/heim prof > $M/heim_prof.csv || exit 1
python3 -I $S/summarize-bench.py $R/heim/ab    > $M/heim_isolation_ab.csv || exit 1
python3 -I $S/summarize-bench.py $R/heim/ab2   > $M/heim_isolation_ab2.csv || exit 1
python3 -I $S/summarize-bench.py $R/heim/ab3   > $M/heim_threshold.csv || exit 1
python3 -I $S/summarize-bench.py $R/heim/final > $M/heim_final.csv || exit 1
python3 -I $S/tables.py $M $C/data/summary
