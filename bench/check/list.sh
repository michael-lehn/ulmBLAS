#!/bin/sh
#  list.sh: the runs of the full check, one per line: NAME TESTER ARGS...
#  (every routine of level 1, 2, 3, types s d c z, modes cb and f7; per
#  routine three ranges of sizes: small (all sizes up to about 20, all
#  combinations of the options), mid and big; increments 1, 2, 3 and
#  negative ones; several alpha and beta including 0 and 1)
INC1="-X 6 1 2 3 -1 -2 -3 -Y 6 1 2 3 -1 -2 -3"
INC2="-X 4 1 2 -1 -3 -Y 4 1 2 -1 -3"
for X in cb f7; do
for p in s d c z; do
    case $p in
    s|d) A="-a 5 0.0 1.0 -1.0 0.7 -1.5"; A2="-a 3 0.0 1.0 -1.5"; B2="-b 3 0.0 1.0 1.3"
         AB1="-a 1 -1.5 -b 1 1.3"
         L1="amax asum axpy copy dot nrm2 rot rotg rotm rotmg scal swap"
         [ $p = s ] && L1="$L1 sdsdot dsdot"
         L2="gemv gbmv symv sbmv spmv trmv tbmv tpmv trsv tbsv tpsv ger syr spr syr2 spr2"
         L3="gemm symm syrk syr2k trmm trsm"; TR="-A 2 N T -B 2 N T"
         AB="-a 3 0.0 1.0 -1.5 -b 3 0.0 1.0 1.3" ;;
    *)   A="-a 5 0.0 0.0 1.0 0.0 -1.0 0.0 0.7 -1.5 0.0 1.0"; A2="-a 3 0.0 0.0 1.0 0.0 -1.5 0.7"
         B2="-b 3 0.0 0.0 1.0 0.0 1.3 -0.4"; AB1="-a 1 -1.5 0.7 -b 1 1.3 -0.4"
         L1="amax asum axpy copy dotc dotu nrm2 rot rotg scal rscal swap"
         L2="gemv gbmv hemv hbmv hpmv trmv tbmv tpmv trsv tbsv tpsv geru gerc her hpr her2 hpr2"
         L3="gemm symm hemm syrk herk syr2k her2k trmm trsm"; TR="-A 3 N T C -B 3 N T C"
         AB="-a 3 0.0 0.0 1.0 0.0 -1.5 0.7 -b 3 0.0 0.0 1.0 0.0 1.3 -0.4" ;;
    esac
    T1=$X-x${p}l1blastst; T2=$X-x${p}l2blastst; T3=$X-x${p}l3blastst
    for r in $L1; do
        echo "$X-$p-l1-$r-small $T1 -R $r -N 1 100 1 $INC1 $A"
        echo "$X-$p-l1-$r-mid $T1 -R $r -N 100 2000 100 $INC1 $A"
        echo "$X-$p-l1-$r-big $T1 -R $r -N 10000 100000 30000 $INC1 $A"
    done
    for r in $L2; do
        case $p$r in [sd]*mv|[sd]*sv) TA="-A 2 n t" ;; *mv|*sv) TA="-A 3 n t c" ;; *) TA="" ;; esac
        G="-R $r -U 2 L U $TA -D 2 N U"
        echo "$X-$p-l2-$r-small $T2 $G -M 1 20 1 -N 1 20 1 -P 0 4 1 -Q 0 4 1 $INC2 $A2 $B2"
        echo "$X-$p-l2-$r-mid $T2 $G -M 21 140 17 -N 21 140 17 -P 0 4 2 -Q 0 4 2 $INC2 $A2 $B2"
        echo "$X-$p-l2-$r-big $T2 $G -M 200 1000 200 -N 200 1000 200 -P 3 3 1 -Q 2 2 1 -X 2 1 -2 -Y 2 1 -3 $AB1"
    done
    for r in $L3; do
        G="-R $r -S 2 L R -U 2 L U $TR -D 2 N U"
        echo "$X-$p-l3-$r-small $T3 $G -M 1 24 1 -N 1 24 1 -K 1 24 7 $AB"
        echo "$X-$p-l3-$r-mid $T3 $G -M 25 300 25 -N 25 300 25 -K 25 300 91 $AB1"
        echo "$X-$p-l3-$r-big $T3 $G -M 500 1500 1000 -N 500 1500 1000 -K 500 1500 1000 $AB1"
    done
done
done
