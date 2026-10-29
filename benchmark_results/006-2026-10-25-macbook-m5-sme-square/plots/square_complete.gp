load "_style.gp"
set terminal svg size 1250,960 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "square_complete.svg"
VARS = ALL; ERR = 0
TITLE = "DGEMM, square matrices, Apple M5 (MacBook Air), one performance core, double precision\nAccelerate, BLIS SME2, ulmBLAS NEON and all 12 fixed ulmBLAS SME2 configurations (medians over the rounds)"
load "_panels.gp"
