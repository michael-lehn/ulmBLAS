load "_style.gp"
load "_selection.gp"
set terminal svg size 1250,960 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "square_readable.svg"
VARS = SEL; ERR = 1
TITLE = "DGEMM, square matrices, Apple M5 (MacBook Air), one performance core, double precision\nAccelerate, BLIS SME2, ulmBLAS NEON and selected fixed ulmBLAS SME2 configurations (medians over the rounds)"
load "_panels.gp"
