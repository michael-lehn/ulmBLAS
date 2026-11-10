/*
 *  LSAME of the reference BLAS: true if the characters CA and CB are the
 *  same letter regardless of case.  Not a BLAS routine of ulmBLAS, but
 *  part of every reference BLAS library: LAPACK built against an external
 *  BLAS calls it.  As the reference, only the first character counts; the
 *  hidden lengths of the CHARACTER arguments are not used.  LOGICAL is
 *  int (gfortran default).
 */
int
lsame_(const char *ca, const char *cb)
{
    int a = (unsigned char)*ca, b = (unsigned char)*cb;

    if (a >= 'a' && a <= 'z') {
	a -= 'a' - 'A';
    }
    if (b >= 'a' && b <= 'z') {
	b -= 'a' - 'A';
    }
    return a == b;
}
