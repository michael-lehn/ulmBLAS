      PROGRAM UBLAT1
*
*     Tests of the Fortran 77 interface of ulmBLAS that the Netlib test
*     programs do not cover (written for ulmBLAS, output in the format of
*     the Netlib level 1 test programs):
*
*       1  CROTG, ZROTG    hand-computed rotations (LAPACK 3.10 or newer
*                          conventions) and their defining identities
*       2  CSROT, ZDROT    hand-computed results, negative increments
*       3  SCABS1, DCABS1  exact values
*       4  XERBLA          name, length of the name (hidden argument of a
*                          CHARACTER*(*) dummy) and INFO for invalid
*                          arguments of routines of all levels and types
*
*     .. Parameters ..
      INTEGER           NOUT
      PARAMETER         (NOUT=6)
*     .. Scalars in Common ..
      LOGICAL           PASS
*     .. Local Scalars ..
      INTEGER           IC
*     .. Local Arrays ..
      CHARACTER*14      L(4)
*     .. Common blocks ..
      COMMON            /UCOMB/PASS
*     .. Data statements ..
      DATA              L/'CROTG, ZROTG', 'CSROT, ZDROT',
     +                  'SCABS1, DCABS1', 'XERBLA'/
*     .. Executable Statements ..
      WRITE (NOUT,99999)
      DO 10 IC = 1, 4
         WRITE (NOUT,99998) IC, L(IC)
         PASS = .TRUE.
         IF (IC.EQ.1) CALL CHKROTG
         IF (IC.EQ.2) CALL CHKROT
         IF (IC.EQ.3) CALL CHKABS1
         IF (IC.EQ.4) CALL CHKERB
         IF (PASS) WRITE (NOUT,99997)
   10 CONTINUE
      STOP
*
99999 FORMAT (' ulmBLAS Fortran 77 interface: additional tests',/1X)
99998 FORMAT (/' Test of subprogram number',I3,12X,A14)
99997 FORMAT ('                                    ----- PASS -----')
      END

      SUBROUTINE FAIL(MSG)
*     Reports a failed check.
      CHARACTER*(*)     MSG
      LOGICAL           PASS
      COMMON            /UCOMB/PASS
      IF (PASS) WRITE (6,'(39X,A)') 'FAIL'
      PASS = .FALSE.
      WRITE (6,'(1X,A)') MSG
      RETURN
      END

      LOGICAL FUNCTION NEARC(A, B, TOL)
*     |A - B| <= TOL for double complex A, B
      DOUBLE COMPLEX    A, B
      DOUBLE PRECISION  TOL
      NEARC = ABS(A-B).LE.TOL
      RETURN
      END

      SUBROUTINE CHKROTG
*     CROTG, ZROTG: c real, c*a + s*b = r, -conjg(s)*a + c*b = 0 and the
*     values of the Reference BLAS since LAPACK 3.10 (r = |b| and
*     s = conjg(b)/|b| for a = 0, c = 1 and s = 0 for b = 0, otherwise r with
*     the phase of a).  b is not modified.
      INTEGER           NC
      PARAMETER         (NC=5)
      COMPLEX           CA, CB, CB0, CS
      REAL              CC
      DOUBLE COMPLEX    ZA, ZB, ZB0, ZS, A(NC), B(NC), R(NC), S(NC)
      DOUBLE PRECISION  ZC, C(NC), H, Q, TOLS, TOLD
      INTEGER           K
      LOGICAL           NEARC
      EXTERNAL          CROTG, ZROTG, NEARC, FAIL
      H = 1D0/SQRT(2D0)
      Q = SQRT(2D0)
      TOLS = 1D-5
      TOLD = 1D-13
*     a, b, r, c, s
      A(1) = (3D0,4D0)
      B(1) = (0D0,0D0)
      R(1) = (3D0,4D0)
      C(1) = 1D0
      S(1) = (0D0,0D0)
      A(2) = (0D0,0D0)
      B(2) = (3D0,-4D0)
      R(2) = (5D0,0D0)
      C(2) = 0D0
      S(2) = (0.6D0,0.8D0)
      A(3) = (3D0,0D0)
      B(3) = (0D0,4D0)
      R(3) = (5D0,0D0)
      C(3) = 0.6D0
      S(3) = (0D0,-0.8D0)
      A(4) = (1D0,1D0)
      B(4) = (1D0,-1D0)
      R(4) = DCMPLX(Q,Q)
      C(4) = H
      S(4) = DCMPLX(0D0,H)
      A(5) = (-2D0,0D0)
      B(5) = (0D0,-2D0)
      R(5) = DCMPLX(-2D0*Q,0D0)
      C(5) = H
      S(5) = DCMPLX(0D0,-H)
      DO 10 K = 1, NC
*        single precision
         CA = CMPLX(A(K))
         CB = CMPLX(B(K))
         CB0 = CB
         CALL CROTG(CA, CB, CC, CS)
         IF (.NOT.(NEARC(DCMPLX(CA),R(K),4*TOLS)
     +             .AND. ABS(CC-C(K)).LE.TOLS
     +             .AND. NEARC(DCMPLX(CS),S(K),TOLS)
     +             .AND. CB.EQ.CB0)) THEN
            CALL FAIL('CROTG: wrong r, c or s, or b modified')
            WRITE (6,*) ' case', K, ' r', CA, ' c', CC, ' s', CS
         END IF
*        double precision
         ZA = A(K)
         ZB = B(K)
         ZB0 = ZB
         CALL ZROTG(ZA, ZB, ZC, ZS)
         IF (.NOT.(NEARC(ZA,R(K),4*TOLD) .AND. ABS(ZC-C(K)).LE.TOLD
     +             .AND. NEARC(ZS,S(K),TOLD) .AND. ZB.EQ.ZB0)) THEN
            CALL FAIL('ZROTG: wrong r, c or s, or b modified')
            WRITE (6,*) ' case', K, ' r', ZA, ' c', ZC, ' s', ZS
         END IF
*        the defining identities
         IF (.NOT.(NEARC(ZC*A(K) + ZS*B(K), ZA, 4*TOLD)
     +       .AND. NEARC(-DCONJG(ZS)*A(K) + ZC*B(K), (0D0,0D0),
     +                   4*TOLD))) THEN
            CALL FAIL('ZROTG: c*a + s*b = r, -conjg(s)*a + c*b = 0 '//
     +                'violated')
            WRITE (6,*) ' case', K
         END IF
   10 CONTINUE
      RETURN
      END

      SUBROUTINE CHKROT
*     CSROT, ZDROT: (x,y) = (c*x + s*y, c*y - s*x) with real c = 0.6,
*     s = 0.8, for increments (1,1), (2,-1) and (-1,-2).  Results are
*     exact in binary floating point up to rounding of 0.6 and 0.8.
      INTEGER           N
      PARAMETER         (N=3)
      COMPLEX           CX(2*N), CY(2*N)
      DOUBLE COMPLEX    ZX(2*N), ZY(2*N), X0(N), Y0(N), EX, EY
      INTEGER           I, J, IX, IY, KX, KY, INCX(3), INCY(3)
      LOGICAL           NEARC
      EXTERNAL          CSROT, ZDROT, NEARC, FAIL
      DATA              INCX/1, 2, -1/, INCY/1, -1, -2/
      X0(1) = (1D0,2D0)
      X0(2) = (-3D0,0.5D0)
      X0(3) = (0D0,-1D0)
      Y0(1) = (4D0,-1D0)
      Y0(2) = (2D0,2D0)
      Y0(3) = (-0.5D0,3D0)
      DO 30 J = 1, 3
         DO 10 I = 1, 2*N
            CX(I) = (99.0,99.0)
            CY(I) = (99.0,99.0)
            ZX(I) = (99D0,99D0)
            ZY(I) = (99D0,99D0)
   10    CONTINUE
         KX = 1
         KY = 1
         IF (INCX(J).LT.0) KX = 1 + (1-N)*INCX(J)
         IF (INCY(J).LT.0) KY = 1 + (1-N)*INCY(J)
         IX = KX
         IY = KY
         DO 20 I = 1, N
            CX(IX) = CMPLX(X0(I))
            CY(IY) = CMPLX(Y0(I))
            ZX(IX) = X0(I)
            ZY(IY) = Y0(I)
            IX = IX + INCX(J)
            IY = IY + INCY(J)
   20    CONTINUE
         CALL CSROT(N, CX, INCX(J), CY, INCY(J), 0.6, 0.8)
         CALL ZDROT(N, ZX, INCX(J), ZY, INCY(J), 0.6D0, 0.8D0)
         IX = KX
         IY = KY
         DO 25 I = 1, N
            EX = 0.6D0*X0(I) + 0.8D0*Y0(I)
            EY = 0.6D0*Y0(I) - 0.8D0*X0(I)
            IF (.NOT.(NEARC(DCMPLX(CX(IX)),EX,1D-5)
     +                .AND. NEARC(DCMPLX(CY(IY)),EY,1D-5))) THEN
               CALL FAIL('CSROT: wrong result')
               WRITE (6,*) ' incx', INCX(J), ' incy', INCY(J), ' i', I
            END IF
            IF (.NOT.(NEARC(ZX(IX),EX,1D-14)
     +                .AND. NEARC(ZY(IY),EY,1D-14))) THEN
               CALL FAIL('ZDROT: wrong result')
               WRITE (6,*) ' incx', INCX(J), ' incy', INCY(J), ' i', I
            END IF
            IX = IX + INCX(J)
            IY = IY + INCY(J)
   25    CONTINUE
*        elements between the strided ones are not modified
         DO 28 I = 1, 2*N
            IF (MOD(I-1,ABS(INCX(J))).NE.0 .OR. I.GT.N*ABS(INCX(J)))
     +      THEN
               IF (CX(I).NE.(99.0,99.0) .OR. ZX(I).NE.(99D0,99D0))
     +            CALL FAIL('CSROT, ZDROT: x modified between elements')
            END IF
            IF (MOD(I-1,ABS(INCY(J))).NE.0 .OR. I.GT.N*ABS(INCY(J)))
     +      THEN
               IF (CY(I).NE.(99.0,99.0) .OR. ZY(I).NE.(99D0,99D0))
     +            CALL FAIL('CSROT, ZDROT: y modified between elements')
            END IF
   28    CONTINUE
   30 CONTINUE
      RETURN
      END

      SUBROUTINE CHKABS1
*     SCABS1, DCABS1: |re(z)| + |im(z)|
      REAL              SCABS1
      DOUBLE PRECISION  DCABS1
      EXTERNAL          SCABS1, DCABS1, FAIL
      IF (SCABS1((3.0,-4.0)).NE.7.0 .OR.
     +    SCABS1((-1.5,2.25)).NE.3.75 .OR.
     +    SCABS1((0.0,0.0)).NE.0.0) CALL FAIL('SCABS1: wrong result')
      IF (DCABS1((3D0,-4D0)).NE.7D0 .OR.
     +    DCABS1((-1.5D0,2.25D0)).NE.3.75D0 .OR.
     +    DCABS1((0D0,-6D0)).NE.6D0) CALL FAIL('DCABS1: wrong result')
      RETURN
      END

      SUBROUTINE CHKERB
*     Invalid arguments: XERBLA (below) must receive the name of the
*     routine with its length 6 and the number of the invalid argument.
      INTEGER           INFOT, LENGTH
      CHARACTER*32      NAME
      LOGICAL           CALLED
      COMMON            /UERB/INFOT, LENGTH, NAME, CALLED
      REAL              SA(1)
      DOUBLE PRECISION  DA(1)
      COMPLEX           CA(1)
      DOUBLE COMPLEX    ZA(1)
      EXTERNAL          SGEMV, DTRSV, CHER, ZGEMM, DTRSM, CHER2K,
     +                  SSYMV, ZTBSV, EXPECT
*     level 2
      CALL RESET
      CALL SGEMV('X', 1, 1, 1.0, SA, 1, SA, 1, 0.0, SA, 1)
      CALL EXPECT('SGEMV ', 1)
      CALL RESET
      CALL SSYMV('U', -1, 1.0, SA, 1, SA, 1, 0.0, SA, 1)
      CALL EXPECT('SSYMV ', 2)
      CALL RESET
      CALL DTRSV('U', 'N', 'N', 2, DA, 1, DA, 1)
      CALL EXPECT('DTRSV ', 6)
      CALL RESET
      CALL CHER('L', 1, 1.0, CA, 0, CA, 1)
      CALL EXPECT('CHER  ', 5)
      CALL RESET
      CALL ZTBSV('U', 'N', 'N', 1, -1, ZA, 1, ZA, 1)
      CALL EXPECT('ZTBSV ', 5)
*     level 3
      CALL RESET
      CALL ZGEMM('N', 'N', -1, 1, 1, (1D0,0D0), ZA, 1, ZA, 1,
     +           (0D0,0D0), ZA, 1)
      CALL EXPECT('ZGEMM ', 3)
      CALL RESET
      CALL DTRSM('L', 'U', 'N', 'N', 2, 1, 1D0, DA, 1, DA, 2)
      CALL EXPECT('DTRSM ', 9)
      CALL RESET
      CALL CHER2K('U', 'T', 1, 1, (1.0,0.0), CA, 1, CA, 1, 0.0, CA, 1)
      CALL EXPECT('CHER2K', 2)
      RETURN
      END

      SUBROUTINE RESET
      INTEGER           INFOT, LENGTH
      CHARACTER*32      NAME
      LOGICAL           CALLED
      COMMON            /UERB/INFOT, LENGTH, NAME, CALLED
      INFOT = 0
      LENGTH = 0
      NAME = ' '
      CALLED = .FALSE.
      RETURN
      END

      SUBROUTINE EXPECT(SRNAME, INFO)
      CHARACTER*6       SRNAME
      INTEGER           INFO
      INTEGER           INFOT, LENGTH
      CHARACTER*32      NAME
      LOGICAL           CALLED
      COMMON            /UERB/INFOT, LENGTH, NAME, CALLED
      EXTERNAL          FAIL
      IF (.NOT.CALLED) THEN
         CALL FAIL('XERBLA not called by '//SRNAME)
      ELSE IF (LENGTH.NE.6 .OR. NAME(1:6).NE.SRNAME
     +         .OR. INFOT.NE.INFO) THEN
         CALL FAIL('XERBLA called with a wrong name, length or INFO '//
     +             'by '//SRNAME)
         WRITE (6,*) ' length', LENGTH, ' INFO', INFOT, ' expected 6',
     +               INFO
      END IF
      RETURN
      END

      SUBROUTINE XERBLA(SRNAME, INFO)
*     Replaces the XERBLA of the library (as in the Reference BLAS, the
*     name is a CHARACTER*(*) dummy argument).  Records the arguments.
      CHARACTER*(*)     SRNAME
      INTEGER           INFO
      INTEGER           INFOT, LENGTH
      CHARACTER*32      NAME
      LOGICAL           CALLED
      COMMON            /UERB/INFOT, LENGTH, NAME, CALLED
      CALLED = .TRUE.
      INFOT = INFO
      LENGTH = LEN(SRNAME)
      IF (LENGTH.GE.1 .AND. LENGTH.LE.32) THEN
         NAME = SRNAME
      ELSE
         NAME = '?'
      END IF
      RETURN
      END
