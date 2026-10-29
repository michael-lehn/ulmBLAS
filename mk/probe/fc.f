*     Probe: Fortran 77 compiler.
      PROGRAM PROBE
      DOUBLE PRECISION X
      COMPLEX*16 Z
      X = 2.0D0
      Z = DCMPLX( X, 1.0D0 )
      IF( DBLE( Z ).NE.X )STOP 1
      END
