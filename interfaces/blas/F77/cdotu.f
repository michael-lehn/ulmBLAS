      COMPLEX FUNCTION CDOTU(N,CX,INCX,CY,INCY)

      INTEGER            INCX, INCY, N
      REAL               CX( * ), CY( * )

      COMPLEX            CTEMP
      EXTERNAL           CDOTU_SUB

      CALL CDOTU_SUB( N, CX, INCX, CY, INCY, CTEMP )
      CDOTU = CTEMP

      RETURN
      END
