      COMPLEX FUNCTION CDOTC(N,CX,INCX,CY,INCY)

      INTEGER            INCX, INCY, N
      REAL               CX( * ), CY( * )

      COMPLEX            CTEMP
      EXTERNAL           CDOTC_SUB

      CALL CDOTC_SUB( N, CX, INCX, CY, INCY, CTEMP )
      CDOTC = CTEMP

      RETURN
      END
