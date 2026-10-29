      DOUBLE PRECISION FUNCTION DSDOT( N, SX, INCX, SY, INCY )

      INTEGER            INCX, INCY, N
      REAL               SX( * ), SY( * )

      DOUBLE PRECISION   TEMP
      EXTERNAL           DSDOT_SUB

      CALL DSDOT_SUB( N, SX, INCX, SY, INCY, TEMP )
      DSDOT = TEMP

      RETURN
      END
