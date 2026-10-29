      REAL FUNCTION SDSDOT( N, SB, SX, INCX, SY, INCY )

      INTEGER            INCX, INCY, N
      REAL               SB, SX( * ), SY( * )

      REAL               TEMP
      EXTERNAL           SDSDOT_SUB

      CALL SDSDOT_SUB( N, SB, SX, INCX, SY, INCY, TEMP )
      SDSDOT = TEMP

      RETURN
      END
