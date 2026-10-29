#ifndef ULMBLAS_LEVEL1_ROT_TCC
#define ULMBLAS_LEVEL1_ROT_TCC 1

#include <cmath>
#include <complex>
#include <ulmblas/level1/rot.h>

namespace ulmBLAS {

template <typename IndexType, typename VX, typename VY, typename T>
void
rot(IndexType   n,
    VX          *x,
    IndexType   incX,
    VY          *y,
    IndexType   incY,
    T           c,
    T           s)
{
    for (IndexType i=0; i<n; ++i) {
        VX x_ =  c*x[i*incX] + s*y[i*incY];
        VY y_ = -s*x[i*incX] + c*y[i*incY];
        x[i*incX] = x_;
        y[i*incY] = y_;
    }
}

template <typename IndexType, typename X, typename Y, typename T>
void
rot(IndexType              n,
    std::complex<X>        *x,
    IndexType              incX,
    std::complex<Y>        *y,
    IndexType              incY,
    T                      c,
    const std::complex<T>  &s)
{
    typedef std::complex<T>   CT;

    if (incX != IndexType(1) || incY != IndexType(1)) {
        for (IndexType i=0; i<n; ++i) {
            const CT tmp = c*x[i*incX] + s*y[i*incY];
            y[i*incY]  = c*y[i*incY] - std::conj(s)*x[i*incX];
            x[i*incX]  = tmp;
        }
    } else {
        for (IndexType i=0; i<n; ++i) {
            const CT tmp = c*x[i] + s*y[i];
            y[i] = c*y[i] - std::conj(s)*x[i];
            x[i] = tmp;
        }
    }
}

template <typename A, typename B, typename T>
void
rotg(A &a,
     B &b,
     T &c,
     T &s)
{
    A absA = std::abs(a);
    B absB = std::abs(b);

    T scale = absA + absB;
    T roe = (absA > absB) ? a : b;
    if (scale==0) {
        c = 1;
        s = 0;
        a = 0;
        b = 0;
        return;
    }
    A aScaled = absA / scale;
    B bScaled = absB / scale;
    T r = scale*std::sqrt(aScaled*aScaled + bScaled*bScaled);
    if (roe<0) {
        r = -r;
    }
    c = a / r;
    s = b / r;

    B z = 1;
    if (absA > absB) {
        z = s;
    }
    if ((absA <= absB) && (c != 0)) {
        z = T(1)/c;
    }
    a = r;
    b = z;
}

//
//  Complex Givens rotation as in the Reference BLAS (CROTG, ZROTG since
//  LAPACK 3.10): c real, r = c*a + s*b and 0 = -conj(s)*a + c*b.  For b = 0:
//  c = 1, s = 0, r = a; for a = 0: c = 0, s = conj(b)/|b|, r = |b|;
//  otherwise r has the phase of a and |r| = sqrt(|a|^2 + |b|^2).  std::abs
//  and std::hypot avoid overflow and underflow in the intermediate results.
//
template <typename TA, typename TB, typename T>
void
rotg(std::complex<TA>   &a,
     std::complex<TB>   &b,
     T                  &c,
     std::complex<T>    &s)
{
    typedef std::complex<T>  CT;

    const CT  f = a, g = b;

    if (g==CT(0)) {
        c = 1;
        s = 0;
        a = f;
    } else if (f==CT(0)) {
        const T absG = std::abs(g);
        c = 0;
        s = std::conj(g)/absG;
        a = absG;
    } else {
        const T   absF  = std::abs(f);
        const T   norm  = std::hypot(absF, std::abs(g));
        const CT  phase = f/absF;
        c = absF/norm;
        s = phase*std::conj(g)/norm;
        a = phase*norm;
    }
}

//
//  Modified Givens transformation: (x_i, y_i) = H*(x_i, y_i) with H given by
//  param[0] (flag) and param[1..4] = (h11, h21, h12, h22) as in the
//  Reference BLAS (SROTM, DROTM):
//
//      flag = -2:  H = I (x and y are not referenced)
//      flag = -1:  H = [h11 h12; h21 h22]
//      flag =  0:  H = [1 h12; h21 1]
//      flag =  1:  H = [h11 1; -1 h22]
//
//  Other values of flag are classified by their sign as in the Reference
//  BLAS.
//
template <typename IndexType, typename VX, typename VY, typename T>
void
rotm(IndexType   n,
     VX          *x,
     IndexType   incX,
     VY          *y,
     IndexType   incY,
     const T     *param)
{
    const T flag = param[0];

    if (n<=0 || flag==T(-2)) {
        return;
    }

    T h11, h12, h21, h22;
    if (flag<T(0)) {
        h11 = param[1];
        h21 = param[2];
        h12 = param[3];
        h22 = param[4];
        for (IndexType i=0; i<n; ++i) {
            const VX w = x[i*incX];
            const VY z = y[i*incY];
            x[i*incX] = w*h11 + z*h12;
            y[i*incY] = w*h21 + z*h22;
        }
    } else if (flag==T(0)) {
        h21 = param[2];
        h12 = param[3];
        for (IndexType i=0; i<n; ++i) {
            const VX w = x[i*incX];
            const VY z = y[i*incY];
            x[i*incX] = w + z*h12;
            y[i*incY] = w*h21 + z;
        }
    } else {
        h11 = param[1];
        h22 = param[4];
        for (IndexType i=0; i<n; ++i) {
            const VX w = x[i*incX];
            const VY z = y[i*incY];
            x[i*incX] = w*h11 + z;
            y[i*incY] = -w + h22*z;
        }
    }
}

//
//  The scaling constants of ROTMG (gamma = 4096 and the bounds gamma^2 and
//  1/gamma^2 for d1 and |d2|), with the values of the Reference BLAS, where
//  the bounds of SROTMG and DROTMG are rounded decimal numbers.
//
template <typename T>
struct RotmgScaling
{
    static T gam()     { return T(4096); }
    static T gamSq()   { return T(4096)*T(4096); }
    static T rgamSq()  { return T(1)/(T(4096)*T(4096)); }
};

template <>
struct RotmgScaling<float>
{
    static float gam()     { return 4096.0f; }
    static float gamSq()   { return 1.67772e7f; }
    static float rgamSq()  { return 5.96046e-8f; }
};

template <>
struct RotmgScaling<double>
{
    static double gam()    { return 4096.0; }
    static double gamSq()  { return 16777216.0; }
    static double rgamSq() { return 5.9604645e-8; }
};

//
//  Construct the modified Givens transformation H (see rotm) that zeros the
//  second component of (sqrt(d1)*x1, sqrt(d2)*y1)^T: on return
//  H*(x1, y1)^T = (x1', 0)^T and d1, d2, x1 are overwritten by d1', d2', x1'
//  with d1'*x1'^2 = d1*x1^2 + d2*y1^2.  The algorithm of the Reference BLAS
//  (SROTMG, DROTMG): d1 and |d2| are kept in [1/gamma^2, gamma^2] by
//  rescaling, which turns flag 0 and 1 into the full form flag = -1.  As in
//  the current Reference BLAS, H is zeroed (flag = -1) if d1 < 0 or the
//  transformation does not exist (u <= 0 or q2 < 0), and an H that is
//  already in full form is not reset when the second scale factor is
//  adjusted.  Only the elements of param that the flag uses are set.
//
template <typename T>
void
rotmg(T        &d1,
      T        &d2,
      T        &x1,
      const T  &y1,
      T        *param)
{
    const T Zero(0), One(1);
    const T gam    = RotmgScaling<T>::gam();
    const T gamSq  = RotmgScaling<T>::gamSq();
    const T rgamSq = RotmgScaling<T>::rgamSq();

    T flag = -One, h11 = Zero, h12 = Zero, h21 = Zero, h22 = Zero;
    bool zero = false;

    if (d1<Zero) {
        zero = true;
    } else {
        const T p2 = d2*y1;
        if (p2==Zero) {
            param[0] = T(-2);
            return;
        }
        const T p1 = d1*x1;
        const T q2 = p2*y1;
        const T q1 = p1*x1;

        if (std::abs(q1)>std::abs(q2)) {
            h21 = -y1/x1;
            h12 = p2/p1;
            const T u = One - h12*h21;
            if (u>Zero) {
                flag = Zero;
                d1 /= u;
                d2 /= u;
                x1 *= u;
            } else {
                zero = true;
            }
        } else if (q2<Zero) {
            zero = true;
        } else {
            flag = One;
            h11 = p1/p2;
            h22 = x1/y1;
            const T u   = One + h11*h22;
            const T tmp = d2/u;
            d2 = d1/u;
            d1 = tmp;
            x1 = y1*u;
        }
    }
    if (zero) {
        flag = -One;
        h11 = h12 = h21 = h22 = Zero;
        d1 = d2 = x1 = Zero;
    } else {
        if (d1!=Zero) {
            while (d1<=rgamSq || d1>=gamSq) {
                if (flag==Zero) {
                    h11 = h22 = One;
                } else if (flag==One) {
                    h21 = -One;
                    h12 = One;
                }
                flag = -One;
                if (d1<=rgamSq) {
                    d1  *= gam*gam;
                    x1  /= gam;
                    h11 /= gam;
                    h12 /= gam;
                } else {
                    d1  /= gam*gam;
                    x1  *= gam;
                    h11 *= gam;
                    h12 *= gam;
                }
            }
        }
        if (d2!=Zero) {
            while (std::abs(d2)<=rgamSq || std::abs(d2)>=gamSq) {
                if (flag==Zero) {
                    h11 = h22 = One;
                } else if (flag==One) {
                    h21 = -One;
                    h12 = One;
                }
                flag = -One;
                if (std::abs(d2)<=rgamSq) {
                    d2  *= gam*gam;
                    h21 /= gam;
                    h22 /= gam;
                } else {
                    d2  /= gam*gam;
                    h21 *= gam;
                    h22 *= gam;
                }
            }
        }
    }
    if (flag<Zero) {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    } else if (flag==Zero) {
        param[2] = h21;
        param[3] = h12;
    } else {
        param[1] = h11;
        param[4] = h22;
    }
    param[0] = flag;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_ROT_TCC 1
