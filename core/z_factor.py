from __future__ import annotations
from numba import njit

@njit(cache=True, fastmath=True)
def Z_PR_vapor(A: float, B: float, tol: float = 1e-10) -> float:
    Z_low = B + 1e-12
    Z_high = 1.0 + B
    if Z_high < 1.0:
        Z_high = 1.0

    Z = Z_high
    for _ in range(25):
        f = (Z**3
             - (1.0 - B)*Z**2
             + (A - 3.0*B*B - 2.0*B)*Z
             - (A*B - B*B - B**3))
        df = (3.0*Z*Z
              - 2.0*(1.0 - B)*Z
              + (A - 3.0*B*B - 2.0*B))

        if abs(df) < 1e-14:
            break

        Z_new = Z - f/df
        if Z_low < Z_new < Z_high:
            Z = Z_new

        if abs(f) < tol:
            return Z

    # bisection fallback
    for _ in range(80):
        Z_mid = 0.5*(Z_low + Z_high)
        f_mid = (Z_mid**3
                 - (1.0 - B)*Z_mid**2
                 + (A - 3.0*B*B - 2.0*B)*Z_mid
                 - (A*B - B*B - B**3))
        if f_mid > 0.0:
            Z_high = Z_mid
        else:
            Z_low = Z_mid

        if abs(Z_high - Z_low) < tol:
            return Z_mid

    return 0.5*(Z_low + Z_high)
