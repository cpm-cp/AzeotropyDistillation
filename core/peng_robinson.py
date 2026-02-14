from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace
from typing import Optional

from data.parameters import R, build_critical_parameters
from core.z_factor import Z_PR_vapor


@dataclass(frozen=True)
class PengRobinsonEOS:
    Tc: np.ndarray
    Pc: np.ndarray
    w: np.ndarray
    k_ij: np.ndarray

    def __post_init__(self):
        nc = self.Tc.size
        if self.k_ij.shape != (nc, nc):
            raise ValueError(f"k_ij must be ({nc},{nc})")

    @classmethod
    def default(cls) -> "PengRobinsonEOS":
        crit = build_critical_parameters()
        nc = crit.Tc.size
        return cls(Tc=crit.Tc, Pc=crit.Pc, w=crit.w, k_ij=np.zeros((nc, nc), dtype=float))

    @property
    def nc(self) -> int:
        return int(self.Tc.size)

    def update_kij(self, k_ij: np.ndarray) -> "PengRobinsonEOS":
        k_ij = np.asarray(k_ij, dtype=float)
        return replace(self, k_ij=k_ij)

    def _alpha(self, T: float) -> np.ndarray:
        Tr = T / self.Tc
        kappa = 0.37464 + self.w * (1.5422 - 0.26992*self.w)
        return (1.0 + kappa*(1.0 - np.sqrt(Tr)))**2

    def _a(self, T: float) -> np.ndarray:
        a0 = 0.45724 * (R**2) * (self.Tc**2) / self.Pc
        return a0 * self._alpha(T)

    def _b(self) -> np.ndarray:
        return 0.07780 * R * self.Tc / self.Pc

    def phi_pure(self, P: np.ndarray, T: float) -> np.ndarray:
        """Pure-component fugacity coeffs at each component Psat (vector P, kPa)."""
        P = np.asarray(P, dtype=float)
        a = self._a(T)
        b = self._b()
        sqrt2 = np.sqrt(2.0)

        A = a * P / (R**2 * T**2)
        B = b * P / (R * T)

        phi = np.zeros(self.nc, dtype=float)
        for i in range(self.nc):
            Zi = Z_PR_vapor(A[i], B[i])
            lnphi = ((Zi - 1.0)
                     - np.log(Zi - B[i])
                     - (A[i]/(2.0*sqrt2*B[i])) * np.log((Zi + (1.0+sqrt2)*B[i])/(Zi + (1.0-sqrt2)*B[i])))
            phi[i] = np.exp(lnphi)
        return phi

    def phi_mix(self, y: np.ndarray, P: float, T: float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        y = y / y.sum()

        a = self._a(T)
        b = self._b()
        sqrt2 = np.sqrt(2.0)

        a_ij = np.sqrt(a[:, None]*a[None, :]) * (1.0 - self.k_ij)
        a_mix = np.sum(y[:, None]*y[None, :]*a_ij)
        b_mix = float(np.dot(y, b))

        A = a_mix * P / (R**2 * T**2)
        B = b_mix * P / (R * T)

        Z = Z_PR_vapor(A, B)
        phi = np.zeros(self.nc, dtype=float)

        for i in range(self.nc):
            sum_aij = float(np.dot(y, a_ij[i]))
            lnphi = ((b[i]/b_mix)*(Z - 1.0)
                     - np.log(Z - B)
                     - (A/(2.0*sqrt2*B))*((2.0*sum_aij/a_mix) - (b[i]/b_mix))
                     * np.log((Z + (1.0+sqrt2)*B)/(Z + (1.0-sqrt2)*B)))
            phi[i] = np.exp(lnphi)

        return phi