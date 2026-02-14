from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from typing import Tuple

from core.protocols import EOS, GammaModel, PsatModel, TsatModel, MolarVolumeModel
from core.vle.types import BubblePoint
from data.parameters import R


class VLESolver:
    def __init__(
        self,
        eos: EOS,
        gamma_model: GammaModel,
        psat_model: PsatModel,
        tsat_model: TsatModel,
        molar_volume_model: MolarVolumeModel,
    ):
        self.eos = eos
        self.gamma_model = gamma_model
        self.psat_model = psat_model
        self.tsat_model = tsat_model
        self.molar_volume_model = molar_volume_model
        self.nc = int(eos.nc() if callable(getattr(eos, 'nc', None)) else eos.nc)


    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asanyarray(x, dtype=float)
        s = float(np.sum(x))
        if s <= 0.0:
            raise ValueError("Mole fractions must sum to a positive value.")
        x /= s
        if np.any(x < -1e-12):
            raise ValueError("Mole fractions must be non-negative.")
        return x

    def temperature_bracket(self, P: float) -> Tuple[float, float]:
        Tsat = self.tsat_model(P)
        return float(0.95*np.min(Tsat)), float(1.05*np.max(Tsat))
    
    def _y_tilde(self, T:float, x:np.ndarray, P:float):
        x = self._normalize(x)

        vL = self.molar_volume_model(T)
        gamma = self.gamma_model(x, T)
        Psat = self.psat_model(T)
        phi_sat = self.eos.phi_pure(Psat, T)

        # y_tilde_i = x_i * γ_i * Psat_i * φ_i^sat * exp(vL_i*(P - Psat_i)/(R*T))
        y_tilde = x * gamma * Psat * phi_sat * np.exp(vL * (P - Psat) / (R * T))
        return y_tilde, gamma, Psat, phi_sat

    def bubble_residual(self, T: float, x: np.ndarray, P: float) -> float:
        y_tilde, gamma, Psat, phi_sat = self._y_tilde(T, x, P)
        y = y_tilde / y_tilde.sum()
        phi_v = self.eos.phi_mix(y, P, T)

        # Σ (x_i γ_i Psat_i φ_i^sat exp(...)) / (P φ_i^v) = 1
        return np.sum(y_tilde / phi_v) / P - 1.0

    def bubble_temperature(self, x: np.ndarray, P: float) -> float:
        return self.bubble_point(x, P).T
    
    def bubble_point(self, x:np.ndarray, P:float) -> BubblePoint:
        x = self._normalize(x)

        # Pure-component
        if np.count_nonzero(x) == 1:
            i = int(np.argmax(x))
            T = float(self.tsat_model(P)[i])

            y = x.copy()
            gamma = self.gamma_model(x, T)
            Psat = self.psat_model(T)
            phi_sat = self.eos.phi_pure(Psat, T)
            phi_v = self.eos.phi_mix(y, P, T)
            K = np.divide(y, x, out=np.ones_like(x), where=x > 0)

            return BubblePoint(
                T=T, x=x, y=y, P=float(P),
                gamma=gamma, Psat=Psat, phis_sat=phi_sat, phi_v=phi_v, K=K
            )
        
        T_low, T_high = self.temperature_bracket(P)

        f_low = self.bubble_residual(T_low, x, P)
        f_high = self.bubble_residual(T_high, x, P)
        if f_low * f_high > 0:
            raise RuntimeError(
                f"No bracket: f({T_low:.2f})={f_low:.3e}, f({T_high})={f_high:.3e}, for x={x}"
            )
        
        T = float(brentq(self.bubble_residual, T_low, T_high, args=(x, P), xtol=1e-7))
        y_tilde, gamma, Psat, phi_sat, = self._y_tilde(T, x, P)
        y = y_tilde / y_tilde.sum()
        phi_v = self.eos.phi_mix(y, P, T)
        K = y / x

        return BubblePoint(
            T=T, x=x, y=y, P=float(P),
            gamma=gamma, Psat=Psat, phis_sat=phi_sat, phi_v=phi_v, K=K
        )