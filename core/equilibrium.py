from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from typing import Tuple

from core.protocols import EOS, GammaModel, PsatModel, TsatModel, MolarVolumeModel
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
        self.nc = eos.nc

    def temperature_bracket(self, P: float) -> Tuple[float, float]:
        Tsat = self.tsat_model(P)
        return float(0.95*np.min(Tsat)), float(1.05*np.max(Tsat))

    def bubble_residual(self, T: float, x: np.ndarray, P: float) -> float:
        x = np.asarray(x, dtype=float)
        x = x / x.sum()

        vL = self.molar_volume_model(T)
        gamma = self.gamma_model(x, T) if callable(self.gamma_model) else self.gamma_model(x, T, vL)
        Psat = self.psat_model(T)
        phi_sat = self.eos.phi_pure(Psat, T)

        y_tilde = x * gamma * Psat * phi_sat * np.exp(vL * (P - Psat) / (R*T))
        y = y_tilde / y_tilde.sum()

        phi_v = self.eos.phi_mix(y, P, T)

        # ecuación consistente con tu forma final:
        return np.sum(y_tilde / phi_v) / P - 1.0

    def bubble_temperature(self, x: np.ndarray, P: float) -> float:
        x = np.asarray(x, dtype=float)
        x = x / x.sum()

        # caso puro
        if np.count_nonzero(x) == 1:
            i = int(np.argmax(x))
            return float(self.tsat_model(P)[i])

        T_low, T_high = self.temperature_bracket(P)

        f_low = self.bubble_residual(T_low, x, P)
        f_high = self.bubble_residual(T_high, x, P)
        if f_low * f_high > 0:
            raise RuntimeError(
                f"No bracket: f({T_low:.2f})={f_low:.3e}, f({T_high:.2f})={f_high:.3e} for x={x}"
            )

        return float(brentq(self.bubble_residual, T_low, T_high, args=(x, P), xtol=1e-7))
