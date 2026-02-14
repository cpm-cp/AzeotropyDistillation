import numpy as np
from scipy.optimize import brentq
from functools import lru_cache
from data.parameters import COMPONENTS, WAGNER, R, RACKETT_Zra


def molar_volume(temperature: float) -> np.ndarray:
    """
    Rackett equation for saturated liquid molar volume [m³/mol].
    Returned array follows COMPONENTS order.
    """
    nc = COMPONENTS.nc
    vL = np.zeros(nc)

    for i, c in enumerate(COMPONENTS.components):
        data = WAGNER[c]
        Tc = data["Tc"]
        Pc = data["Pc"]
        Zra = RACKETT_Zra[c]

        Tr = temperature / Tc
        vL[i] = (
            1e-3 * R * Tc / Pc
            * Zra ** (1 + (1 - Tr) ** (2 / 7))
        )

    return vL


def saturation_pressure_wagner(temperature: float) -> np.ndarray:
    """
    Wagner vapor pressure correlation.
    Returns Psat [kPa] ordered by COMPONENTS.
    """
    nc = COMPONENTS.nc
    Psat = np.zeros(nc)

    for i, c in enumerate(COMPONENTS.components):
        d = WAGNER[c]
        Tc, Pc = d["Tc"], d["Pc"]

        tau = 1.0 - temperature / Tc
        exponent = (
            d["A"] * tau
            + d["B"] * tau ** 1.5
            + d["C"] * tau ** 3
            + d["D"] * tau ** 6
        ) / (1.0 - tau)

        Psat[i] = Pc * np.exp(exponent)

    return Psat


@lru_cache(maxsize=32)
def saturation_temperature_wagner(pressure: float) -> np.ndarray:
    """
    Inverse Wagner equation: Tsat(P).
    Returns Tsat [K] ordered by COMPONENTS.
    """
    nc = COMPONENTS.nc
    Tsat = np.zeros(nc)

    for i, c in enumerate(COMPONENTS.components):
        d = WAGNER[c]
        Tc, Pc = d["Tc"], d["Pc"]
        A, B, C, D = d["A"], d["B"], d["C"], d["D"]

        def residual(T: float) -> float:
            tau = 1.0 - T / Tc
            exponent = (
                A * tau
                + B * tau ** 1.5
                + C * tau ** 3
                + D * tau ** 6
            ) / (1.0 - tau)
            return Pc * np.exp(exponent) - pressure

        Tsat[i] = brentq(
            residual,
            0.3 * Tc,
            Tc * (1 - 1e-6),
            xtol=1e-6
        )

    return Tsat
