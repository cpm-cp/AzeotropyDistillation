import numpy as np
from data.parameters import R, COMPONENTS
from core.saturation import molar_volume
from data.parameters import WILSON_LAMBDA

def wilson_activity_coeff(x: np.ndarray, temperature: float) -> np.ndarray:
    """
    Wilson activity coefficients (registry-aware).
    x must be ordered according to COMPONENTS.
    """

    x = np.asarray(x, dtype=float)

    if not np.isclose(x.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"Las fracciones molares deben sumar 1. Suma: {x.sum():.6f}"
        )

    nc = COMPONENTS.nc
    V = molar_volume(temperature)

    # --- Construir matriz Λ_ij desde registry
    Lambda = np.zeros((nc, nc))

    for i, ci in enumerate(COMPONENTS.components):
        for j, cj in enumerate(COMPONENTS.components):
            if i == j:
                Lambda[i, j] = 1.0
            else:
                ld_ij = WILSON_LAMBDA[(ci, cj)]
                Lambda[i, j] = (
                    (V[j] / V[i]) *
                    np.exp(-ld_ij / (R * temperature))
                )

    # --- Wilson equations
    S = Lambda @ x
    ln_gamma = 1.0 - np.log(S) - (Lambda.T @ (x / S))

    return np.exp(ln_gamma)
