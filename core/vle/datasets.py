from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from core.equilibrium import VLESolver
from core.vle.types import BubblePoint


SQRT3_OVER_2 = np.sqrt(3.0) / 2.0

@dataclass(frozen=True)
class TernarySurfaceData:
    """Dataset for ternary bubble-temperrature map on the simplex."""
    P: float
    components: tuple[str, ...]
    x: np.ndarray
    xy: np.ndarray
    T: np.ndarray

def simplex_grid(N: int) -> np.ndarray:
    """Triangular grid on x1-x2-x3 simplex with resolution N (N>=2)."""
    if N < 2:
        raise ValueError("N must be >= 2")
    pts: list[tuple[float, float, float]] = []
    for i in range(N + 1):
        x1 = i / N
        for j in range(N + 1 - i):
            x2 = j / N
            x3 = 1.0 - x1 - x2
            pts.append((x1, x2, x3))
    return np.asarray(pts, dtype=float)

def barycentric_to_xy(x: np.ndarray) -> np.ndarray:
    """
    Map barycentric composition x=[x1,x2,x3] to 2D coordinates of an equilateral triangle:
    V1(x1=1)->(0,0), V2(x2=1)->(1,0), V3(x3=1)->(0.5, sqrt(3)/2)
    """
    x = np.asarray(x, dtype=float)
    X = x[:, 1] + 0.5 * x[:, 2]
    Y = SQRT3_OVER_2 * x[:, 2]
    return np.column_stack([X, Y])

def ternary_bubble_temperature_surface(
    solver: VLESolver,
    P: float,
    N: int,
    components: tuple[str, str, str],
) -> TernarySurfaceData:
    """Compute bubble-temperature T(x,P) for a simplex grid."""
    x = simplex_grid(N)
    xy = barycentric_to_xy(x)

    T = np.empty(x.shape[0], dtype=float)
    for k in range(x.shape[0]):
        bp: BubblePoint = solver.bubble_point(x[k], P)
        T[k] = bp.T

    return TernarySurfaceData(P=float(P), components=components, x=x, xy=xy, T=T)
