from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class BubblePoint:
    """Bubble-point solution at given (x, P)."""
    T: float
    x: np.ndarray
    y: np.ndarray
    P: float
    gamma: np.ndarray
    Psat: np.ndarray
    phis_sat: np.ndarray
    phi_v: np.ndarray
    K: np.ndarray