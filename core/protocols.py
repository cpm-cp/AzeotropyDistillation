from __future__ import annotations
from typing import Protocol
import numpy as np

class EOS(Protocol):
    nc: int
    def phi_pure(self, P: np.ndarray, T: float) -> np.ndarray: ...
    def phi_mix(self, y: np.ndarray, P: float, T: float) -> np.ndarray: ...

class GammaModel(Protocol):
    def __call__(self, x: np.ndarray, T: float) -> np.ndarray: ...

class PsatModel(Protocol):
    def __call__(self, T: float) -> np.ndarray: ...

class TsatModel(Protocol):
    def __call__(self, P: float) -> np.ndarray: ...

class MolarVolumeModel(Protocol):
    def __call__(self, T: float) -> np.ndarray: ...
