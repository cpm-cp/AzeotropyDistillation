import numpy as np
from typing import Final, Dict
from dataclasses import dataclass
from data.component_registry import ComponentRegistry

R : Final[float] = 8.31446261815324 # J/molK

COMPONENTS: Final[ComponentRegistry] = ComponentRegistry(["Acetone", "Chloroform", "Methanol"])

RACKETT_Zra : Final[Dict[str, float]] = {
    "Acetone": 0.2459,
    "Chloroform": 0.2748,
    "Methanol": 0.2318,
}

WAGNER : Final[Dict[str, Dict[str, float]]] = {
    "Acetone": {
        "A": -7.49745, 
        "B": 1.31738, 
        "C": -2.67542, 
        "D": -2.67774, 
        "Tc": 508.2, 
        "Pc": 4701.0
    },

    "Chloroform": {
        "A": -6.50441, 
        "B": 0.01117, 
        "C": -0.37736, 
        "D": -2.22322, 
        "Tc": 536.4, 
        "Pc": 5472.0
    },

    "Methanol": {
        "A": -8.60523453, 
        "B": 0.89447567, 
        "C": -3.24255929, 
        "D": 1.31992805, 
        "Tc": 512.6, 
        "Pc": 8097.0
    }
}

@dataclass(frozen=True)
class CriticalParameters:
    Tc: np.ndarray
    Pc: np.ndarray
    w: np.ndarray

def build_critical_parameters() -> CriticalParameters:
    Tc = np.array([WAGNER[c]["Tc"] for c in COMPONENTS])
    Pc = np.array([WAGNER[c]["Pc"] for c in COMPONENTS])
    w = np.array([0.307, 0.222, 0.564]) # Later always with name-order

    return CriticalParameters(Tc=Tc, Pc=Pc, w=w)

WILSON_LAMBDA = {
    ("Acetone", "Chloroform"): -349.3,
    ("Acetone", "Methanol"):   -810.7,
    ("Chloroform", "Acetone"): -1586.4,
    ("Chloroform", "Methanol"):-1489.3,
    ("Methanol", "Acetone"):   2716.4,
    ("Methanol", "Chloroform"):7528.4,
}
