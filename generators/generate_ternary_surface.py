from core.peng_robinson import PengRobinsonEOS
from core.wilson import wilson_activity_coeff
from core.saturation import (
    saturation_pressure_wagner,
    saturation_temperature_wagner,
    molar_volume
)
from data.parameters import build_critical_parameters, COMPONENTS
from core.equilibrium import VLESolver
from core.vle.datasets import ternary_bubble_temperature_surface

def main():
    # Pressure in kPa (consistent with Psat/Pc data)
    P = 101.325  # kPa ~ 1 atm
    N = 40       # grid resolution

    crit = build_critical_parameters()
    import numpy as np

    eos = PengRobinsonEOS(crit.Tc, crit.Pc, crit.w, k_ij=np.zeros((crit.Tc.size, crit.Tc.size)))

    solver = VLESolver(
        eos=eos,
        gamma_model=wilson_activity_coeff,
        psat_model=saturation_pressure_wagner,
        tsat_model=saturation_temperature_wagner,
        molar_volume_model=molar_volume,
    )

    data = ternary_bubble_temperature_surface(
        solver=solver,
        P=P,
        N=N,
        components=tuple(COMPONENTS.components),
    )

    print(f"Computed {data.T.size} points")
    print(f"T range: {data.T.min():.2f} - {data.T.max():.2f} K")


if __name__ == "__main__":
    main()