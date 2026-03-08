from __future__ import annotations

import numpy as np

from core.peng_robinson import PengRobinsonEOS
from core.wilson import wilson_activity_coeff
from core.saturation import (
    saturation_pressure_wagner,
    saturation_temperature_wagner,
    molar_volume,
)
from data.parameters import build_critical_parameters, COMPONENTS
from core.equilibrium import VLESolver
from core.vle.datasets import ternary_bubble_temperature_surface

from plotting.ternary_temperature_map import make_ternary_temperature_figure


def build_solver() -> VLESolver:
    crit = build_critical_parameters()
    eos = PengRobinsonEOS(
        crit.Tc,
        crit.Pc,
        crit.w,
        k_ij=np.zeros((crit.Tc.size, crit.Tc.size)),
    )
    return VLESolver(
        eos=eos,
        gamma_model=wilson_activity_coeff,
        psat_model=saturation_pressure_wagner,
        tsat_model=saturation_temperature_wagner,
        molar_volume_model=molar_volume,
    )


def main():
    P = 101.325  # kPa
    N = 40

    solver = build_solver()

    data = ternary_bubble_temperature_surface(
        solver=solver,
        P=P,
        N=N,
        components=tuple(COMPONENTS.components),
    )

    Tsat = saturation_temperature_wagner(P)

    fig = make_ternary_temperature_figure(
        xy=data.xy,
        T=data.T,
        components=data.components,
        Tsat_pure=Tsat,
        P_kPa=P,
        mode="3d",
        wireframe=True,
        wire_stride=2,
        min_plane=True,
        title="Bubble point temperature surface",
    )

    fig.write_html("ternary_surface.html", auto_open=True)
    print("Saved: ternary_surface.html")
      


if __name__ == "__main__":
    main()