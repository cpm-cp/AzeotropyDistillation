from __future__ import annotations

import numpy as np
import flet as ft
from flet_webview import WebView
from pathlib import Path
import tempfile

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


def _write_plotly_html(fig, filename: str = "ternary_plot.html") -> str:
    """
    Write Plotly fig to a temp html file and return file:// URL for WebView.
    """
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    tmp_dir = Path(tempfile.gettempdir()) / "distillation_app"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / filename
    out.write_text(html, encoding="utf-8")
    return out.as_uri()  # file:///...


def main(page: ft.Page):
    page.title = "Distillation | Acetone–Methanol–Chloroform"
    page.window_width = 1200
    page.window_height = 820
    page.scroll = ft.ScrollMode.AUTO

    solver = build_solver()

    P_field = ft.TextField(label="Pressure [kPa]", value="101.325", width=180)
    N_slider = ft.Slider(min=15, max=90, divisions=75, value=40, label="{value}", width=320)
    stride_slider = ft.Slider(min=1, max=6, divisions=5, value=2, label="{value}", width=220)

    wire_cb = ft.Checkbox(label="Wireframe", value=True)
    plane_cb = ft.Checkbox(label="Tmin plane", value=True)

    status = ft.Text("")
    progress = ft.ProgressRing(visible=False)

    # WebView placeholder
    web = WebView(url="about:blank", expand=True)

    def run_compute(e):
        try:
            progress.visible = True
            status.value = "Computing..."
            page.update()

            P = float(P_field.value)
            N = int(N_slider.value)
            wire_stride = int(stride_slider.value)
            wireframe = bool(wire_cb.value)
            min_plane = bool(plane_cb.value)

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
                wireframe=wireframe,
                wire_stride=wire_stride,
                min_plane=min_plane,
                title="Bubble point temperature surface",
            )

            url = _write_plotly_html(fig, "ternary_surface.html")
            web.url = url

            status.value = (
                f"Computed {data.T.size} points | "
                f"T range: {data.T.min():.2f}–{data.T.max():.2f} K"
            )
        except Exception as ex:
            status.value = f"Error: {ex}"
        finally:
            progress.visible = False
            page.update()

    run_btn = ft.ElevatedButton("Run", on_click=run_compute)

    controls = ft.Container(
        content=ft.Row(
            [
                P_field,
                ft.Column([ft.Text("Grid resolution N"), N_slider], spacing=4),
                ft.Column([ft.Text("Wire stride"), stride_slider], spacing=4),
                ft.Column([wire_cb, plane_cb], spacing=6),
                run_btn,
                progress,
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=10,
    )

    page.add(
        ft.Column(
            [
                controls,
                status,
                ft.Divider(),
                ft.Container(web, expand=True, padding=10),
            ],
            expand=True,
        )
    )