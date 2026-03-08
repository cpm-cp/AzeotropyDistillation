from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Delaunay


SQRT3_OVER_2 = np.sqrt(3.0) / 2.0


def _triangle_vertices_xy() -> np.ndarray:
    # V1(x1=1)->(0,0), V2(x2=1)->(1,0), V3(x3=1)->(0.5, sqrt(3)/2)
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, SQRT3_OVER_2],
        ],
        dtype=float,
    )


def _triangle_mesh_plane(
    z_value: float,
    mode: str = "3d",
) -> go.Mesh3d:
    """
    Create a flat triangular plane at z=z_value covering the simplex triangle.
    """
    V = _triangle_vertices_xy()
    x = V[:, 0]
    y = V[:, 1]
    z = np.full(3, z_value, dtype=float)
    # single triangle: (0,1,2)
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=[0],
        j=[1],
        k=[2],
        opacity=0.25,
        flatshading=True,
        name="Tmin plane",
        showlegend=False,
        hoverinfo="skip",
    )


def _wireframe_traces(
    xy: np.ndarray,
    z: np.ndarray,
    simplices: np.ndarray,
    stride: int = 1,
) -> go.Scatter3d:
    """
    Draw wireframe lines for a triangulated mesh by plotting triangle edges.
    stride>1 reduces density (useful for performance).
    """
    # edges as pairs; we will concatenate segments with None separators
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    # optionally thin triangles
    tris = simplices[:: max(1, int(stride))]

    for a, b, c in tris:
        edges = [(a, b), (b, c), (c, a)]
        for u, v in edges:
            xs += [float(xy[u, 0]), float(xy[v, 0]), None]
            ys += [float(xy[u, 1]), float(xy[v, 1]), None]
            zs += [float(z[u]), float(z[v]), None]

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(width=2),
        name="Wireframe",
        showlegend=False,
        hoverinfo="skip",
        opacity=0.65,
    )


def make_ternary_temperature_figure(
    xy: np.ndarray,
    T: np.ndarray,
    components: Sequence[str],
    Tsat_pure: Optional[Sequence[float]] = None,
    P_kPa: Optional[float] = None,
    title: Optional[str] = None,
    mode: str = "3d",
    wireframe: bool = True,
    wire_stride: int = 2,
    min_plane: bool = True,
    annotate_pure: bool = True,
    show_minmax: bool = True,
    camera: Optional[str] = None,  # "view1" | "view2" | None
) -> go.Figure:
    """
    Ternary bubble temperature plot.

    Parameters
    ----------
    mode:
        "2d": flat mesh colored by T (z=0)
        "3d": temperature surface (z=T)
    wireframe:
        Draw triangle-edge wireframe (paper-like mesh).
    min_plane:
        Draw a triangular plane at Tmin (paper-like minimum temperature plane).
    camera:
        "view1" or "view2" presets (paper-like perspectives).
    """
    xy = np.asarray(xy, dtype=float)
    T = np.asarray(T, dtype=float)

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (n,2)")
    if T.ndim != 1 or T.shape[0] != xy.shape[0]:
        raise ValueError("T must be (n,) and match xy length")
    if len(components) != 3:
        raise ValueError("components must be length 3")

    tri = Delaunay(xy)
    simplices = tri.simplices
    i, j, k = simplices[:, 0], simplices[:, 1], simplices[:, 2]

    Tmin = float(np.nanmin(T))
    Tmax = float(np.nanmax(T))

    is3d = mode.lower() == "3d"
    z_nodes = T if is3d else np.zeros_like(T)

    fig = go.Figure()

    # Surface mesh
    fig.add_trace(
        go.Mesh3d(
            x=xy[:, 0],
            y=xy[:, 1],
            z=z_nodes,
            i=i,
            j=j,
            k=k,
            intensity=T,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="T [K]"),
            flatshading=True,
            opacity=1.0,
            name="Tbubble",
            hovertemplate="X=%{x:.3f}<br>Y=%{y:.3f}<br>T=%{intensity:.2f} K<extra></extra>",
        )
    )

    # Wireframe (mesh grid look)
    if wireframe:
        # For 2D mode, keep it at z=0; for 3D, use z=T
        fig.add_trace(
            _wireframe_traces(
                xy=xy,
                z=z_nodes if is3d else np.zeros_like(T),
                simplices=simplices,
                stride=max(1, int(wire_stride)),
            )
        )

    # Triangle boundary lines (binary edges)
    V = _triangle_vertices_xy()
    boundary = np.vstack([V, V[0]])  # close loop
    boundary_z = np.full(boundary.shape[0], Tmin if is3d else 0.0, dtype=float)
    fig.add_trace(
        go.Scatter3d(
            x=boundary[:, 0],
            y=boundary[:, 1],
            z=boundary_z,
            mode="lines",
            line=dict(width=6),
            name="Edges",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Tmin plane (triangle)
    if min_plane and is3d:
        fig.add_trace(_triangle_mesh_plane(z_value=Tmin))

    # Pure component vertices + annotations
    if annotate_pure:
        pure_xy = V
        pure_z = (
            np.array(Tsat_pure, dtype=float)
            if (is3d and Tsat_pure is not None and len(Tsat_pure) >= 3)
            else np.full(3, Tmin if is3d else 0.0, dtype=float)
        )

        texts = []
        for idx, name in enumerate(components):
            if Tsat_pure is not None and len(Tsat_pure) >= 3:
                texts.append(f"{name}<br>Tsat={float(Tsat_pure[idx]):.2f} K")
            else:
                texts.append(f"{name}<br>pure")

        fig.add_trace(
            go.Scatter3d(
                x=pure_xy[:, 0],
                y=pure_xy[:, 1],
                z=pure_z if is3d else np.zeros(3),
                mode="markers+text",
                marker=dict(size=7),
                text=texts,
                textposition="top center",
                showlegend=False,
                name="Pure points",
            )
        )

    # Global min/max markers
    if show_minmax:
        idx_min = int(np.nanargmin(T))
        idx_max = int(np.nanargmax(T))

        def mk_point(idx: int, label: str):
            return go.Scatter3d(
                x=[xy[idx, 0]],
                y=[xy[idx, 1]],
                z=[z_nodes[idx] if is3d else 0.0],
                mode="markers+text",
                marker=dict(size=7, symbol="diamond"),
                text=[f"{label}<br>T={T[idx]:.2f} K"],
                textposition="top left",
                showlegend=False,
            )

        fig.add_trace(mk_point(idx_min, "Min"))
        fig.add_trace(mk_point(idx_max, "Max"))

    # Title
    if title is None:
        title = "Bubble point temperature surface"
        if P_kPa is not None:
            title += f" @ {P_kPa:.3f} kPa"

    # Axes labels (paper-like)
    zaxis = dict(title="T [K]", visible=True) if is3d else dict(visible=False)
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=50, b=10),
        width=950,     # makes it visually wider
        height=720,
        scene=dict(
            xaxis=dict(title="", visible=False),
            yaxis=dict(title="", visible=False),
            zaxis=zaxis,
            aspectmode="manual",
            # Make XY visually comparable to Z range (so it doesn't look like a needle)
            aspectratio=dict(x=1.4, y=1.2, z=0.75),
        ),
        showlegend=False,
    )

    # Camera presets (paper-like “View 1 / View 2”)
    if is3d:
        fig.update_layout(scene_camera=dict(eye=dict(x=1.6, y=1.4, z=0.9)))
    else:
        fig.update_layout(scene_camera=dict(eye=dict(x=0.0, y=0.0, z=2.2)))

    return fig