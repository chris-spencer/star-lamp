"""Interactive 3D preview of STL (Plotly) for Streamlit."""

from __future__ import annotations

import io

import plotly.graph_objects as go
import trimesh

DEFAULT_MAX_FACES = 45_000


def figure_from_stl_bytes(
    stl_bytes: bytes,
    *,
    max_faces: int = DEFAULT_MAX_FACES,
    title: str = "Shade preview",
    height_px: int = 520,
) -> go.Figure:
    """Load STL bytes into a rotatable Plotly mesh (optionally decimated for speed)."""
    loaded = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    if len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(int(max_faces))
        except BaseException:
            pass

    v = mesh.vertices
    f = mesh.faces
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color="#b8c5d6",
                flatshading=True,
                lighting=dict(ambient=0.55, diffuse=0.85, specular=0.35),
                lightposition=dict(x=80, y=120, z=200),
            )
        ],
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=dict(
            aspectmode="data",
            xaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                zeroline=False,
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                zeroline=False,
            ),
            bgcolor="rgb(32, 36, 42)",
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.95)),
        ),
        paper_bgcolor="rgb(24, 27, 32)",
        font=dict(color="#e8eaed"),
        margin=dict(l=0, r=0, t=48, b=0),
        height=height_px,
    )
    return fig
