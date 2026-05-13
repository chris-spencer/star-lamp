"""Interactive 3D preview of STL (Plotly) for Streamlit."""

from __future__ import annotations

import io

import numpy as np
import plotly.graph_objects as go
import trimesh

DEFAULT_PREVIEW_FACE_CAP = 220_000  # only simplify *browser* mesh when STL exceeds this (Plotly FPS)


def _clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    mask = m.nondegenerate_faces()
    if hasattr(m, "unique_faces"):
        mask = mask & m.unique_faces()
    m.update_faces(mask)
    m.remove_unreferenced_vertices()
    return m


def _split_long_preview_edges(mesh: trimesh.Trimesh, max_faces_budget: int) -> trimesh.Trimesh:
    """
    Hole booleans tend to spawn very long skinny triangles; Plotly's smooth shading exaggerates those
    as dark streaks. Subdivide-by-edge-length only when the mesh is still moderate size.
    """
    if len(mesh.faces) > 100_000:
        return mesh
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    if diag <= 1e-9:
        return mesh
    max_edge = diag * 0.022
    try:
        m = mesh.subdivide_to_size(max_edge=max_edge, max_iter=6)
        m = _clean_mesh(m)
    except BaseException:
        return mesh
    if len(m.faces) > max_faces_budget:
        return mesh
    return m


def _maybe_decimate_for_plotly(mesh: trimesh.Trimesh, face_cap: int) -> trimesh.Trimesh:
    """Avoid changing hole geometry; decimate only for very heavy meshes."""
    if len(mesh.faces) <= face_cap:
        return mesh
    try:
        red = mesh.simplify_quadric_decimation(int(face_cap))
    except BaseException:
        return mesh
    return _clean_mesh(red)


def figure_from_stl_bytes(
    stl_bytes: bytes,
    *,
    max_faces: int = DEFAULT_PREVIEW_FACE_CAP,
    title: str = "Shade preview",
    height_px: int = 520,
) -> go.Figure:
    """Load STL bytes into a rotatable Plotly mesh (minimal decimation; keeps hole shapes truthful)."""
    loaded = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    mesh = _clean_mesh(mesh)
    mesh = _split_long_preview_edges(mesh, max_faces)
    mesh = _maybe_decimate_for_plotly(mesh, max_faces)

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
                color="#0d0d0d",
                # Flat facets avoid smooth-normal artefacts on skinny CSG facets; readable on white bg.
                flatshading=True,
                lighting=dict(ambient=0.42, diffuse=0.5, specular=0.2, roughness=0.35),
                lightposition=dict(x=-80, y=160, z=260),
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
            bgcolor="#ffffff",
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.95)),
        ),
        paper_bgcolor="#ffffff",
        font=dict(color="#1f1f1f"),
        margin=dict(l=0, r=0, t=48, b=0),
        height=height_px,
    )
    return fig
