"""Hollow lamp shade + star holes as STL, built in Python (trimesh + manifold3d)."""

from __future__ import annotations

import io
from typing import Literal, Optional

import numpy as np
import pandas as pd
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix

from horizon import horizontal_unit_vector

ENGINE = "manifold"


def _mag_clamped(mag: float) -> float:
    if mag > 6.5:
        return 6.5
    if mag < -1.5:
        return -1.5
    return mag


def _hole_radius(mag: float, base_rr: float, min_rr: float) -> float:
    delta = base_rr - min_rr
    return base_rr - (_mag_clamped(mag) / 6.5) * delta


def _rot_align_z_to(d: np.ndarray) -> np.ndarray:
    """Rigid rotation mapping +Z to unit direction d (column vectors)."""
    d = np.asarray(d, dtype=np.float64).reshape(3)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.eye(4)
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(d, z))
    if c >= 1.0 - 1e-12:
        return np.eye(4)
    if c <= -1.0 + 1e-12:
        return rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    v = np.cross(z, d)
    s = np.linalg.norm(v)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    r3 = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s**2))
    m = np.eye(4)
    m[:3, :3] = r3
    return m


def _rod_world_matrix_dir(d: np.ndarray, ray_start: float, height: float) -> np.ndarray:
    r = _rot_align_z_to(d)
    tz = translation_matrix([0.0, 0.0, ray_start + height / 2.0])
    return r @ tz


def _star_cylinder_sky(
    alt_deg: float,
    az_deg: float,
    mag: float,
    ray_start: float,
    ray_end: float,
    base_rr: float,
    minimum_rr: float,
    sections: int,
) -> Optional[trimesh.Trimesh]:
    x, y, z = horizontal_unit_vector(alt_deg, az_deg)
    d = np.array([x, y, z], dtype=np.float64)
    h = float(ray_end) - float(ray_start)
    if h <= 1e-4:
        return None
    rad = _hole_radius(mag, base_rr, minimum_rr)
    if rad <= 1e-4:
        return None
    cyl = trimesh.creation.cylinder(radius=float(rad), height=float(h), sections=int(sections))
    cyl.apply_transform(_rod_world_matrix_dir(d, float(ray_start), float(h)))
    return cyl


def _rod_world_matrix(
    ra_hours: float,
    dec_deg: float,
    ray_start: float,
    height: float,
) -> np.ndarray:
    """Place a Z-aligned cylinder so it runs from ray_start to ray_start+height along +Z after celestial rotations."""
    theta = 15.0 * ra_hours
    phi = 90.0 - dec_deg
    rz = rotation_matrix(np.radians(theta), [0, 0, 1])
    rx = rotation_matrix(np.radians(phi), [1, 0, 0])
    tz = translation_matrix([0.0, 0.0, ray_start + height / 2.0])
    return rz @ rx @ tz


def _star_cylinder(
    ra_h: float,
    dec: float,
    mag: float,
    ray_start: float,
    ray_end: float,
    base_rr: float,
    minimum_rr: float,
    sections: int,
) -> Optional[trimesh.Trimesh]:
    h = float(ray_end) - float(ray_start)
    if h <= 1e-4:
        return None
    rad = _hole_radius(mag, base_rr, minimum_rr)
    if rad <= 1e-4:
        return None
    cyl = trimesh.creation.cylinder(radius=float(rad), height=float(h), sections=int(sections))
    cyl.apply_transform(_rod_world_matrix(ra_h, dec, float(ray_start), float(h)))
    return cyl


def _hollow_sphere(R_out: float, R_in: float, ico_subdiv: int) -> trimesh.Trimesh:
    outer = trimesh.creation.icosphere(subdivisions=int(ico_subdiv), radius=float(R_out))
    inner = trimesh.creation.icosphere(subdivisions=int(ico_subdiv), radius=float(R_in))
    return trimesh.boolean.difference([outer, inner], engine=ENGINE)


def _clip_upper_hemisphere(mesh: trimesh.Trimesh, R: float) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=[4.0 * R, 4.0 * R, 2.0 * R])
    box.apply_translation([0.0, 0.0, R])
    return trimesh.boolean.intersection([mesh, box], engine=ENGINE)


def build_lamp_mesh(
    df: pd.DataFrame,
    *,
    shell_radius: float,
    shell_thickness: float,
    base_rod_radius: float,
    minimum_rod_radius: float,
    ico_subdiv: int = 4,
    cylinder_sections: int = 32,
    hole_penetration_mm: float | None = None,
    placement: Literal["equatorial", "local_sky"] = "local_sky",
) -> trimesh.Trimesh:
    """
    CSG: hollow sphere clipped to z ≥ 0, subtract radial cylinders that cut the full wall.

    ``local_sky``: holes aim along altitude/azimuth so the upper dome matches the sky
    above your horizon (zenith = +Z, rim = horizon).

    ``equatorial``: classic RA/Dec / planetarium mapping on a sphere.
    """
    r_out = float(shell_radius)
    r_in = float(r_out - shell_thickness)
    if r_in <= 1e-3:
        raise ValueError("Inner radius must be positive; reduce wall thickness.")

    # Extend past both surfaces so manifold subtract reliably opens the wall
    eps = hole_penetration_mm if hole_penetration_mm is not None else max(0.5, 0.01 * r_out)
    ray_start = r_in - eps
    ray_end = r_out + eps
    if ray_start <= 0:
        ray_start = 1e-3

    shell = _hollow_sphere(r_out, r_in, ico_subdiv)
    shell = _clip_upper_hemisphere(shell, r_out)

    if df is None or df.empty:
        return shell

    cyls: list[trimesh.Trimesh] = []
    for _, row in df.iterrows():
        if placement == "local_sky":
            c = _star_cylinder_sky(
                float(row["alt_deg"]),
                float(row["az_deg"]),
                float(row["mag"]),
                ray_start,
                ray_end,
                base_rod_radius,
                minimum_rod_radius,
                cylinder_sections,
            )
        else:
            c = _star_cylinder(
                float(row["ra"]),
                float(row["dec"]),
                float(row["mag"]),
                ray_start,
                ray_end,
                base_rod_radius,
                minimum_rod_radius,
                cylinder_sections,
            )
        if c is not None and c.faces is not None and len(c.faces) > 0:
            cyls.append(c)

    for c in cyls:
        shell = trimesh.boolean.difference([shell, c], engine=ENGINE)

    return shell


def mesh_to_stl_bytes(mesh: trimesh.Trimesh) -> bytes:
    bio = io.BytesIO()
    mesh.export(bio, file_type="stl")
    return bio.getvalue()


def mesh_quality_presets() -> list[tuple[str, tuple[int, int]]]:
    """Label → (icosphere subdivisions, cylinder edge count)."""
    return [
        ("Balanced (good speed)", (4, 32)),
        ("High (smoother, slower)", (5, 48)),
        ("Draft (quick preview)", (3, 24)),
    ]
