"""Integration tests for mesh_export STL builder (trimesh + manifold)."""

from __future__ import annotations

import unittest

import pandas as pd

import mesh_export


class TestSTLBuilder(unittest.TestCase):
    def test_empty_selection_builds_shell(self) -> None:
        mesh = mesh_export.build_lamp_mesh(
            pd.DataFrame(),
            shell_radius=50.0,
            shell_thickness=3.0,
            base_rod_radius=1.5,
            minimum_rod_radius=0.35,
            ico_subdiv=3,
            cylinder_sections=16,
            placement="local_sky",
        )
        self.assertIsNotNone(mesh.faces)
        self.assertGreater(len(mesh.faces), 10)
        stl = mesh_export.mesh_to_stl_bytes(mesh)
        self.assertGreater(len(stl), 500)

    def test_sky_holes_build_and_export(self) -> None:
        df = pd.DataFrame(
            {
                "ra": [5.9195, 6.0],
                "dec": [7.407, 10.0],
                "mag": [3.5, 4.2],
                "alt_deg": [45.0, 50.0],
                "az_deg": [120.0, 200.0],
            }
        )
        mesh = mesh_export.build_lamp_mesh(
            df,
            shell_radius=45.0,
            shell_thickness=2.5,
            base_rod_radius=1.2,
            minimum_rod_radius=0.4,
            ico_subdiv=3,
            cylinder_sections=16,
            placement="local_sky",
        )
        self.assertGreater(len(mesh.faces), 10)
        stl = mesh_export.mesh_to_stl_bytes(mesh)
        self.assertGreater(len(stl), 500)


if __name__ == "__main__":
    unittest.main()
