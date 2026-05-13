"""Regression tests for sky-padding altitude filters (instantaneous horizon)."""

from __future__ import annotations

import datetime as dt
import unittest

import pandas as pd

from horizon import HorizonConfig, alt_az_arrays, peak_altitudes_deg_array


class TestFillerAltitudeGate(unittest.TestCase):
    def test_padding_keeps_only_stars_above_fake_horizon_now(self) -> None:
        """Stars that can peak ≥ min_alt may still be below horizon at this UTC instant."""
        when = dt.datetime(2026, 6, 21, 12, 0, tzinfo=dt.timezone.utc)
        lat, lon = 53.11, -1.222
        min_alt = 20.0

        cand = pd.DataFrame(
            {
                "id": [1, 2],
                "ra": [0.0, 12.0],
                "dec": [45.0, -60.0],
                "mag": [5.5, 5.6],
            }
        )

        peaks = peak_altitudes_deg_array(cand["dec"].to_numpy(dtype=float), lat)
        cand_peak = cand.loc[peaks >= float(min_alt) - 1e-6].copy()

        cfg = HorizonConfig(lat_deg=float(lat), lon_deg_east=float(lon), when_utc=when)
        instant_alts, _ = alt_az_arrays(
            cand_peak["ra"].to_numpy(dtype=float),
            cand_peak["dec"].to_numpy(dtype=float),
            cfg,
        )

        cand_ok = cand_peak.assign(_pad_alt_now=instant_alts)
        visible = cand_ok[cand_ok["_pad_alt_now"] >= float(min_alt) - 1e-9]

        for _, row in visible.iterrows():
            self.assertGreaterEqual(float(row["_pad_alt_now"]), min_alt - 1e-3)


if __name__ == "__main__":
    unittest.main()
