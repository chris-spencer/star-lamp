"""Altitude / azimuth for stars given observer location and UTC time."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def _julian_date(utc: datetime) -> float:
    if utc.tzinfo is None:
        utc = utc.replace(tzinfo=timezone.utc)
    else:
        utc = utc.astimezone(timezone.utc)
    y, m = utc.year, utc.month
    d = (
        utc.day
        + (utc.hour + (utc.minute + utc.second / 60.0) / 60.0) / 24.0
    )
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return jd


def greenwich_mean_sidereal_time_deg(jd: float) -> float:
    """GMST at JD (approximate IAU-like formula, good to ~1s for our use)."""
    t = (jd - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - t * t * t / 38710000.0
    )
    return gmst % 360.0


def local_sidereal_time_deg(jd: float, lon_deg_east: float) -> float:
    return (greenwich_mean_sidereal_time_deg(jd) + lon_deg_east) % 360.0


def _wrap_pm180_vec(x: np.ndarray) -> np.ndarray:
    x = np.mod(x, 360.0)
    return np.where(x > 180.0, x - 360.0, x)


@dataclass(frozen=True)
class HorizonConfig:
    lat_deg: float
    lon_deg_east: float
    when_utc: datetime


def alt_az_arrays(
    ra_hours: np.ndarray, dec_deg: np.ndarray, cfg: HorizonConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized altitude/azimuth; RA in decimal hours matching :func:`altitude_azimuth`."""
    jd = _julian_date(cfg.when_utc)
    lst = local_sidereal_time_deg(jd, cfg.lon_deg_east)
    ra_deg = np.mod(ra_hours, 24.0) * 15.0
    h_deg = _wrap_pm180_vec(lst - ra_deg)
    h = np.radians(h_deg)
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    lat_rad = math.radians(cfg.lat_deg)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    sin_alt = np.sin(dec) * sin_lat + np.cos(dec) * cos_lat * np.cos(h)
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    alt = np.degrees(np.arcsin(sin_alt))

    y = -np.sin(h) * np.cos(dec)
    x_h = cos_lat * np.sin(dec) - sin_lat * np.cos(dec) * np.cos(h)
    az = (np.degrees(np.arctan2(y, x_h)) + 360.0) % 360.0
    return alt, az


def peak_altitude_deg(dec_deg: float, lat_deg: float) -> float:
    """Best possible altitude (upper culmination) for a given declination and latitude."""
    dec = math.radians(dec_deg)
    lat = math.radians(lat_deg)
    a = math.sin(dec) * math.sin(lat)
    b = math.cos(dec) * math.cos(lat)
    s = min(1.0, max(-1.0, a + abs(b)))
    return math.degrees(math.asin(s))


def peak_altitudes_deg_array(dec_deg: np.ndarray | pd.Series, lat_deg: float) -> np.ndarray:
    """Vectorized :func:`peak_altitude_deg` for many declinations (same latitude)."""
    d = np.asarray(dec_deg, dtype=float)
    dec = np.radians(d)
    lat = math.radians(lat_deg)
    a = np.sin(dec) * math.sin(lat)
    b = np.cos(dec) * math.cos(lat)
    s = np.clip(a + np.abs(b), -1.0, 1.0)
    return np.degrees(np.arcsin(s))


def _star_label(row: pd.Series) -> str:
    p = row.get("proper")
    if isinstance(p, str) and p.strip():
        return p.strip()
    con = row.get("con", "")
    hip = row.get("hip")
    if hip is not None and not (isinstance(hip, float) and math.isnan(hip)):
        return f"{con} HIP {int(hip)}"
    return f"id {row.get('id', '?')}"


def infeasible_star_labels(df: pd.DataFrame, lat_deg: float, min_alt_deg: float) -> list[str]:
    """Stars that never reach min_alt_deg from this latitude (no need to search time)."""
    if df.empty:
        return []
    peaks = peak_altitudes_deg_array(df["dec"].to_numpy(dtype=float, copy=False), lat_deg)
    bad: list[str] = []
    for i, (_, row) in enumerate(df.iterrows()):
        pk = float(peaks[i])
        if pk < min_alt_deg - 1e-6:
            bad.append(f"{_star_label(row)} (peak {pk:.1f}°)")
    return bad


def _min_altitude_over_stars(
    df: pd.DataFrame,
    lat_deg: float,
    lon_deg_east: float,
    when_utc: datetime,
) -> float:
    """Minimum altitude among selected stars at this instant (bottleneck height)."""
    if df.empty:
        return 90.0
    cfg = HorizonConfig(lat_deg, lon_deg_east, when_utc)
    ra = df["ra"].to_numpy(dtype=float, copy=False)
    dec = df["dec"].to_numpy(dtype=float, copy=False)
    alts, _ = alt_az_arrays(ra, dec, cfg)
    return float(np.min(alts))


def _refine_maximin(
    df: pd.DataFrame,
    lat_deg: float,
    lon_deg_east: float,
    center: datetime,
    t_min: datetime,
    half_width: timedelta,
    *,
    floor_alt: float | None,
    step_minutes: int = 1,
) -> tuple[datetime, float]:
    """1-minute sweep in [center-half_width, center+half_width] to maximize bottleneck altitude."""
    lo_t = max(t_min, center - half_width)
    hi_t = center + half_width
    step = timedelta(minutes=step_minutes)
    best_t = center
    best_score = _min_altitude_over_stars(df, lat_deg, lon_deg_east, center)
    t = lo_t
    while t <= hi_t:
        score = _min_altitude_over_stars(df, lat_deg, lon_deg_east, t)
        if floor_alt is not None and score < floor_alt - 1e-9:
            t += step
            continue
        if score > best_score + 1e-9:
            best_score = score
            best_t = t
        t += step
    return best_t.replace(microsecond=0), best_score


def find_best_utc_maximin(
    df: pd.DataFrame,
    lat_deg: float,
    lon_deg_east: float,
    min_alt_deg: float,
    start_utc: datetime,
    *,
    max_days: int = 366,
    step_minutes: int = 20,
) -> tuple[datetime | None, str]:
    """
    Time in [start_utc, start_utc + max_days] that maximizes the *lowest* star altitude
    (maximin). If any instant meets min_alt_deg, the best time is chosen among those;
    otherwise the global maximin is returned with an explanatory note.
    """
    if max_days <= 0 or step_minutes <= 0:
        return None, "Search window and coarse step must be positive."

    if df.empty:
        return None, "No stars selected."

    t0 = start_utc
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    else:
        t0 = t0.astimezone(timezone.utc)

    end = t0 + timedelta(days=max_days)
    grid_step = timedelta(minutes=step_minutes)

    best_any_t = t0
    best_any_score = _min_altitude_over_stars(df, lat_deg, lon_deg_east, t0)
    best_feas_t: datetime | None = None
    best_feas_score = -999.0

    t = t0
    while t <= end:
        score = _min_altitude_over_stars(df, lat_deg, lon_deg_east, t)
        if score > best_any_score + 1e-9:
            best_any_score = score
            best_any_t = t
        if score >= min_alt_deg - 1e-9 and score > best_feas_score + 1e-9:
            best_feas_score = score
            best_feas_t = t
        t += grid_step

    use_feasible = best_feas_t is not None
    coarse_t = best_feas_t if use_feasible else best_any_t
    floor = min_alt_deg if use_feasible else None

    final_t, final_score = _refine_maximin(
        df,
        lat_deg,
        lon_deg_east,
        coarse_t,
        t0,
        grid_step,
        floor_alt=floor,
        step_minutes=1,
    )

    notes: list[str] = []
    if use_feasible:
        notes.append(
            f"When all stars are at least {min_alt_deg:.1f}° up, this time pushes the lowest "
            f"one as high as possible ({final_score:.1f}°)."
        )
    else:
        notes.append(
            f"No sampled time in this window had every star ≥ {min_alt_deg:.1f}°. "
            f"This time maximizes the lowest star ({final_score:.1f}°) — try a lower floor or "
            "a longer search."
        )

    never_floor = infeasible_star_labels(df, lat_deg, min_alt_deg)
    if never_floor:
        notes.append(
            "Some targets never reach your altitude floor from this latitude (see preview table "
            "once you lower the floor or drop those stars)."
        )

    return final_t, " ".join(notes)


def horizontal_unit_vector(alt_deg: float, az_deg: float) -> tuple[float, float, float]:
    """
    East-North-Zenith Cartesian unit vector for local horizontal coordinates.
    Matches altitude_azimuth: azimuth north = 0°, east = 90°.
    """
    alt = math.radians(alt_deg)
    az = math.radians(az_deg)
    ca = math.cos(alt)
    x = ca * math.sin(az)
    y = ca * math.cos(az)
    z = math.sin(alt)
    return x, y, z


def altitude_azimuth(ra_hours: float, dec_deg: float, cfg: HorizonConfig) -> tuple[float, float]:
    """
    Return (altitude_deg, azimuth_deg) measured from the horizon, north=0° east=90°.
    """
    alts, azs = alt_az_arrays(np.array([ra_hours], dtype=float), np.array([dec_deg], dtype=float), cfg)
    return float(alts[0]), float(azs[0])


def add_horizon_columns(df: pd.DataFrame, cfg: HorizonConfig) -> pd.DataFrame:
    if df.empty:
        return df.assign(alt_deg=pd.Series(dtype=float), az_deg=pd.Series(dtype=float))

    ra = df["ra"].to_numpy(dtype=float, copy=False)
    dec = df["dec"].to_numpy(dtype=float, copy=False)
    alts, azs = alt_az_arrays(ra, dec, cfg)
    out = df.copy()
    out["alt_deg"] = alts
    out["az_deg"] = azs
    return out
