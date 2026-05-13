"""Altitude / azimuth for stars given observer location and UTC time."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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


def ra_hours_to_deg(ra_h: float) -> float:
    return (ra_h % 24.0) * 15.0


def _wrap_pm180(x: float) -> float:
    x = x % 360.0
    if x > 180.0:
        x -= 360.0
    return x


@dataclass(frozen=True)
class HorizonConfig:
    lat_deg: float
    lon_deg_east: float
    when_utc: datetime


def peak_altitude_deg(dec_deg: float, lat_deg: float) -> float:
    """Best possible altitude (upper culmination) for a given declination and latitude."""
    dec = math.radians(dec_deg)
    lat = math.radians(lat_deg)
    a = math.sin(dec) * math.sin(lat)
    b = math.cos(dec) * math.cos(lat)
    s = min(1.0, max(-1.0, a + abs(b)))
    return math.degrees(math.asin(s))


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
    bad: list[str] = []
    for _, row in df.iterrows():
        pk = peak_altitude_deg(float(row["dec"]), lat_deg)
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
    cfg = HorizonConfig(lat_deg, lon_deg_east, when_utc)
    lo = 90.0
    for _, row in df.iterrows():
        alt, _ = altitude_azimuth(float(row["ra"]), float(row["dec"]), cfg)
        lo = min(lo, alt)
    return lo


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


def altitude_azimuth(ra_hours: float, dec_deg: float, cfg: HorizonConfig) -> tuple[float, float]:
    """
    Return (altitude_deg, azimuth_deg) measured from the horizon, north=0° east=90°.
    """
    jd = _julian_date(cfg.when_utc)
    lst = local_sidereal_time_deg(jd, cfg.lon_deg_east)
    ra_deg = ra_hours_to_deg(ra_hours)
    h_deg = _wrap_pm180(lst - ra_deg)
    h = math.radians(h_deg)
    dec = math.radians(dec_deg)
    lat = math.radians(cfg.lat_deg)

    sin_alt = math.sin(dec) * math.sin(lat) + math.cos(dec) * math.cos(lat) * math.cos(h)
    sin_alt = min(1.0, max(-1.0, sin_alt))
    alt = math.asin(sin_alt)

    y = -math.sin(h) * math.cos(dec)
    x = math.cos(lat) * math.sin(dec) - math.sin(lat) * math.cos(dec) * math.cos(h)
    az = math.atan2(y, x)
    az_deg = math.degrees(az) % 360.0
    return math.degrees(alt), az_deg


def add_horizon_columns(df: pd.DataFrame, cfg: HorizonConfig) -> pd.DataFrame:
    if df.empty:
        return df.assign(alt_deg=pd.Series(dtype=float), az_deg=pd.Series(dtype=float))

    alts = []
    azs = []
    for _, row in df.iterrows():
        alt, az = altitude_azimuth(float(row["ra"]), float(row["dec"]), cfg)
        alts.append(alt)
        azs.append(az)
    out = df.copy()
    out["alt_deg"] = alts
    out["az_deg"] = azs
    return out
