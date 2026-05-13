"""Load HYG 3.x and resolve stars by constellation or proper name."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import pandas as pd

from constellations import CONSTELLATION_BY_LABEL

HYG_URL = (
    "https://github.com/astronexus/HYG-Database/raw/"
    "cb19d26a9910f5c0794a0dec72f29f2977eca2cc/hygdata_v3.csv"
)


def default_hyg_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "hygdata_v3.csv"


def ensure_hyg_csv(path: Path | None = None, force: bool = False) -> Path:
    p = path or default_hyg_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not force:
        return p
    urllib.request.urlretrieve(HYG_URL, p)
    return p


def load_hyg(path: Path | None = None) -> pd.DataFrame:
    if path is not None and path.exists():
        p = path
    else:
        p = ensure_hyg_csv(path)
    df = pd.read_csv(p, dtype={"proper": str, "con": str, "bayer": str})
    for col in ("ra", "dec", "mag"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["ra"].notna() & df["dec"].notna() & df["mag"].notna()]
    df = df[df["proper"] != "Sol"]
    return df


def normalize_con_code(df: pd.DataFrame, raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    u = raw.upper()
    for c in df["con"].dropna().unique():
        s = str(c)
        if s.upper() == u:
            return s
    return None


def parse_extra_codes(text: str, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    out: list[str] = []
    bad: list[str] = []
    for part in re.split(r"[,;\s]+", text.strip()):
        if not part:
            continue
        canon = normalize_con_code(df, part)
        if canon:
            out.append(canon)
        else:
            bad.append(part)
    return out, bad


def stars_for_constellation(df: pd.DataFrame, con: str, mag_limit: float) -> pd.DataFrame:
    con = con.strip()
    if not con:
        return df.iloc[0:0]
    sub = df[(df["con"] == con) & (df["mag"] <= mag_limit)].copy()
    return sub.sort_values("mag")


def stars_by_proper_names(df: pd.DataFrame, names: list[str], mag_limit: float) -> pd.DataFrame:
    if not names:
        return df.iloc[0:0]
    norm = {n.strip().lower(): n.strip() for n in names if n and n.strip()}
    if not norm:
        return df.iloc[0:0]
    m = df["proper"].fillna("").str.strip().str.lower().isin(norm.keys())
    sub = df[m & (df["mag"] <= mag_limit)].copy()
    return sub.sort_values("mag")


def report_proper_name_issues(df: pd.DataFrame, names: list[str], mag_limit: float) -> tuple[list[str], list[str]]:
    """Return (unknown_names, too_faint_messages) for requested proper names."""
    unknown: list[str] = []
    faint: list[str] = []
    for raw in names:
        key = raw.strip().lower()
        if not key:
            continue
        sub = df[df["proper"].fillna("").str.strip().str.lower() == key]
        if sub.empty:
            unknown.append(raw.strip())
        else:
            m = float(sub.iloc[0]["mag"])
            if m > mag_limit:
                faint.append(f"{raw.strip()} (mag {m:.2f} > limit {mag_limit})")
    return unknown, faint


def merge_star_selections(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    all_df = pd.concat(parts, ignore_index=True)
    if all_df.empty:
        return all_df
    return all_df.drop_duplicates(subset=["id"], keep="first").sort_values("mag")


def resolve_labels_to_codes(labels: list[str]) -> list[str]:
    codes: list[str] = []
    for label in labels:
        code = CONSTELLATION_BY_LABEL.get(label)
        if code:
            codes.append(code)
    return codes
