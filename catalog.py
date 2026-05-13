"""Load HYG 3.x and resolve stars by constellation or proper name."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal

import pandas as pd

from constellations import CONSTELLATION_BY_LABEL

HYG_URL = (
    "https://github.com/astronexus/HYG-Database/raw/"
    "cb19d26a9910f5c0794a0dec72f29f2977eca2cc/hygdata_v3.csv"
)
# Plain CSV symlink area on upstream ``main`` (file name bumps: v41, v42, …).
RAW_MAIN_PREFIX = "https://raw.githubusercontent.com/astronexus/HYG-Database/main/"
GITHUB_API = "https://api.github.com/repos/astronexus/HYG-Database"
_m = re.search(r"/([0-9a-f]{40})/hygdata_v3\.csv", HYG_URL)
if _m is None:
    raise RuntimeError("Pinned HYG_URL must embed a full 40-char git commit.")
HYG_PINNED_COMMIT: str = _m.group(1)

_HYG_PLAIN_CURRENT = re.compile(r"^hygdata_v\d+(?:_\d+)?\.csv$")


def _github_json(
    url: str,
    timeout: float,
    *,
    github_token: str | None = None,
) -> object | None:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "star-lamp/1.0 (HYG freshness; github.com)",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token:
        tok = github_token.strip()
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError):
        return None


def hyg_plain_csv_version_key(filename: str) -> tuple[int, int]:
    """Sort key for ``hygdata_v38`` / ``hygdata_v35_1`` style names."""
    m = re.match(r"hygdata_v(\d+)(?:_(\d+))?\.csv$", filename, re.I)
    if m is None:
        return (-1, -1)
    return int(m.group(1)), int(m.group(2) or 0)


def hyg_current_catalog_relpath(
    timeout: float = 12.0,
    *,
    github_token: str | None = None,
) -> str | None:
    """Latest plain ``hyg/CURRENT/hygdata_v*.csv`` on ``main``, or ``None`` if the API fails."""
    url = f"{GITHUB_API}/contents/hyg/CURRENT?ref=main"
    blob = _github_json(url, timeout, github_token=github_token)
    if not isinstance(blob, list):
        return None
    names = [
        e.get("name")
        for e in blob
        if isinstance(e, dict)
        and e.get("type") == "file"
        and isinstance(e.get("name"), str)
        and _HYG_PLAIN_CURRENT.match(e["name"])
    ]
    if not names:
        return None
    best = max(names, key=lambda n: hyg_plain_csv_version_key(n))
    return f"hyg/CURRENT/{best}"


def hyg_resolve_current_catalog_download(
    timeout: float = 30.0,
    *,
    github_token: str | None = None,
) -> tuple[str, str] | None:
    """``(raw_url, repo_relative_path)`` for the newest CURRENT plain CSV."""
    rel = hyg_current_catalog_relpath(timeout, github_token=github_token)
    return None if rel is None else (RAW_MAIN_PREFIX + rel, rel)


def hyg_latest_main_commit_for_path(
    relpath: str,
    timeout: float = 12.0,
    *,
    github_token: str | None = None,
) -> str | None:
    """Most recent ``main`` commit that touched ``relpath`` (full repo path)."""
    q = urllib.parse.quote(relpath, safe="")
    api = f"{GITHUB_API}/commits?path={q}&sha=main&per_page=1"
    blob = _github_json(api, timeout, github_token=github_token)
    if not isinstance(blob, list) or not blob:
        return None
    sha = blob[0].get("sha")
    if isinstance(sha, str) and len(sha.strip()) >= 40:
        return sha.strip().lower()[:40]
    return None


def hyg_upstream_current_commit(
    timeout: float = 12.0,
    *,
    github_token: str | None = None,
) -> str | None:
    """Git SHA for the upstream tip commit affecting the CURRENT plain CSV (for freshness checks)."""
    rel = hyg_current_catalog_relpath(timeout, github_token=github_token)
    return (
        None
        if rel is None
        else hyg_latest_main_commit_for_path(rel, timeout, github_token=github_token)
    )


def default_hyg_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "hygdata_v3.csv"


def hyg_commit_sidecar(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.name + ".commit")


def read_hyg_download_commit(csv_path: Path) -> str | None:
    side = hyg_commit_sidecar(csv_path)
    if not side.exists():
        return None
    try:
        txt = side.read_text(encoding="utf-8").strip().split()[0].lower()
        return txt[:40] if len(txt) >= 40 else None
    except OSError:
        return None


def write_hyg_download_commit(csv_path: Path, git_commit_sha40: str) -> None:
    sha = git_commit_sha40.strip().lower()[:40]
    if len(sha) != 40:
        return
    hyg_commit_sidecar(csv_path).write_text(sha + "\n", encoding="utf-8")


HyFreshness = Literal["missing", "current", "stale", "unknown"]


def hyg_catalog_freshness(csv_path: Path, upstream_main_commit: str | None) -> HyFreshness:
    if not csv_path.exists():
        return "missing"
    if upstream_main_commit is None:
        return "unknown"
    u = upstream_main_commit.lower()[:40]
    stored = read_hyg_download_commit(csv_path)
    if stored:
        return "current" if stored == u else "stale"
    return "stale"


def ensure_hyg_csv(
    path: Path | None = None,
    force: bool = False,
    *,
    github_token: str | None = None,
) -> Path:
    p = path or default_hyg_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not force:
        return p
    if force:
        resolved = hyg_resolve_current_catalog_download(timeout=60.0, github_token=github_token)
        if resolved is None:
            urllib.request.urlretrieve(HYG_URL, p)
            write_hyg_download_commit(p, HYG_PINNED_COMMIT)
        else:
            url, rel = resolved
            urllib.request.urlretrieve(url, p)
            latest = hyg_latest_main_commit_for_path(rel, timeout=30.0, github_token=github_token)
            if latest:
                write_hyg_download_commit(p, latest)
        return p
    urllib.request.urlretrieve(HYG_URL, p)
    write_hyg_download_commit(p, HYG_PINNED_COMMIT)
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
