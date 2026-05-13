"""Streamlit UI: pick constellations / named stars, check horizon, export printable STL."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from catalog import (
    default_hyg_path,
    ensure_hyg_csv,
    load_hyg,
    merge_star_selections,
    parse_extra_codes,
    report_proper_name_issues,
    resolve_labels_to_codes,
    stars_by_proper_names,
    stars_for_constellation,
)
from constellations import LABELS_SORTED
from horizon import (
    HorizonConfig,
    add_horizon_columns,
    find_best_utc_maximin,
)
import mesh_export
import stl_preview

st.set_page_config(page_title="Star lamp / planetarium shade", layout="wide")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@st.cache_data(show_spinner=False)
def cached_hyg(csv_path_str: str) -> pd.DataFrame:
    return load_hyg(Path(csv_path_str))


_MAG_PRESETS: list[tuple[str, float | None]] = [
    ("Main stars only — clearest for kids", 3.5),
    ("Mostly bright stars", 4.5),
    ("Typical mix (default)", 5.0),
    ("Lots of stars (incl. fainter dots)", 6.2),
    ("Custom limit…", None),
]


def main() -> None:
    if "utc_date" not in st.session_state:
        st.session_state.utc_date = dt.date.today()
    if "utc_time" not in st.session_state:
        st.session_state.utc_time = dt.time(22, 0)

    st.title("Constellation lamp shade")

    with st.sidebar:
        st.header("Catalog")
        hyg_path = default_hyg_path()
        if st.button("Download / refresh HYG v3"):
            ensure_hyg_csv(hyg_path, force=True)
            cached_hyg.clear()
            st.success(f"Saved {hyg_path}")
        if not hyg_path.exists():
            st.warning("HYG catalog not found yet — click the button above once.")
        else:
            st.caption(str(hyg_path))

        st.subheader("Star brightness")
        st.caption(
            "Fainter stars have **higher** magnitude numbers. Only stars **at or above** "
            "your chosen brightness are used (we keep mag ≤ limit)."
        )
        preset_idx = st.selectbox(
            "Preset",
            range(len(_MAG_PRESETS)),
            index=2,
            format_func=lambda i: _MAG_PRESETS[i][0],
            help="Tighter limits drop faint pinholes so famous constellation shapes stand out.",
        )
        preset_limit = _MAG_PRESETS[preset_idx][1]
        if preset_limit is None:
            mag_limit = st.slider(
                "Max magnitude to include",
                2.0,
                6.5,
                5.0,
                0.1,
                help="Stars with apparent magnitude ≤ this value are included.",
            )
        else:
            mag_limit = preset_limit
            st.caption(
                f"Active limit: **magnitude ≤ {mag_limit:.1f}** — dimmer (higher-number) stars are left out."
            )

        st.divider()
        if st.button(
            "Reset session",
            help="Clear constellations cache in memory, time, STL, and reload defaults.",
        ):
            for _k in list(st.session_state.keys()):
                del st.session_state[_k]
            st.rerun()

    if not hyg_path.exists():
        st.stop()

    df = cached_hyg(str(hyg_path.resolve()))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Constellations")
        labels = st.multiselect(
            "IAU constellations",
            options=LABELS_SORTED,
            default=["Ursa Major", "Orion", "Cassiopeia"],
            help="Uses your brightness preset from the sidebar (stars with mag ≤ that limit).",
        )
        extra = st.text_input(
            "Extra IAU codes (optional)",
            placeholder="e.g. UMi, CMa",
            help="Three-letter abbreviations from the HYG `con` column, comma or space separated.",
        )

    with c2:
        st.subheader("Named stars")
        extra_names = st.text_area(
            "Proper names (comma-separated)",
            value="Polaris",
            height=100,
            help="HYG `proper` field, e.g. Polaris, Betelgeuse, Sirius. Case-insensitive.",
        )

    parts: list[pd.DataFrame] = []
    for code in resolve_labels_to_codes(labels):
        parts.append(stars_for_constellation(df, code, mag_limit))
    extra_list, extra_bad = parse_extra_codes(extra, df)
    if extra_bad:
        st.warning(f"Unknown constellation codes: {', '.join(extra_bad)}")
    for code in extra_list:
        sub = df[(df["con"] == code) & (df["mag"] <= mag_limit)]
        if sub.empty:
            st.warning(f"No stars found for constellation code `{code}` with mag ≤ {mag_limit}.")
        parts.append(sub.copy())

    names = [x.strip() for x in extra_names.replace("\n", ",").split(",") if x.strip()]
    unk, faint = report_proper_name_issues(df, names, mag_limit)
    if unk:
        st.warning("Unknown proper names (not in HYG): " + ", ".join(unk))
    if faint:
        st.warning(
            "Magnitudes too faint for your limit — raise max magnitude or pick another star: "
            + "; ".join(faint)
        )
    parts.append(stars_by_proper_names(df, names, mag_limit))

    selected = merge_star_selections(parts)

    st.subheader("Observer & time")
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        lat = st.number_input(
            "Latitude (°, +north)",
            value=53.1142,
            format="%.4f",
            help="Default: Sherwood Observatory, Sutton-in-Ashfield, UK.",
        )
    with oc2:
        lon = st.number_input(
            "Longitude (°, +east)",
            value=-1.2220,
            format="%.4f",
            help="Default: Sherwood Observatory (1°13′W → −1.2220°).",
        )
    with oc3:
        min_alt = st.number_input(
            'Minimum altitude (°, "fake horizon")',
            value=0.0,
            step=1.0,
            help="With **Sky** dome layout, only stars at least this high appear in the STL. "
            "Preview always lists all selected stars.",
        )

    pud = st.session_state.pop("pending_utc_date", None)
    put = st.session_state.pop("pending_utc_time", None)
    if pud is not None:
        st.session_state.utc_date = pud
    if put is not None:
        st.session_state.utc_time = put

    tc1, tc2 = st.columns(2)
    with tc1:
        obs_date = st.date_input("Date (UTC)", key="utc_date")
    with tc2:
        obs_time = st.time_input("Time (UTC)", key="utc_time")
    when = dt.datetime.combine(obs_date, obs_time, tzinfo=dt.timezone.utc)

    if "find_msg" in st.session_state:
        kind, text = st.session_state.pop("find_msg")
        (st.success if kind == "ok" else st.warning)(text)

    with st.expander("Best time for this selection (maximize joint height)"):
        st.caption(
            "Chooses a UTC time in the search window that makes the **lowest** star in your "
            "list as high as possible (best “common” height). If your minimum altitude can be "
            "met somewhere, only those moments are considered, then the lowest star is pushed "
            "as high as possible. Astronomical horizon only (no Sun/twilight)."
        )
        max_days = st.slider("Maximum days to search", 30, 730, 366, step=1)
        step_coarse = st.selectbox("Coarse step (minutes)", options=[10, 15, 20, 30], index=2)
        go = st.button(
            "Set date & time to best joint height",
            disabled=selected.empty,
            help="Uses latitude, longitude, minimum altitude as a floor (when achievable), and current stars.",
        )
        if go:
            found, msg = find_best_utc_maximin(
                selected,
                float(lat),
                float(lon),
                float(min_alt),
                when,
                max_days=int(max_days),
                step_minutes=int(step_coarse),
            )
            if found:
                st.session_state.pending_utc_date = found.date()
                st.session_state.pending_utc_time = found.time()
                st.session_state.find_msg = (
                    "ok",
                    f"{found.isoformat()} UTC — {msg}",
                )
            else:
                st.session_state.find_msg = ("bad", msg)
            st.rerun()

    cfg = HorizonConfig(lat_deg=float(lat), lon_deg_east=float(lon), when_utc=when)
    with_horizon = add_horizon_columns(selected, cfg)

    st.subheader("Preview")
    ok = False
    if with_horizon.empty:
        st.info("No stars match your filters — lower magnitude, add names, or fix codes.")
    else:
        show = with_horizon[
            [
                "proper",
                "con",
                "bayer",
                "mag",
                "ra",
                "dec",
                "alt_deg",
                "az_deg",
            ]
        ].copy()
        show["proper"] = show["proper"].fillna("")
        show["bayer"] = show["bayer"].fillna("")

        viol = show[show["alt_deg"] < min_alt]
        ok = len(viol) == 0
        if ok:
            st.success(
                f"All {len(show)} stars are ≥ {min_alt:.1f}° altitude at {when.isoformat()} "
                f"(lat {lat:.2f}°, lon {lon:.2f}°)."
            )
        else:
            st.error(
                f"{len(viol)} star(s) below {min_alt:.1f}° — change date/time, location, or selection."
            )
        st.dataframe(show.sort_values("alt_deg", ascending=False), width="stretch")

    st.subheader("Geometry & STL export")
    dome_mapping = st.radio(
        "Dome layout",
        options=["sky", "globe"],
        index=0,
        format_func=lambda k: (
            "Sky at your date & time — like looking up (recommended)"
            if k == "sky"
            else "Celestial globe — fixed RA/Dec on the shell"
        ),
        help="Sky mode maps holes by altitude/azimuth so Orion matches your southern sky at the "
        "chosen instant. Globe mode uses classic planetarium RA/Dec (no horizon cut on the mesh).",
    )

    if with_horizon.empty:
        for_stl = with_horizon
    else:
        if dome_mapping == "sky":
            for_stl = with_horizon[with_horizon["alt_deg"] >= float(min_alt) - 1e-9].copy()
            st.caption(
                f"STL will drill **{len(for_stl)}** star(s) at ≥ {float(min_alt):.1f}° "
                f"(of {len(with_horizon)} in the table)."
            )
        else:
            for_stl = with_horizon.copy()
            st.caption(
                f"STL will drill all **{len(for_stl)}** stars; horizon altitude only affects the table."
            )

    if not with_horizon.empty and for_stl.empty and dome_mapping == "sky":
        st.warning(
            "No stars clear your minimum altitude for the STL. Lower the altitude cut, adjust "
            "date/time, or switch to **Celestial globe** to print all listed stars."
        )

    g1, g2 = st.columns(2)
    with g1:
        outer_diameter = st.number_input(
            "Outer diameter (mm)",
            value=160.0,
            min_value=40.0,
            help="Hollow shell outer diameter; inner lip is this minus twice the wall thickness.",
        )
        shell_radius = float(outer_diameter) / 2.0
        shell_thickness = st.number_input("Shell wall thickness (mm)", value=3.0, min_value=0.5)
    with g2:
        base_rr = st.number_input("Rod base radius (mm)", value=1.5, min_value=0.2)
        min_rr = st.number_input("Rod min radius (mm)", value=0.35, min_value=0.1)
        mq_map = dict(mesh_export.mesh_quality_presets())
        mq = st.selectbox(
            "STL mesh quality",
            options=list(mq_map.keys()),
            index=0,
            help="Finer meshes look smoother when printed; High is heavier to compute.",
        )
        ico_sub, cyl_sec = mq_map[mq]

    st.markdown("##### Export STL")
    st.caption(
        "**trimesh** + **manifold3d**: hollow shell with through-holes for each star. "
        "Download the file and print."
    )

    if "stl_blob" not in st.session_state:
        st.session_state["stl_blob"] = None
    if "stl_name" not in st.session_state:
        st.session_state["stl_name"] = "lamp_shade.stl"

    build_py_stl = st.button(
        "Build STL for 3D printing",
        type="primary",
        disabled=with_horizon.empty or for_stl.empty,
        help="Creates `output/lamp_shade.stl` and enables download.",
    )

    if build_py_stl:
        try:
            with st.spinner("Computing mesh — wait for all star subtractions to finish…"):
                mesh = mesh_export.build_lamp_mesh(
                    for_stl,
                    shell_radius=float(shell_radius),
                    shell_thickness=float(shell_thickness),
                    base_rod_radius=float(base_rr),
                    minimum_rod_radius=float(min_rr),
                    ico_subdiv=int(ico_sub),
                    cylinder_sections=int(cyl_sec),
                    placement="local_sky" if dome_mapping == "sky" else "equatorial",
                )
                stl_bytes = mesh_export.mesh_to_stl_bytes(mesh)
            stl_path = OUTPUT_DIR / "lamp_shade.stl"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stl_path.write_bytes(stl_bytes)
            st.session_state["stl_blob"] = stl_bytes
            st.session_state["stl_name"] = "lamp_shade.stl"
            st.success(
                f"STL ready ({len(stl_bytes) // 1024} KB, watertight={mesh.is_watertight}, "
                f"{len(for_stl)} holes). Saved `{stl_path}` — download below."
            )
        except Exception as e:
            st.session_state["stl_blob"] = None
            st.error(f"Mesh build failed: {e}")

    blob = st.session_state.get("stl_blob")
    if blob:
        st.markdown("##### 3D preview")
        st.caption(
            "Drag to rotate, scroll or pinch to zoom. Very dense meshes are simplified slightly for speed."
        )
        try:
            fig = stl_preview.figure_from_stl_bytes(blob)
            st.plotly_chart(fig, use_container_width=True, key="stl_preview_chart")
        except Exception as e:
            st.warning(f"Could not build 3D preview: {e}")

        st.download_button(
            label=f"Download {st.session_state.get('stl_name', 'lamp_shade.stl')}",
            data=blob,
            file_name=st.session_state.get("stl_name", "lamp_shade.stl"),
            mime="model/stl",
            key="stl_download_btn",
        )


if __name__ == "__main__":
    main()
