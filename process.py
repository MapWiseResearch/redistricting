#!/usr/bin/env python3
"""
process.py — Build unified, coast-clipped redistricting plan files.

For each plan directory (fair, dem, gop, post_vra):
  1. Read per-state district-data.csv (from zip) and district-shapes.geojson
  2. Fuzzy-intersect column names across all states, remap to canonical names
  3. Join CSV rows to GeoJSON features by normalised district label
  4. Assign district codes (AK-01, CA-10, …)
  5. Rename & reorder columns: elections first, then demographics
  6. Fix invalid geometries, clip to coastal boundary (template.geojson),
     simplify polygons, drop slivers
  7. Write <plan>.geojson and <plan>.csv to the base directory

Usage:
    python3 process.py
"""

import csv
import io
import json
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import topojson
from rapidfuzz import process, fuzz as _fuzz
from shapely import set_precision
from shapely.geometry import shape, MultiPolygon
from shapely.strtree import STRtree
from shapely.validation import make_valid

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent
RAW_DATA = BASE / "raw_data"
GEOJSONS_OUT = BASE / "geojsons"
PROCESSED_OUT = BASE / "processed_data"
TEMPLATE = BASE / "template.geojson"
PLAN_DIRS = ["fair", "dem", "gop", "post_vra", "current"]

META_COLS = ["district_code", "state", "district_num"]

# Douglas-Peucker tolerance (~0.005° ≈ 500 m at mid-latitudes)
SIMPLIFY_TOLERANCE = 0.002

# Drop polygon parts smaller than this after clipping (sq-degrees)
SLIVER_AREA = 1e-6

# ---------------------------------------------------------------------------
# Column rename / reorder
# ---------------------------------------------------------------------------

_ELECTION_DATASET_LABELS = {
    "PRES": "pres", "SEN": "sen", "GOV": "gov", "AG": "ag",
    "LTG": "ltg", "SOS": "sos", "TREAS": "treas", "AUD": "aud",
    "COMP": "comp", "CMPTR": "cmptr",
}
_DEMO_DATASET_LABELS = {
    "CENS": "pop", "CENS_ADJ": "pop_adj", "ACS": "acs",
    "VAP": "vap", "VAP_NH": "vap_nh", "CVAP": "cvap",
}
_FIELD_LABELS = {
    "Total": "total", "Dem": "dem", "Rep": "rep",
    "White": "white", "Hispanic": "hispanic", "Black": "black",
    "Asian": "asian", "Native": "native", "Pacific": "pacific",
    "BlackAlone": "black_alone", "AsianAlone": "asian_alone",
    "NativeAlone": "native_alone", "PacificAlone": "pacific_alone",
    "OtherAlone": "other_alone", "TwoOrMore": "two_or_more",
}
_ELECTION_DATASET_ORDER = {
    "PRES": 0, "SEN": 1, "GOV": 2, "COMP": 3,
    "AG": 4, "LTG": 5, "SOS": 6, "TREAS": 7, "AUD": 8, "CMPTR": 9,
}
_DEMO_DATASET_ORDER = {
    "CENS": 0, "CENS_ADJ": 1, "ACS": 2,
    "VAP": 3, "VAP_NH": 4, "CVAP": 5,
}
_FIELD_ORDER = {
    "total": 0, "dem": 1, "rep": 2,
    "white": 3, "hispanic": 4, "black": 5, "asian": 6,
    "native": 7, "pacific": 8,
    "black_alone": 9, "asian_alone": 10, "native_alone": 11,
    "pacific_alone": 12, "other_alone": 13, "two_or_more": 14,
}


def _parse_dra_col(col: str) -> dict | None:
    parts = col.split("_", 3)
    if len(parts) < 3:
        return None
    category = parts[0]
    if category not in ("E", "T", "V", "X"):
        return None
    year_raw = parts[1]
    year_key = re.sub(r"[^0-9]", "", year_raw.split("-")[0])
    try:
        year_int = int(year_key)
    except ValueError:
        return None
    rest = parts[2] if len(parts) == 3 else parts[2] + "_" + parts[3]
    rest_parts = rest.split("_")
    known_datasets = sorted(
        set(_ELECTION_DATASET_LABELS) | set(_DEMO_DATASET_LABELS),
        key=len,
        reverse=True,
    )
    dataset = rest_parts[0]
    field_parts = rest_parts[1:]
    for candidate in known_datasets:
        candidate_parts = candidate.split("_")
        if rest_parts[:len(candidate_parts)] == candidate_parts:
            dataset = candidate
            field_parts = rest_parts[len(candidate_parts):]
            break
    field = "_".join(field_parts)
    return {"category": category, "year_raw": year_raw,
            "year_int": year_int, "dataset": dataset, "field": field}


def rename_col(col: str) -> str:
    p = _parse_dra_col(col)
    if p is None:
        return col.lower()
    cat, yr, dataset, field = p["category"], p["year_raw"], p["dataset"], p["field"]
    yr2 = re.sub(r"20(\d\d)", r"\1", yr)
    field_label = _FIELD_LABELS.get(field, field.lower())
    if cat == "E":
        ds_label = _ELECTION_DATASET_LABELS.get(dataset, dataset.lower())
        return f"{ds_label}{yr2}_{field_label}"
    if cat in ("T", "V"):
        ds_label = _DEMO_DATASET_LABELS.get(dataset, dataset.lower())
        return f"{ds_label}_{field_label}"
    return col.lower()


def col_sort_key(col: str) -> tuple:
    p = _parse_dra_col(col)
    if p is None:
        return (2, 0, 0, 99, col)
    cat, yr, dataset = p["category"], p["year_int"], p["dataset"]
    field_label = _FIELD_LABELS.get(p["field"], p["field"].lower())
    field_ord = _FIELD_ORDER.get(field_label, 99)
    if cat == "E":
        return (0, yr, _ELECTION_DATASET_ORDER.get(dataset, 99), field_ord, col)
    if cat in ("T", "V"):
        return (1, _DEMO_DATASET_ORDER.get(dataset, 99), 0, field_ord, col)
    return (2, 0, 0, 99, col)

# ---------------------------------------------------------------------------
# State / district helpers
# ---------------------------------------------------------------------------

STATE_ABBR_MAP = {
    "ak": "AK", "al": "AL", "ar": "AR", "az": "AZ",
    "ca": "CA", "co": "CO", "ct": "CT", "de": "DE",
    "fl": "FL", "ga": "GA", "hi": "HI", "ia": "IA",
    "id": "ID", "il": "IL", "in": "IN", "ks": "KS",
    "ky": "KY", "la": "LA", "ma": "MA", "md": "MD",
    "me": "ME", "mi": "MI", "mn": "MN", "mo": "MO",
    "ms": "MS", "mt": "MT", "nc": "NC", "nd": "ND",
    "ne": "NE", "nh": "NH", "nj": "NJ", "nm": "NM",
    "nv": "NV", "ny": "NY", "oh": "OH", "ok": "OK",
    "or": "OR", "pa": "PA", "ri": "RI", "sc": "SC",
    "sd": "SD", "tn": "TN", "tx": "TX", "ut": "UT",
    "va": "VA", "vt": "VT", "wa": "WA", "wi": "WI",
    "wv": "WV", "wy": "WY",
}

_STRIP_PATTERNS = re.compile(
    r"(?i)^(congressional\s+)?district\s*#?\s*|^cd\s*#?\s*|^district\s*"
)


def normalise_label(label: str) -> str:
    label = _STRIP_PATTERNS.sub("", label.strip()).strip()
    try:
        return str(int(label))
    except ValueError:
        return label.lower()


def district_code(state_abbr: str, label: str) -> str:
    norm = normalise_label(label)
    try:
        return f"{state_abbr}-{int(norm):02d}"
    except ValueError:
        return f"{state_abbr}-{norm.upper()}"

# ---------------------------------------------------------------------------
# Data reading
# ---------------------------------------------------------------------------

def read_csv_from_zip(zip_path: Path) -> tuple[list[str], list[dict]]:
    with zipfile.ZipFile(zip_path) as z:
        with z.open("district-data.csv") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            cols = reader.fieldnames or []
            rows = list(reader)
    return list(cols), rows


def collect_state_data(plan_dir: Path) -> dict[str, dict]:
    states: dict[str, dict] = {}
    for entry in sorted(plan_dir.iterdir()):
        if not entry.is_dir():
            continue
        state_key = entry.name.lower()
        abbr = STATE_ABBR_MAP.get(state_key)
        if abbr is None:
            continue
        zip_path = entry / "district-data.zip"
        geo_path = entry / "district-shapes.geojson"
        if not zip_path.exists() or not geo_path.exists():
            continue
        try:
            cols, rows = read_csv_from_zip(zip_path)
            with open(geo_path, encoding="utf-8") as f:
                features = json.load(f).get("features", [])
        except Exception as e:
            print(f"  SKIP {state_key}: {e}")
            continue
        states[state_key] = {"abbr": abbr, "cols": cols, "rows": rows, "features": features}
    return states

# ---------------------------------------------------------------------------
# Fuzzy column intersection
# ---------------------------------------------------------------------------

_EXCLUSIVE_DATASETS = frozenset({"VAP", "CVAP", "ACS", "CENS", "CENS_ADJ"})
_COMPATIBLE_DATASET_GROUPS = (
    frozenset({"CENS", "CENS_ADJ"}),
)
_FIELD_SUFFIX = {"D": "Dem", "R": "Rep", "T": "Total", "Tot": "Total"}
_DATASET_KEYWORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)(?:^|[_\s.(])pres(ident)?[_\s.),]?"), "PRES"),
    (re.compile(r"(?i)(?:^|[_\s.(])sen(ate|ator)?[_\s.),]?"), "SEN"),
    (re.compile(r"(?i)(?:^|[_\s.(])gov(ernor)?[_\s.),]?"), "GOV"),
    (re.compile(r"(?i)(?:^|[_\s.(])ag[_\s.),]"), "AG"),
]


def _get_col_year(col: str) -> str | None:
    parts = col.split("_", 2)
    if len(parts) < 2:
        return None
    m = re.match(r"^(\d+)", parts[1])
    if not m:
        return None
    n = int(m.group(1))
    if 8 <= n <= 30:
        return f"{n:02d}"
    if 2008 <= n <= 2030:
        return f"{n % 100:02d}"
    return None


def _normalize_col(col: str) -> str:
    if col.startswith("X_"):
        parts = col.split("_", 2)
        if len(parts) == 3:
            year, rest = parts[1], parts[2]
            field_m = re.search(r"_([A-Za-z]+)$", rest)
            field = None
            if field_m:
                raw = field_m.group(1)
                field = _FIELD_SUFFIX.get(raw, raw.capitalize())
            dataset = None
            for pattern, name in _DATASET_KEYWORDS:
                if pattern.search(rest):
                    dataset = name
                    break
            if dataset and field:
                return f"X_{year}_{dataset}_{field}"
    col = re.sub(r"(?i)\bpresident\b", "PRES", col)
    col = re.sub(r"(?i)\bsenator\b", "SEN", col)
    col = re.sub(r"(?i)\bgovernor\b", "GOV", col)
    col = re.sub(r"_D$", "_Dem", col)
    col = re.sub(r"_R$", "_Rep", col)
    col = re.sub(r"_Tot\b", "_Total", col)
    return col


def _canonical_match_parts(col: str) -> dict | None:
    normalized = _normalize_col(col)
    parsed = _parse_dra_col(normalized)
    if parsed is None:
        return None
    parsed = parsed.copy()
    if parsed["category"] == "X" and parsed["dataset"] in _ELECTION_DATASET_LABELS:
        parsed["category"] = "E"
    return parsed


def _col_score(a: str, b: str, **_kwargs) -> float:
    parsed_a = _canonical_match_parts(a)
    parsed_b = _canonical_match_parts(b)
    if parsed_a is not None and parsed_b is not None:
        if parsed_a["category"] != parsed_b["category"]:
            return 0.0
        if parsed_a["year_int"] != parsed_b["year_int"]:
            return 0.0

        field_a = _FIELD_SUFFIX.get(parsed_a["field"], parsed_a["field"]).upper()
        field_b = _FIELD_SUFFIX.get(parsed_b["field"], parsed_b["field"]).upper()
        if field_a != field_b:
            return 0.0

        datasets = {parsed_a["dataset"], parsed_b["dataset"]}
        compatible = any(datasets <= group for group in _COMPATIBLE_DATASET_GROUPS)
        if parsed_a["dataset"] != parsed_b["dataset"] and not compatible:
            return 0.0

    ya, yb = _get_col_year(a), _get_col_year(b)
    if ya is not None and yb is not None and ya != yb:
        return 0.0
    toks_a = {t.upper() for t in re.split(r"[_\s\-]+", a)}
    toks_b = {t.upper() for t in re.split(r"[_\s\-]+", b)}
    excl_a = toks_a & _EXCLUSIVE_DATASETS
    excl_b = toks_b & _EXCLUSIVE_DATASETS
    compatible = any((excl_a | excl_b) <= group for group in _COMPATIBLE_DATASET_GROUPS)
    if excl_a and excl_b and not excl_a & excl_b and not compatible:
        return 0.0
    return _fuzz.token_sort_ratio(_normalize_col(a), _normalize_col(b))


def fuzzy_intersect_columns(
    states: dict[str, dict], threshold: int = 80, min_coverage: float = 0.5
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Return columns present (with fuzzy match) in at least `min_coverage` fraction of states.

    States that lack a matched column will leave it blank in the output rather than
    causing the column to be dropped entirely.  This lets X_22_* demographic fields
    survive even when a handful of states were exported without them.
    """
    skip = {"ID", "Label"}
    state_cols = {k: [c for c in v["cols"] if c not in skip] for k, v in states.items()}
    if not state_cols:
        return [], {}

    ref_key = max(state_cols, key=lambda k: len(state_cols[k]))
    ref_cols = state_cols[ref_key]
    other_keys = [k for k in state_cols if k != ref_key]
    n_states = len(states)

    canonical_cols: list[str] = []
    remap: dict[str, dict[str, str]] = {k: {} for k in states}
    for col in ref_cols:
        remap[ref_key][col] = col

    for ref_col in ref_cols:
        per_state_match: dict[str, str] = {ref_key: ref_col}
        matched = 1  # ref_key always counts

        for other_key in other_keys:
            other_col_list = state_cols[other_key]
            if not other_col_list:
                continue
            if ref_col in other_col_list:
                per_state_match[other_key] = ref_col
                matched += 1
                continue
            result = process.extractOne(ref_col, other_col_list, scorer=_col_score)
            if result and result[1] >= threshold:
                per_state_match[other_key] = result[0]
                matched += 1
            # else: state simply won't have this column (filled blank downstream)

        if matched / n_states >= min_coverage:
            canonical_cols.append(ref_col)
            for state_key, state_col in per_state_match.items():
                remap[state_key][state_col] = ref_col

    return canonical_cols, remap

# ---------------------------------------------------------------------------
# Geometry cleaning
# ---------------------------------------------------------------------------

def build_coast_mask() -> object:
    print(f"Building coast mask from {TEMPLATE.name} ...")
    tmpl = gpd.read_file(TEMPLATE)
    tmpl = tmpl[tmpl["NAMELSAD"] != "Delete"].copy()
    tmpl["geometry"] = tmpl["geometry"].apply(make_valid)
    coast = tmpl.geometry.union_all()
    print(f"  Coast mask ready: bounds {tuple(round(x, 2) for x in coast.bounds)}")
    return coast


def _drop_slivers(geom, min_area: float = SLIVER_AREA):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom if geom.area >= min_area else None
    if geom.geom_type == "MultiPolygon":
        parts = [p for p in geom.geoms if p.area >= min_area]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else MultiPolygon(parts)
    if not hasattr(geom, "geoms"):
        return None
    parts = []
    for sub in geom.geoms:
        if sub.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        cleaned = _drop_slivers(sub, min_area)
        if cleaned is not None:
            parts.append(cleaned)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(
        [p for p in parts if p.geom_type == "Polygon"]
    )


def _to_multipolygon(geom):
    """Extract only Polygon/MultiPolygon parts from any geometry (incl. GeometryCollection)."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom if geom.area > 0 else None
    if geom.geom_type == "MultiPolygon":
        return geom
    if hasattr(geom, "geoms"):
        parts = []
        for sub in geom.geoms:
            if sub.geom_type == "Polygon" and sub.area > 0:
                parts.append(sub)
            elif sub.geom_type == "MultiPolygon":
                parts.extend(p for p in sub.geoms if p.area > 0)
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else MultiPolygon(parts)
    return None


def resolve_overlaps(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Subtract overlap areas from the smaller district in each overlapping pair."""
    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        for j in tree.query(g):
            if j <= i:
                continue
            gj = geoms[j]
            if gj is None or gj.is_empty:
                continue
            if not g.intersects(gj):
                continue
            try:
                inter = g.intersection(gj)
            except Exception:
                continue
            if inter.is_empty or inter.area < 1e-8:
                continue
            # Subtract overlap from the smaller district, keep only polygons
            if g.area >= gj.area:
                geoms[j] = _to_multipolygon(make_valid(gj.difference(inter)))
            else:
                geoms[i] = _to_multipolygon(make_valid(g.difference(inter)))
                g = geoms[i]  # update local ref for subsequent iterations
    gdf = gdf.copy()
    gdf["geometry"] = geoms
    return gdf[gdf["geometry"].notna() & ~gdf["geometry"].is_empty].copy()


def clean_geometry(geom, coast):
    """Fix invalid geometry and clip to coast. Simplification done later as a batch."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    try:
        clipped = geom.intersection(coast)
    except Exception:
        clipped = geom.buffer(0).intersection(coast)
    if clipped is None or clipped.is_empty:
        return None
    return _drop_slivers(clipped)

# ---------------------------------------------------------------------------
# Joining
# ---------------------------------------------------------------------------

def exact_join(csv_rows: list[dict], geo_features: list[dict]) -> dict[str, dict]:
    csv_map: dict[str, dict] = {}
    for row in csv_rows:
        lbl = row.get("Label", "")
        if lbl == "Un":
            continue
        csv_map[normalise_label(lbl)] = row

    result: dict[str, dict] = {}
    for feat in geo_features:
        props = feat.get("properties", {})
        geo_name = str(props.get("NAME", ""))
        norm_name = normalise_label(geo_name)
        if norm_name in csv_map:
            result[str(props.get("id", geo_name))] = csv_map[norm_name]
        else:
            print(f"    WARNING: no CSV match for GeoJSON district '{geo_name}'")
    return result

# ---------------------------------------------------------------------------
# Main plan builder
# ---------------------------------------------------------------------------

def build_plan(plan_dir: Path, coast) -> None:
    print(f"\n{'='*60}")
    print(f"Plan: {plan_dir.name}")
    print(f"{'='*60}")

    states = collect_state_data(plan_dir)
    if not states:
        print("  No state data found — skipping.")
        return

    print(f"  States: {sorted(states.keys())}")

    common_cols, col_remap = fuzzy_intersect_columns(states)
    print(f"  Common data columns: {len(common_cols)}")

    sorted_data_cols = sorted(common_cols, key=col_sort_key)
    rename_map = {col: rename_col(col) for col in sorted_data_cols}
    print(f"  Column order: {[rename_map[c] for c in sorted_data_cols]}")

    rows_out: list[dict] = []
    geo_rows: list[dict] = []   # {props dict + shapely geometry}

    for state_key, state in sorted(states.items()):
        abbr = state["abbr"]
        print(f"  {abbr} ({len(state['features'])} districts) ...", end="")

        join_map = exact_join(state["rows"], state["features"])
        state_remap = col_remap.get(state_key, {})
        matched = 0

        for feat in state["features"]:
            props = feat.get("properties", {})
            feat_id = str(props.get("id", ""))
            geo_name = str(props.get("NAME", ""))

            if feat_id == "0" or geo_name == "0":
                continue

            csv_row = join_map.get(feat_id)
            if csv_row is None:
                print(f"\n    WARNING: no CSV data for '{geo_name}' in {abbr}")
                continue

            label = csv_row.get("Label", geo_name)
            code = district_code(abbr, label)

            remapped = {state_remap.get(k, k): v for k, v in csv_row.items()}

            # Data row (for CSV)
            row_out: dict = {"district_code": code, "state": abbr,
                             "district_num": normalise_label(label)}
            for col in sorted_data_cols:
                row_out[rename_map[col]] = remapped.get(col, "")
            rows_out.append(row_out)

            # Geometry row (for GeoJSON)
            geo_props: dict = {"district_code": code, "NAMELSAD": code, "state": abbr,
                               "district_num": normalise_label(label)}
            for col in sorted_data_cols:
                geo_props[rename_map[col]] = remapped.get(col, "")

            raw_geom = feat.get("geometry")
            geo_rows.append({"props": geo_props, "geometry": raw_geom})
            matched += 1

        print(f" {matched} matched")

    plan_name = plan_dir.name

    # ---- Build GeoDataFrame, clip, topology-simplify, write ----
    print(f"\n  Clipping {len(geo_rows)} features ...")

    gdf = gpd.GeoDataFrame(
        [r["props"] for r in geo_rows],
        geometry=[shape(r["geometry"]) if r["geometry"] else None for r in geo_rows],
        crs="EPSG:4326",
    )

    # Step 1: fix + clip each geometry individually
    gdf["geometry"] = gdf["geometry"].apply(lambda g: clean_geometry(g, coast))
    before = len(gdf)
    gdf = gdf[gdf["geometry"].notna() & ~gdf["geometry"].is_empty].copy()
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    if before - len(gdf):
        print(f"  Dropped {before - len(gdf)} empty features after clipping")

    # Step 2: snap all coordinates to a shared grid so cross-state borders
    # that are nearly (but not exactly) identical become identical,
    # then run topology-aware simplification so shared edges simplify together
    print(f"  Simplifying {len(gdf)} features with topology preservation ...")
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: set_precision(g, 1e-5) if g is not None else g
    )
    gdf = gdf[gdf["geometry"].notna() & ~gdf["geometry"].is_empty].copy()
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    topo = topojson.Topology(gdf, prequantize=False)
    gdf = topo.toposimplify(SIMPLIFY_TOLERANCE).to_gdf()

    # Fix any geometries invalidated by simplification
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g
    )
    gdf = gdf[gdf["geometry"].notna() & ~gdf["geometry"].is_empty].copy()
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    # Step 3: resolve remaining cross-state border overlaps
    print(f"  Resolving cross-state overlaps ...")
    gdf = resolve_overlaps(gdf)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    print(f"  {len(gdf)} features remaining")

    # Final pass: ensure no GeometryCollections reach the output
    gdf["geometry"] = gdf["geometry"].apply(_to_multipolygon)
    gdf = gdf[gdf["geometry"].notna() & ~gdf["geometry"].is_empty].copy()
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    GEOJSONS_OUT.mkdir(exist_ok=True)
    geojson_out = GEOJSONS_OUT / f"{plan_name}.geojson"
    features_out = []
    for _, row in gdf.iterrows():
        geom = row.geometry.__geo_interface__
        props = {k: v for k, v in row.items() if k != "geometry"}
        features_out.append({"type": "Feature", "geometry": geom, "properties": props})
    with open(geojson_out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features_out}, f,
                  separators=(",", ":"))
    print(f"  Written: {geojson_out}")

    # ---- Write CSV ----
    PROCESSED_OUT.mkdir(exist_ok=True)
    csv_out = PROCESSED_OUT / f"{plan_name}.csv"
    all_cols = META_COLS + [rename_map[c] for c in sorted_data_cols]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"  Written: {csv_out} ({len(rows_out)} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    coast = build_coast_mask()

    for plan_name in PLAN_DIRS:
        plan_dir = RAW_DATA / plan_name
        if not plan_dir.is_dir():
            print(f"Skipping {plan_name} (directory not found)")
            continue
        build_plan(plan_dir, coast)

    print("\nDone.")


if __name__ == "__main__":
    main()
