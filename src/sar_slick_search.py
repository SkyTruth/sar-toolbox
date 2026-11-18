import argparse
import json
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from itertools import combinations
from pathlib import Path

import polars as pl
import shapely
from pyreqwest.client import SyncClientBuilder
from pyreqwest.exceptions import RequestError
from shapely.errors import GEOSException
from shapely.geometry import shape

from src.alos2_search import (
    LAT_MARGIN_DEG,
    fetch_item,
    frame_latitude,
    item_footprint,
    list_scene_ids,
    orbit_for_date,
    parse_scene_id,
    scene_urls,
)
from utils.variables import (
    ALOS2_ORBIT_PAD,
    ALOS2_ORBIT_REF_DATE,
    ASF_API_BASE,
    ASF_DATASET,
    ASF_DEFAULT_PROCESSING_LEVEL,
    GRANULE_CSV_PATH,
    NISAR_ARCHIVE_START,
)

ASF_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
RETRIES = 6
SENSORS = {"nisar": "NISAR", "palsar": "PALSAR-2"}
# JAXA's L2.2 archive starts Aug 2014, and orbit numbers are only modelled from then on
PALSAR_ARCHIVE_START = datetime.combine(ALOS2_ORBIT_REF_DATE, datetime.min.time(), UTC)
# Same as src.alos2_search; JAXA's S3 bucket doesn't rate limit like ASF
PALSAR_WORKERS = 16
# Orbits per S3 listing task (about 20 days), so a long date span can be listed in parallel
PALSAR_ORBIT_CHUNK = 300

NISAR_SCHEMA = {
    "nisar_id": pl.String,
    "nisar_time": pl.Datetime("us", "UTC"),
    # Not dt_hours, which an S2 match file already uses for the S2 scene
    "nisar_dt_hours": pl.Float64,
    "nisar_direction": pl.String,
    "nisar_track": pl.Int32,
    "nisar_frame": pl.Int32,
    "nisar_collection": pl.String,
    "nisar_coverage_frac": pl.Float64,
    "nisar_size_mb": pl.Float64,
    "nisar_url": pl.String,
    "nisar_browse_url": pl.String,
    "nisar_footprint_wkt": pl.String,
}

PALSAR_SCHEMA = {
    "palsar_id": pl.String,
    "palsar_time": pl.Datetime("us", "UTC"),
    "palsar_dt_hours": pl.Float64,
    "palsar_mode": pl.String,
    "palsar_direction": pl.String,
    "palsar_path": pl.Int32,
    "palsar_frame": pl.Int32,
    "palsar_polarization": pl.String,
    "palsar_coverage_frac": pl.Float64,
    "palsar_url": pl.String,
    "palsar_hv_url": pl.String,
    "palsar_msk_url": pl.String,
    "palsar_footprint_wkt": pl.String,
}


def load_slicks(path: Path) -> tuple[pl.DataFrame, str]:
    """Every input row, plus the input's GeoParquet metadata so the output stays GeoParquet."""
    geo = pl.read_parquet_metadata(path).get("geo")
    if geo is None:
        raise ValueError(f"{path} has no GeoParquet 'geo' metadata")
    geo_metadata = json.loads(geo)
    # bbox is optional in GeoParquet and would describe the input, not the output
    for column in geo_metadata["columns"].values():
        column.pop("bbox", None)

    # Polars doesn't know the geoarrow.wkb extension type; the WKB bytes are all we need
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Extension type 'geoarrow.wkb'")
        df = pl.read_parquet(path)

    # An S2 match file has one row per slick x S2 scene; the count makes triple coincidences easy to find
    if "s2_id" in df.columns:
        df = df.with_columns(
            n_s2_matches=pl.col("s2_id").is_not_null().sum().over("slick")
        )

    return df, json.dumps(geo_metadata)


def retry(call, *args):
    """call(*args), retried on RuntimeError / RequestError."""
    for attempt in range(RETRIES):
        try:
            return call(*args)
        except (RuntimeError, RequestError) as exc:
            if attempt == RETRIES - 1:
                raise RuntimeError(str(exc)) from exc
            # ASF rate limits (429) without a Retry-After header, so back off hard: 5, 10, 20, 40, 80 s
            time.sleep(5 * 2**attempt)


def query_asf(client, params: dict) -> list[dict]:
    resp = client.get(ASF_API_BASE).query(params).build().send()
    if resp.status != 200:
        raise RuntimeError(f"{resp.status} from ASF")
    return resp.json()["features"]


def search_nisar_slick(
    client, slick: dict, window: timedelta, processing_level: str
) -> list[dict]:
    """NISAR scenes whose footprint overlaps the slick within +/- window of the slick timestamp."""
    geom = shapely.make_valid(shapely.from_wkb(slick["geometry"]))
    slick_time = slick["slick_timestamp"]
    params = {
        "dataset": ASF_DATASET,
        "processingLevel": processing_level,
        # Slicks can have thousands of vertices; the hull keeps the URL short and only adds candidates
        "intersectsWith": shapely.to_wkt(
            shapely.convex_hull(geom), rounding_precision=5
        ),
        "start": (slick_time - window).strftime(ASF_TIME_FORMAT),
        "end": (slick_time + window).strftime(ASF_TIME_FORMAT),
        "output": "geojson",
    }

    rows = []
    for feature in retry(query_asf, client, params):
        props = feature["properties"]
        footprint = shapely.make_valid(shape(feature["geometry"]))
        # Ratio of areas in degrees; fine at slick scale
        coverage = footprint.intersection(geom).area / geom.area
        # Hits on the hull that miss the slick itself
        if coverage == 0:
            continue

        nisar_time = datetime.fromisoformat(props["startTime"])
        h5_bytes = props.get("bytes", {}).get(props["fileName"], {}).get("bytes")
        rows.append(
            {
                "slick": slick["slick"],
                "nisar_id": props["sceneName"],
                "nisar_time": nisar_time,
                "nisar_dt_hours": (nisar_time - slick_time).total_seconds() / 3600,
                "nisar_direction": props["flightDirection"],
                "nisar_track": props["pathNumber"],
                "nisar_frame": props["frameNumber"],
                "nisar_collection": props["collectionName"],
                "nisar_coverage_frac": coverage,
                # Same units as the "Size (MB)" column in ASF's CSV output
                "nisar_size_mb": h5_bytes / 1e6 if h5_bytes else None,
                "nisar_url": props["url"],
                "nisar_browse_url": (props.get("browse") or [None])[0],
                "nisar_footprint_wkt": footprint.wkt,
            }
        )
    return rows


def search_nisar(
    slicks: pl.DataFrame, window: timedelta, processing_level: str, workers: int
) -> tuple[pl.DataFrame, list[tuple]]:
    """One ASF query per slick."""
    matches: list[dict] = []
    failures: list[tuple] = []

    with SyncClientBuilder().build() as client, ThreadPoolExecutor(workers) as pool:

        def search(slick: dict) -> tuple[list[dict], tuple | None]:
            try:
                return search_nisar_slick(client, slick, window, processing_level), None
            # RuntimeError: ASF kept failing (raised in retry)
            # GEOSException: a geometry shapely couldn't intersect
            except (RuntimeError, GEOSException) as exc:
                return [], (slick["slick"], str(exc))

        results = pool.map(search, slicks.iter_rows(named=True))
        for i, (rows, failure) in enumerate(results, start=1):
            matches.extend(rows)
            if failure:
                failures.append(failure)
            if i % 100 == 0 or i == slicks.height:
                print(f"  searched {i}/{slicks.height}")

    schema = {"slick": slicks.schema["slick"], **NISAR_SCHEMA}
    return pl.DataFrame(matches, schema=schema), failures


def palsar_orbit_chunks(
    slicks: pl.DataFrame, window: timedelta
) -> list[tuple[int, int]]:
    """Orbit ranges covering every slick's window, merged, then cut into chunks to list in parallel."""
    ranges = sorted(
        (
            orbit_for_date((t - window).date()) - ALOS2_ORBIT_PAD,
            orbit_for_date((t + window).date()) + ALOS2_ORBIT_PAD,
        )
        for t in slicks["slick_timestamp"]
    )
    merged: list[list[int]] = []
    for lo, hi in ranges:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [
        (start, min(start + PALSAR_ORBIT_CHUNK - 1, hi))
        for lo, hi in merged
        for start in range(lo, hi + 1, PALSAR_ORBIT_CHUNK)
    ]


def search_palsar(
    slicks: pl.DataFrame, window: timedelta
) -> tuple[pl.DataFrame, list[tuple]]:
    """PALSAR-2 L2.2 scenes on JAXA's S3 bucket whose footprint overlaps each slick within +/- window."""
    geoms = {
        slick: shapely.make_valid(shapely.from_wkb(wkb))
        for slick, wkb in slicks.select("slick", "geometry").iter_rows()
    }
    times = dict(slicks.select("slick", "slick_timestamp").iter_rows())
    hours = window.total_seconds() / 3600

    with (
        SyncClientBuilder().build() as client,
        ThreadPoolExecutor(PALSAR_WORKERS) as pool,
    ):
        chunks = palsar_orbit_chunks(slicks, window) if slicks.height else []
        listed = pool.map(lambda chunk: retry(list_scene_ids, client, *chunk), chunks)
        scenes_by_date = defaultdict(list)
        for scene_id in (s for scene_ids in listed for s in scene_ids):
            scene = parse_scene_id(scene_id)
            if scene is not None:
                scene["lat"] = frame_latitude(scene["Frame Number"])
                scenes_by_date[scene["Acquisition Date"]].append(scene)

        # Filter on what the name tells us (date, rough latitude) before fetching any STAC items
        candidates = {}
        for slick, geom in geoms.items():
            _, min_lat, _, max_lat = geom.bounds
            first = (times[slick] - window).date()
            last = (times[slick] + window).date()
            candidates[slick] = [
                scene
                for day in range((last - first).days + 1)
                for scene in scenes_by_date[first + timedelta(days=day)]
                if min_lat - LAT_MARGIN_DEG <= scene["lat"] <= max_lat + LAT_MARGIN_DEG
            ]

        # Slicks close in time and space share candidates, so fetch each STAC item once
        scene_ids = sorted(
            {s["Granule Name"] for scenes in candidates.values() for s in scenes}
        )
        n_listed = sum(len(scenes) for scenes in scenes_by_date.values())
        print(
            f"  listed {n_listed} PALSAR-2 scenes, fetching {len(scene_ids)} candidate STAC items"
        )

        def fetch(scene_id: str) -> tuple[str, tuple | None, str | None]:
            try:
                item = retry(fetch_item, client, scene_id)
                scene_time = datetime.fromisoformat(item["properties"]["datetime"])
                return scene_id, (item_footprint(item), scene_time), None
            # RuntimeError: S3 kept failing (raised in retry); the rest: an unusable STAC item
            except (RuntimeError, GEOSException, KeyError, ValueError) as exc:
                return scene_id, None, f"{scene_id}: {exc}"

        items, item_errors = {}, {}
        for i, (scene_id, item, error) in enumerate(
            pool.map(fetch, scene_ids), start=1
        ):
            if error:
                item_errors[scene_id] = error
            else:
                items[scene_id] = item
            if i % 2000 == 0 or i == len(scene_ids):
                print(f"  fetched {i}/{len(scene_ids)}")

    matches: list[dict] = []
    failures: list[tuple] = []
    for slick, scenes in candidates.items():
        errors = [
            item_errors[s["Granule Name"]]
            for s in scenes
            if s["Granule Name"] in item_errors
        ]
        if errors:
            failures.append((slick, errors[0]))
            continue

        geom = geoms[slick]
        try:
            for scene in scenes:
                footprint, scene_time = items[scene["Granule Name"]]
                dt_hours = (scene_time - times[slick]).total_seconds() / 3600
                if abs(dt_hours) > hours:
                    continue
                coverage = footprint.intersection(geom).area / geom.area
                if coverage == 0:
                    continue

                urls = scene_urls(scene)
                matches.append(
                    {
                        "slick": slick,
                        "palsar_id": scene["Granule Name"],
                        "palsar_time": scene_time,
                        "palsar_dt_hours": dt_hours,
                        "palsar_mode": scene["Beam Mode"],
                        "palsar_direction": scene["Ascending or Descending?"],
                        "palsar_path": scene["Path Number"],
                        "palsar_frame": scene["Frame Number"],
                        "palsar_polarization": scene["Polarization"],
                        "palsar_coverage_frac": coverage,
                        "palsar_url": urls["URL"],
                        "palsar_hv_url": urls["HV URL"],
                        "palsar_msk_url": urls["MSK URL"],
                        "palsar_footprint_wkt": footprint.wkt,
                    }
                )
        except GEOSException as exc:
            failures.append((slick, str(exc)))

    schema = {"slick": slicks.schema["slick"], **PALSAR_SCHEMA}
    return pl.DataFrame(matches, schema=schema), failures


def attach(
    result: pl.DataFrame,
    matches: pl.DataFrame,
    failures: list[tuple],
    in_archive: pl.Expr,
    sensor: str,
) -> pl.DataFrame:
    """Add <sensor>_search_status, and repeat each row once per matched scene of its slick (closest in time first)."""
    failed_ids = pl.Series(
        [slick_id for slick_id, _ in failures], dtype=result.schema["slick"]
    )
    status = (
        pl.when(~in_archive)
        .then(pl.lit(f"before_{sensor}"))
        .when(pl.col("slick").is_in(failed_ids.implode()))
        .then(pl.lit("failed"))
        .otherwise(pl.lit("searched"))
    )
    return result.with_columns(status.alias(f"{sensor}_search_status")).join(
        matches.sort("slick", pl.col(f"{sensor}_dt_hours").abs()),
        on="slick",
        how="left",
        maintain_order="left_right",
    )


def print_summary(
    result: pl.DataFrame,
    failures: dict[str, list[tuple]],
    hours: float,
    processing_level: str,
) -> None:
    slicks = result.unique("slick", keep="first")
    print(f"\n{slicks.height} slicks in file")

    with_scene = {}
    if "n_s2_matches" in result.columns:
        with_scene["S2"] = set(slicks.filter(pl.col("n_s2_matches") > 0)["slick"])
    for sensor in failures:
        label = SENSORS[sensor]
        status = slicks[f"{sensor}_search_status"]
        # An S2 match file repeats each slick-scene pair once per S2 scene
        pairs = result.filter(pl.col(f"{sensor}_id").is_not_null()).unique(
            ["slick", f"{sensor}_id"]
        )
        with_scene[label] = set(pairs["slick"])
        product = f" {processing_level}" if sensor == "nisar" else " L2.2"
        print(
            f"{label}{product}: {(status == f'before_{sensor}').sum()} slicks before its data, "
            f"{(status == 'searched').sum()} searched, {(status == 'failed').sum()} failed; "
            f"{len(with_scene[label])} have a scene within {hours:g} h "
            f"({pairs.height} slick-scene pairs, {pairs[f'{sensor}_id'].n_unique()} unique scenes)"
        )

    for n in range(2, len(with_scene) + 1):
        for combo in combinations(with_scene, n):
            both = set.intersection(*(with_scene[label] for label in combo))
            print(f"  {' + '.join(combo)}: {len(both)} slicks")

    if "hitl_short_name" in result.columns:
        counts = {"slicks": pl.col("slick").n_unique()}
        if "n_s2_matches" in result.columns:
            counts["with_s2"] = (
                pl.col("slick").filter(pl.col("n_s2_matches") > 0).n_unique()
            )
        for sensor in failures:
            counts[f"with_{sensor}"] = (
                pl.col("slick").filter(pl.col(f"{sensor}_id").is_not_null()).n_unique()
            )
        print(
            result.group_by("hitl_short_name")
            .agg(**counts)
            .sort("slicks", descending=True)
        )

    for sensor in failures:
        columns = [
            c
            for c in [
                "slick",
                "slick_timestamp",
                "hitl_short_name",
                "n_s2_matches",
                f"{sensor}_id",
                f"{sensor}_dt_hours",
                f"{sensor}_coverage_frac",
            ]
            if c in result.columns
        ]
        matched = (
            result.filter(pl.col(f"{sensor}_id").is_not_null())
            .unique(["slick", f"{sensor}_id"], keep="first", maintain_order=True)
            .select(columns)
        )
        print(f"\n{SENSORS[sensor]} matches")
        with pl.Config(tbl_rows=50, fmt_str_lengths=100, tbl_width_chars=250):
            print(matched)

    for sensor, sensor_failures in failures.items():
        for slick_id, err in sensor_failures:
            print(f"  {SENSORS[sensor]} FAILED slick {slick_id}: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find NISAR and/or PALSAR-2 scenes collected within a time window of each slick in a slick GeoParquet."
    )
    parser.add_argument(
        "--slicks",
        type=Path,
        required=True,
        help="GeoParquet of slicks with 'slick', 'slick_timestamp' (UTC) and 'geometry' columns, e.g. an S2 match file.",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        choices=list(SENSORS),
        default=list(SENSORS),
        help="SAR missions to search: nisar (ASF) and/or palsar (PALSAR-2 L2.2 on JAXA's S3 bucket). Default both.",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24,
        help="Search +/- this many hours around each slick timestamp (default 24).",
    )
    parser.add_argument(
        "--processing-level",
        default=ASF_DEFAULT_PROCESSING_LEVEL,
        choices=["GSLC", "GOFF", "GCOV", "GUNW", "SME2"],
        help=f"NISAR product type ({ASF_DEFAULT_PROCESSING_LEVEL} default).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent ASF requests for NISAR (default 8).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GeoParquet path. Defaults to data/nisar_granule_files/; a CSV of matched granules per sensor is written next to it.",
    )
    args = parser.parse_args()

    request_date = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    window = timedelta(hours=args.window_hours)
    out = args.out or (
        GRANULE_CSV_PATH
        / f"slick_sar_matches_{args.window_hours:g}h_requestMade{request_date}.parquet"
    )

    rows, geo_metadata = load_slicks(args.slicks)
    slicks = rows.unique("slick", keep="first", maintain_order=True)
    # Every input row is kept, in input order, once per matched scene of its slick; rows whose
    # slick has no scene appear once with null columns for that sensor
    result = rows
    failures: dict[str, list[tuple]] = {}

    if "nisar" in args.sensors:
        # No point asking ASF about slicks whose window closes before NISAR data exists
        in_archive = pl.col("slick_timestamp") + window >= NISAR_ARCHIVE_START
        searchable = slicks.filter(in_archive)
        print(
            f"Searching ASF for NISAR {args.processing_level} scenes within {args.window_hours:g} h "
            f"of {searchable.height} slicks"
        )
        matches, failures["nisar"] = search_nisar(
            searchable, window, args.processing_level, args.workers
        )
        result = attach(result, matches, failures["nisar"], in_archive, "nisar")

    if "palsar" in args.sensors:
        in_archive = pl.col("slick_timestamp") + window >= PALSAR_ARCHIVE_START
        searchable = slicks.filter(in_archive)
        print(
            f"Searching JAXA's S3 bucket for PALSAR-2 L2.2 scenes within {args.window_hours:g} h "
            f"of {searchable.height} slicks"
        )
        matches, failures["palsar"] = search_palsar(searchable, window)
        result = attach(result, matches, failures["palsar"], in_archive, "palsar")

    out.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out, metadata={"geo": geo_metadata})
    written = [out]

    if "nisar" in args.sensors:
        # Same column names as src.asf_sar_search, so src.nisar_dl can download straight from it
        nisar_csv = out.with_name(f"{out.stem}_nisar_granules.csv")
        result.filter(pl.col("nisar_id").is_not_null()).unique(
            "nisar_id", keep="first", maintain_order=True
        ).select(
            pl.col("nisar_id").alias("Granule Name"),
            pl.col("nisar_time").alias("Start Time"),
            pl.col("nisar_direction").alias("Ascending or Descending?"),
            pl.col("nisar_track").alias("Path Number"),
            pl.col("nisar_frame").alias("Frame Number"),
            pl.col("nisar_url").alias("URL"),
            pl.col("nisar_size_mb").alias("Size (MB)"),
        ).write_csv(nisar_csv)
        written.append(nisar_csv)

    if "palsar" in args.sensors:
        # Same column names as src.alos2_search, so src.alos2_dl can download straight from it
        palsar_csv = out.with_name(f"{out.stem}_palsar_granules.csv")
        result.filter(pl.col("palsar_id").is_not_null()).unique(
            "palsar_id", keep="first", maintain_order=True
        ).select(
            pl.col("palsar_id").alias("Granule Name"),
            pl.col("palsar_time").alias("Acquisition Time"),
            pl.col("palsar_mode").alias("Beam Mode"),
            pl.col("palsar_path").alias("Path Number"),
            pl.col("palsar_frame").alias("Frame Number"),
            pl.col("palsar_direction").alias("Ascending or Descending?"),
            pl.col("palsar_polarization").alias("Polarization"),
            pl.col("palsar_url").alias("URL"),
            pl.col("palsar_hv_url").alias("HV URL"),
            pl.col("palsar_msk_url").alias("MSK URL"),
        ).write_csv(palsar_csv)
        written.append(palsar_csv)

    print_summary(result, failures, args.window_hours, args.processing_level)
    n_failed = sum(len(f) for f in failures.values())
    if n_failed:
        print(
            f"\n{n_failed} slick searches failed (<sensor>_search_status 'failed'); re-run to retry them."
        )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
