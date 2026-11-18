import argparse
import math
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime

import polars as pl
from pyreqwest.client import SyncClientBuilder
from shapely.affinity import translate
from shapely.geometry import Polygon, box, shape

from utils.variables import (
    ALOS2_GRANULE_CSV_PATH,
    ALOS2_HH_ONLY_MODES,
    ALOS2_MODES,
    ALOS2_ORBIT_PAD,
    ALOS2_ORBIT_REF,
    ALOS2_ORBIT_REF_DATE,
    ALOS2_ORBITS_PER_DAY,
    JAXA_L22_BUCKET_URL,
    JAXA_L22_PREFIX,
    REQUEST_DATE,
    SEARCH_AOIS,
)

# e.g. ALOS2585563700-250327_WBDR2.2GUD: orbit, frame, date, beam mode, look (R/L), direction (A/D)
SCENE_RE = re.compile(r"ALOS2(\d{5})(\d{4})-(\d{6})_(\w{3})([RL])2\.2GU([AD])")
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
INCLINATION = math.radians(97.9)
# Scenes are ~350 km tall and the frame -> latitude estimate is good to ~1.3 deg
LAT_MARGIN_DEG = 3.0

SCHEMA = {
    "Granule Name": pl.String,
    "Platform": pl.String,
    "Beam Mode": pl.String,
    "Path Number": pl.Int32,
    "Frame Number": pl.Int32,
    "Orbit": pl.Int32,
    "Acquisition Date": pl.Date,
    "Ascending or Descending?": pl.String,
    "Look Direction": pl.String,
    "Polarization": pl.String,
    "URL": pl.String,
    "HV URL": pl.String,
    "MSK URL": pl.String,
    "Footprint WKT": pl.String,
}


def orbit_for_date(d: date) -> int:
    return round(
        ALOS2_ORBIT_REF + ALOS2_ORBITS_PER_DAY * (d - ALOS2_ORBIT_REF_DATE).days
    )


def frame_latitude(frame: int) -> float:
    """Approximate scene-centre latitude; frame / 20 is the argument of latitude in degrees."""
    return math.degrees(
        math.asin(math.sin(INCLINATION) * math.sin(math.radians(frame * 0.05)))
    )


def parse_scene_id(scene_id: str) -> dict | None:
    m = SCENE_RE.fullmatch(scene_id)
    if m is None:
        return None
    orbit, frame, yymmdd, mode, look, direction = m.groups()
    orbit = int(orbit)
    return {
        "Granule Name": scene_id,
        "Platform": "ALOS-2",
        "Beam Mode": mode,
        "Path Number": (14 * orbit + 23) % 207 + 1,
        "Frame Number": int(frame),
        "Orbit": orbit,
        "Acquisition Date": date(
            2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:])
        ),
        "Ascending or Descending?": "ASCENDING" if direction == "A" else "DESCENDING",
        "Look Direction": "RIGHT" if look == "R" else "LEFT",
        "Polarization": "HH" if mode in ALOS2_HH_ONLY_MODES else "HH+HV",
    }


def list_scene_ids(client, orbit_lo: int, orbit_hi: int) -> list[str]:
    """List scene IDs between two orbit numbers (keys are sorted by orbit, so stop once past orbit_hi)."""
    scene_ids: list[str] = []
    query = {
        "list-type": "2",
        "prefix": JAXA_L22_PREFIX,
        "start-after": f"{JAXA_L22_PREFIX}ALOS2{orbit_lo:05d}",
    }
    while True:
        resp = client.get(JAXA_L22_BUCKET_URL).query(query).build().send()
        if resp.status != 200:
            raise RuntimeError(f"{resp.status} listing {JAXA_L22_PREFIX}")
        root = ET.fromstring(resp.bytes())

        for key in root.iterfind("s3:Contents/s3:Key", S3_NS):
            name = key.text.removeprefix(JAXA_L22_PREFIX)
            if int(name[5:10]) > orbit_hi:
                return scene_ids
            # One .json (STAC item) per scene; the other six files share its stem
            if name.endswith(".json"):
                scene_ids.append(name.removesuffix(".json"))

        token = root.find("s3:NextContinuationToken", S3_NS)
        if token is None:
            return scene_ids
        query["continuation-token"] = token.text


def fetch_item(client, scene_id: str) -> dict:
    """Scene's STAC item. Its geometry and datetime (scene centre, matches summary.xml) are usable; the rest has known defects."""
    resp = (
        client.get(f"{JAXA_L22_BUCKET_URL}/{JAXA_L22_PREFIX}{scene_id}.json")
        .build()
        .send()
    )
    if resp.status != 200:
        raise RuntimeError(f"{resp.status} for {scene_id}.json")
    return resp.json()


def item_footprint(item: dict):
    """Scene footprint from its STAC item's geometry."""
    geom = shape(item["geometry"])

    # Footprints crossing the antimeridian come back ~360 deg wide; unwrap and keep a copy on each side
    min_x, _, max_x, _ = geom.bounds
    if max_x - min_x > 180:
        unwrapped = Polygon(
            [(x + 360 if x < 0 else x, y) for x, y in geom.exterior.coords]
        )
        geom = unwrapped.union(translate(unwrapped, xoff=-360))
    return geom


def fetch_footprint(client, scene_id: str):
    """Scene footprint from its STAC item."""
    return item_footprint(fetch_item(client, scene_id))


def scene_urls(scene: dict) -> dict:
    """Download URLs for a parsed scene; HH-only modes have no HV file."""
    base = f"{JAXA_L22_BUCKET_URL}/{JAXA_L22_PREFIX}{scene['Granule Name']}"
    return {
        "URL": f"{base}_HH_SLP.tif",
        "HV URL": f"{base}_HV_SLP.tif" if scene["Polarization"] == "HH+HV" else None,
        "MSK URL": f"{base}_MSK.tif",
    }


def construct_s3_search(
    bbox: str,
    start: str,
    end: str,
    modes: list[str] | None = None,
    request_info: str = REQUEST_DATE,
    location: str | None = None,
    workers: int = 16,
) -> pl.DataFrame:
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox.split(","))
    aoi = box(min_lon, min_lat, max_lon, max_lat)
    start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)

    with SyncClientBuilder().build() as client:
        scene_ids = list_scene_ids(
            client,
            orbit_for_date(start_date) - ALOS2_ORBIT_PAD,
            orbit_for_date(end_date) + ALOS2_ORBIT_PAD,
        )

        # Filter on what the name tells us before fetching any footprints
        candidates = []
        for scene_id in scene_ids:
            scene = parse_scene_id(scene_id)
            if scene is None or not start_date <= scene["Acquisition Date"] <= end_date:
                continue
            if modes and scene["Beam Mode"] not in modes:
                continue
            lat = frame_latitude(scene["Frame Number"])
            if min_lat - LAT_MARGIN_DEG <= lat <= max_lat + LAT_MARGIN_DEG:
                candidates.append(scene)

        with ThreadPoolExecutor(workers) as pool:
            footprints = list(
                pool.map(
                    lambda s: fetch_footprint(client, s["Granule Name"]), candidates
                )
            )

    rows = [
        scene | scene_urls(scene) | {"Footprint WKT": footprint.wkt}
        for scene, footprint in zip(candidates, footprints)
        if footprint.intersects(aoi)
    ]
    df = pl.DataFrame(rows, schema=SCHEMA).sort("Acquisition Date")

    location_part = f"_{location}" if location is not None else ""
    df.write_csv(
        ALOS2_GRANULE_CSV_PATH
        / f"ALOS-2_L2.2{location_part}_granules_{start}-{end}_requestMade{request_info}.csv"
    )
    if df.is_empty():
        print(
            "No PALSAR-2 L2.2 scenes intersect this AOI in the date range. JAXA's free ScanSAR "
            "archive mostly covers tropical land, coasts and polar regions; open ocean and much of "
            "the US, Europe and the Middle East are rarely or never observed."
        )
    print(
        df.select(
            [
                "Granule Name",
                "Acquisition Date",
                "Beam Mode",
                "Polarization",
                "URL",
            ]
        )
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List JAXA's public S3 archive to return a csv file of PALSAR-2 ScanSAR L2.2 scenes."
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=ALOS2_MODES,
        help="Only these beam modes. WBD/WWD/VBD are HH+HV; WBS/WWS are HH only.",
    )
    aoi_group = parser.add_mutually_exclusive_group(required=True)
    aoi_group.add_argument(
        "--bbox",
        help=("A bounding box defined as: 'min_lon,min_lat,max_lon,max_lat'"),
    )
    aoi_group.add_argument(
        "--aoi",
        choices=sorted(SEARCH_AOIS),
        help=("Named AOI from SEARCH_AOIS in utils/variables.py; its bbox is used."),
    )
    parser.add_argument(
        "--sd",
        "--start",
        "--s_date",
        dest="start",
        required=True,
        help=("Start Date: YYYY-MM-DD"),
    )
    parser.add_argument(
        "--ed",
        "--end",
        "--e_date",
        dest="end",
        required=True,
        help=("End Date: YYYY-MM-DD"),
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Optional location label added to the output filename. Defaults to the --aoi name if given.",
    )

    args = parser.parse_args()
    bbox = SEARCH_AOIS[args.aoi]["bbox"] if args.aoi else args.bbox
    location = args.location or args.aoi

    request_date = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")

    construct_s3_search(
        bbox=bbox,
        start=args.start,
        end=args.end,
        modes=args.mode,
        request_info=request_date,
        location=location,
    )


if __name__ == "__main__":
    main()
