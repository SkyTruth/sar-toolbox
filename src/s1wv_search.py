import argparse
from datetime import UTC, date, datetime, timedelta

import polars as pl
import shapely
from pyreqwest.client import SyncClientBuilder
from shapely.affinity import translate
from shapely.geometry import Polygon, box, shape

from utils.variables import (
    CDSE_CATALOGUE_URL,
    REQUEST_DATE,
    S1WV_GRANULE_CSV_PATH,
    S1WV_PRODUCT_TYPE,
    SEARCH_AOIS,
)

# One row per vignette: a WV product holds ~10-180 vignettes (~20 x 20 km, ~100 km apart along the orbit),
# and only those that intersect the AOI are listed
SCHEMA = {
    "Granule Name": pl.String,
    "Vignette": pl.Int32,
    "Product Id": pl.String,
    "Platform": pl.String,
    "Product Start": pl.Datetime("us", "UTC"),
    "Polarization": pl.String,
    "Ascending or Descending?": pl.String,
    "Relative Orbit": pl.Int32,
    "Online": pl.Boolean,
    "Center Lon": pl.Float64,
    "Center Lat": pl.Float64,
    "Footprint WKT": pl.String,
}
PAGE_SIZE = 1000  # CDSE's maximum $top


def odata_filter(aoi: Polygon, start: date, end: date) -> str:
    """WV SLC products intersecting the AOI and starting between start and end (both inclusive, UTC)."""
    end_exclusive = end + timedelta(days=1)
    return (
        "Collection/Name eq 'SENTINEL-1'"
        " and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{S1WV_PRODUCT_TYPE}')"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{aoi.wkt}')"
        f" and ContentDate/Start ge {start.isoformat()}T00:00:00.000Z"
        f" and ContentDate/Start lt {end_exclusive.isoformat()}T00:00:00.000Z"
    )


def search_products(client, aoi: Polygon, start: date, end: date) -> list[dict]:
    """Every matching product (with its attributes), following CDSE's paging."""
    products: list[dict] = []
    query = {
        "$filter": odata_filter(aoi, start, end),
        "$expand": "Attributes",
        "$orderby": "ContentDate/Start asc",
        "$top": str(PAGE_SIZE),
    }
    skip = 0
    while True:
        resp = (
            client.get(f"{CDSE_CATALOGUE_URL}/Products")
            .query(query | {"$skip": str(skip)})
            .build()
            .send()
        )
        if resp.status != 200:
            raise RuntimeError(
                f"{resp.status} from the CDSE catalogue: {resp.text()[:300]}"
            )
        page = resp.json()["value"]
        products += page
        if len(page) < PAGE_SIZE:
            return products
        skip += PAGE_SIZE


def dedupe_reprocessed(products: list[dict]) -> list[dict]:
    """Reprocessed copies differ only in the final 4-character product ID; keep the most recently published."""
    latest: dict[str, dict] = {}
    for p in products:
        key = p["Name"].removesuffix(".SAFE")[:-5]
        if key not in latest or p["PublicationDate"] > latest[key]["PublicationDate"]:
            latest[key] = p
    return list(latest.values())


def unwrap_antimeridian(geom):
    """A vignette crossing the antimeridian comes back ~360 deg wide; unwrap and keep a copy on each side."""
    min_x, _, max_x, _ = geom.bounds
    if max_x - min_x <= 180:
        return geom
    unwrapped = Polygon([(x + 360 if x < 0 else x, y) for x, y in geom.exterior.coords])
    return unwrapped.union(translate(unwrapped, xoff=-360))


def vignette_rows(product: dict, aoi: Polygon) -> list[dict]:
    """The product's vignettes that intersect the AOI. Footprint parts are in acquisition order, so a part's
    index is the vignette number s1wv_dl uses to find its measurement file (and checks against the annotation)."""
    attrs = {a["Name"]: a["Value"] for a in product["Attributes"]}
    name = product["Name"].removesuffix(".SAFE")
    parts = shapely.get_parts(shape(product["GeoFootprint"]))
    rows = []
    for i, part in enumerate(parts):
        footprint = unwrap_antimeridian(part)
        if not footprint.intersects(aoi):
            continue
        centre = part.centroid
        rows.append(
            {
                "Granule Name": name,
                "Vignette": i,
                "Product Id": product["Id"],
                "Platform": f"Sentinel-1{attrs.get('platformSerialIdentifier', '')}",
                "Product Start": datetime.fromisoformat(
                    product["ContentDate"]["Start"]
                ),
                "Polarization": attrs.get("polarisationChannels"),
                "Ascending or Descending?": attrs.get("orbitDirection"),
                "Relative Orbit": attrs.get("relativeOrbitNumber"),
                "Online": product["Online"],
                "Center Lon": centre.x,
                "Center Lat": centre.y,
                "Footprint WKT": footprint.wkt,
            }
        )
    return rows


def construct_cdse_search(
    bbox: str,
    start: str,
    end: str,
    request_info: str = REQUEST_DATE,
    location: str | None = None,
) -> pl.DataFrame:
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox.split(","))
    aoi = box(min_lon, min_lat, max_lon, max_lat)

    with SyncClientBuilder().timeout(timedelta(seconds=120)).build() as client:
        products = search_products(
            client, aoi, date.fromisoformat(start), date.fromisoformat(end)
        )
    products = dedupe_reprocessed(products)

    rows = [row for p in products for row in vignette_rows(p, aoi)]
    df = pl.DataFrame(rows, schema=SCHEMA).sort("Product Start", "Vignette")

    location_part = f"_{location}" if location is not None else ""
    S1WV_GRANULE_CSV_PATH.mkdir(parents=True, exist_ok=True)
    df.write_csv(
        S1WV_GRANULE_CSV_PATH
        / f"S1_{S1WV_PRODUCT_TYPE.rstrip('_')}{location_part}_vignettes_{start}-{end}_requestMade{request_info}.csv"
    )
    if df.is_empty():
        print(
            f"No WV vignettes intersect this AOI in the date range ({len(products)} WV products pass over its "
            "bbox, but their vignettes miss it). WV only images the open ocean, in ~20 km vignettes ~100 km "
            "apart, so a small AOI is often missed; a larger AOI or a longer date range helps."
        )
    offline = df.filter(~pl.col("Online")).height
    if offline:
        print(
            f"{offline} vignettes are in offline products and can't be downloaded until CDSE restores them"
        )
    print(
        df.select(
            "Granule Name",
            "Vignette",
            "Product Start",
            "Polarization",
            "Center Lon",
            "Center Lat",
            "Online",
        )
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the CDSE catalogue for a csv file of Sentinel-1 Wave mode (WV) SLC vignettes."
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
        help=("End Date: YYYY-MM-DD (inclusive)"),
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

    construct_cdse_search(
        bbox=bbox,
        start=args.start,
        end=args.end,
        request_info=request_date,
        location=location,
    )


if __name__ == "__main__":
    main()
