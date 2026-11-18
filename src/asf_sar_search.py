import argparse
from datetime import UTC, datetime

import polars as pl

from utils.variables import (
    ASF_API_BASE,
    ASF_DATA_MATURITY,
    ASF_DATASET,
    ASF_DEFAULT_PROCESSING_LEVEL,
    ASF_OUT,
    GRANULE_CSV_PATH,
    REQUEST_DATE,
    SEARCH_AOIS,
)


def construct_api_search(
    bbox: str,
    start: str,
    end: str,
    url_base: str = ASF_API_BASE,
    dataset: str = ASF_DATASET,
    processing_level: str = ASF_DEFAULT_PROCESSING_LEVEL,
    request_info: str = REQUEST_DATE,
    data_maturity: str = ASF_DATA_MATURITY,
    out: str = ASF_OUT,
    location: str | None = None,
) -> pl.DataFrame:
    api_query_url = f"{url_base}?dataset={dataset}&processingLevel={processing_level}&dataMaturity={data_maturity}&bbox={bbox}&start={start}&end={end}&output={out}"
    location_part = f"_{location}" if location is not None else ""
    df = pl.read_csv(api_query_url)
    df.write_csv(
        GRANULE_CSV_PATH
        / f"{dataset}_{processing_level}{location_part}_granules_{start}-{end}_requestMade{request_info}.{out}"
    )
    print(
        df.select(
            [
                "Granule Name",
                # "Platform",
                "Processing Date",
                "Processing Level",
                "URL",
                "Size (MB)",
            ]
        )
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query ASF API to return csv file of scene granules."
    )
    parser.add_argument(
        "--dataset",
        default=ASF_DATASET,
        help=("ASF Dataset Name. Currently anything other than NISAR will break."),
    )
    parser.add_argument(
        "--processing-level",
        default=ASF_DEFAULT_PROCESSING_LEVEL,
        choices=["GSLC", "GOFF", "GCOV", "GUNW", "SME2"],
        help=(
            f"NISAR product type ({ASF_DEFAULT_PROCESSING_LEVEL} default). GSLC = Geocoded Single Look Complex, level 2. GOFF = Geocoded Pixel Offsets, level 2. GCOV = Geocoded Polarimetric Covariance Matrix, level 2. GUNW = Geocoded Unwrapped Interferogram, level 2. SME2 = Soil Moisture, level 3."
        ),
    )
    parser.add_argument(
        "--data_maturity",
        default=ASF_DATA_MATURITY,
        help=(
            "Calibration maturity; PROVISIONAL is only valid option right now. VALIDATED has no granules yet."
        ),
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
        "--out",
        default=ASF_OUT,
        help=(
            "ASF Output Format, how to preseve granule data. Anything other than csv will probably break"
        ),
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

    construct_api_search(
        dataset=args.dataset,
        processing_level=args.processing_level,
        request_info=request_date,
        data_maturity=args.data_maturity,
        bbox=bbox,
        start=args.start,
        end=args.end,
        out=args.out,
        location=location,
    )


if __name__ == "__main__":
    main()
