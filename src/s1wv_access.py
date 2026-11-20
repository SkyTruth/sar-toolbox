import argparse
import io
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

import polars as pl
from pyreqwest.exceptions import RequestError
from tqdm import tqdm

from src.s1wv_convert_to_tiff import convert_vignette_to_geotiff, output_path
from src.s1wv_dl import CdseAuth, cdse_client, download_vignette, resolve_vignette
from src.s1wv_search import construct_cdse_search
from utils.geometry import epsg_crs
from utils.variables import CDSE_MAX_DOWNLOADS, S1WV_GEOTIFF_PATH, SEARCH_AOIS


def process_vignette(
    row: dict,
    tokens: dict,
    args: argparse.Namespace,
) -> tuple[Path | None, str | None, str]:
    """
    Download (if needed), then convert one vignette. Runs in a worker process, so its printed output is
    captured and returned with the GeoTIFF written and the error (None if it succeeded).
    """
    output = io.StringIO()
    # Workers reuse the main process's login and only ever refresh it, so CDSE sees one login per run
    auth = CdseAuth.from_state(tokens)
    with redirect_stdout(output), cdse_client() as client:
        try:
            # Already converted: skip the ~100 MB download (the SLC is usually deleted after converting)
            tiff_name = resolve_vignette(client, auth, row)
            out_path = output_path(tiff_name, args.db, args.out_dir, args.crs)
            if out_path.exists() and not args.overwrite:
                print(f"    Skipping {out_path.name} (already exists)")
                return out_path, None, output.getvalue()

            tiff = download_vignette(client, auth, row)  # skips files already on disk
            tif = convert_vignette_to_geotiff(
                tiff,
                remove_slc=not args.keep_slc,
                db=args.db,
                out_dir=args.out_dir,
                overwrite=args.overwrite,
                dst_crs=args.crs,
            )

        # Same failure types the standalone batch loops catch, so one bad vignette doesn't stop the run
        except (RequestError, RuntimeError, OSError, ValueError, ET.ParseError) as exc:
            return None, str(exc), output.getvalue()
    return tif, None, output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search, download and convert Sentinel-1 WV vignettes for a saved AOI in one run."
    )
    parser.add_argument(
        "--aoi",
        required=True,
        choices=sorted(SEARCH_AOIS),
        help="Named AOI from SEARCH_AOIS in utils/variables.py.",
    )
    parser.add_argument(
        "--sd", "--start", dest="start", required=True, help="Start Date: YYYY-MM-DD"
    )
    parser.add_argument(
        "--ed",
        "--end",
        dest="end",
        required=True,
        help="End Date: YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--db", action="store_true", help="Write sigma0 in dB (10*log10)."
    )
    parser.add_argument(
        "--keep-slc",
        action="store_true",
        help="Keep each downloaded measurement .tiff (~105 MB) after converting (default: delete). "
        "The small annotation/calibration XMLs are always kept.",
    )
    parser.add_argument(
        "--out-dir",
        "--outpath",
        dest="out_dir",
        default=S1WV_GEOTIFF_PATH,
        help=f"Folder to write GeoTIFFs to; created if missing. Defaults to {S1WV_GEOTIFF_PATH}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reconvert even if the GeoTIFF already exists (downloads the measurement file again if deleted).",
    )
    parser.add_argument(
        "--crs",
        type=epsg_crs,
        help="Output CRS as an EPSG code, e.g. EPSG:4326 or EPSG:3857 (default: the vignette's UTM zone).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=CDSE_MAX_DOWNLOADS,
        help=f"Vignettes processed in parallel (default: {CDSE_MAX_DOWNLOADS}). CDSE allows "
        f"{CDSE_MAX_DOWNLOADS} concurrent downloads per user, so more doesn't help.",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= CDSE_MAX_DOWNLOADS:
        parser.error(
            f"--workers must be between 1 and {CDSE_MAX_DOWNLOADS} (CDSE's per-user download limit)"
        )

    granules = construct_cdse_search(
        bbox=SEARCH_AOIS[args.aoi]["bbox"],
        start=args.start,
        end=args.end,
        location=args.aoi,
    )
    vignettes = granules.unique(
        subset=["Granule Name", "Vignette"], keep="first", maintain_order=True
    )
    offline = vignettes.filter(~pl.col("Online"))
    vignettes = vignettes.filter(pl.col("Online"))
    print(
        f"\n{vignettes.height} vignettes found for {args.aoi} (S1 WV SLC, {args.start} to {args.end})"
        + (
            f"; {offline.height} more in offline products are skipped"
            if offline.height
            else ""
        )
    )
    if vignettes.is_empty():
        return

    # Log in once here (fails fast on bad credentials) and hand the tokens to the workers
    tokens = CdseAuth().state()

    # Vignettes run in parallel worker processes; each prints when it finishes,
    # above a progress bar of vignettes done (bar.write keeps the bar at the bottom)
    tifs: list[Path] = []
    failures: list[tuple[str, str]] = []
    with (
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        tqdm(
            total=vignettes.height,
            desc=f"Vignettes ({args.workers} workers)",
            unit="vignette",
        ) as bar,
    ):
        futures = {
            pool.submit(
                process_vignette, row, tokens, args
            ): f"{row['Granule Name']} vignette {row['Vignette']}"
            for row in vignettes.iter_rows(named=True)
        }
        for future in as_completed(futures):
            label = futures[future]
            tif, error, output = future.result()
            bar.write(f"\n[{bar.n + 1}/{vignettes.height}] {label}")
            if output:
                bar.write(output.rstrip("\n"))
            if tif:
                tifs.append(tif)
            if error:
                bar.write(f"  FAILED: {error}")
                failures.append((label, error))
            bar.update()

    print(f"\nDone. {len(tifs)} GeoTIFFs written, {len(failures)} vignettes failed.")
    for label, err in failures:
        print(f"  {label}: {err}")


if __name__ == "__main__":
    main()
