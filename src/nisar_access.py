import argparse
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path

from pyreqwest.client import SyncClientBuilder
from pyreqwest.exceptions import RequestError
from tqdm import tqdm

from src.asf_sar_search import construct_api_search
from src.nisar_convert_to_tiff import convert_scene_to_geotiff
from src.nisar_dl import download_scene
from utils.geometry import crop_pixels, epsg_crs
from utils.variables import (
    ASF_DEFAULT_PROCESSING_LEVEL,
    EARTHDATA_TOKEN,
    NISAR_GCOV_PIXEL_SIZE_M,
    NISAR_GCOV_SCENE_SIZE_M,
    NISAR_GEOTIFF_PATH,
    NISAR_H5_PATH,
    NISAR_PROCESSING_INFO,
    NISAR_REMOTE_MAX_PIXELS,
    SEARCH_AOIS,
)


def process_scene(
    granule_name: str,
    url: str,
    remote: bool,
    crop_bbox: list[float] | None,
    args: argparse.Namespace,
) -> tuple[list[Path], str | None, str]:
    """
    Read (remote) or download, then convert one scene. Runs in a worker process, so its printed output is
    captured and returned with the GeoTIFFs written and the error (None if it succeeded).
    """
    tifs: list[Path] = []
    output = io.StringIO()
    with redirect_stdout(output):
        try:
            local_h5 = NISAR_H5_PATH / f"{granule_name}.h5"
            if remote and local_h5.exists():
                # Already on disk (e.g. from --keep-h5); reading locally beats going remote
                h5 = local_h5
            elif remote:
                h5 = url
            else:
                with SyncClientBuilder().build() as client:
                    h5 = download_scene(
                        client, granule_name, url
                    )  # skips if already on disk
            for pol in args.polarization or [None]:
                tifs.append(
                    convert_scene_to_geotiff(
                        h5,
                        frequency=args.frequency,
                        polarization=pol,
                        db=args.db,
                        bbox=crop_bbox,
                        aoi_name=args.aoi,
                        out_dir=args.out_dir,
                        overwrite=args.overwrite,
                        dst_crs=args.crs,
                    )
                )
            # Only reached if every polarization converted; deletes any local .h5 used (not a remote read), as before
            if h5 != url and not args.keep_h5:
                h5.unlink()

        # Same failure types the standalone batch loops catch, so one bad scene doesn't stop the run
        except (
            RequestError,
            RuntimeError,
            OSError,
            KeyError,
            ValueError,
            IndexError,
        ) as exc:
            return tifs, str(exc), output.getvalue()
    return tifs, None, output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search, download and convert NISAR scenes for a saved AOI in one run."
    )
    parser.add_argument(
        "--aoi",
        required=True,
        choices=sorted(SEARCH_AOIS),
        help="Named AOI from SEARCH_AOIS in utils/variables.py.",
    )
    parser.add_argument(
        "--product",
        default=ASF_DEFAULT_PROCESSING_LEVEL,
        choices=sorted(NISAR_PROCESSING_INFO),
    )
    parser.add_argument(
        "--sd", "--start", dest="start", required=True, help="Start Date: YYYY-MM-DD"
    )
    parser.add_argument(
        "--ed", "--end", dest="end", required=True, help="End Date: YYYY-MM-DD"
    )
    parser.add_argument(
        "--frequency",
        "--freq",
        type=str.upper,
        choices=["A", "B"],
        default="B",
        help="A: higher resolution, larger files. B: lower resolution, smaller (default: %(default)s).",
    )
    parser.add_argument(
        "--polarization",
        "--pol",
        type=str.upper,
        nargs="+",
        help="One GeoTIFF per polarization, e.g. HH, HV, VV, HHHH, HVHV, VVVV. Defaults per product.",
    )
    parser.add_argument(
        "--db", action="store_true", help="Write intensity in dB (GSLC/GCOV only)."
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop the GeoTIFF to the AOI bbox, reading only that area from ASF instead of downloading the whole .h5; by default will write the whole scene to disk.",
    )
    parser.add_argument(
        "--keep-h5",
        action="store_true",
        help="Keep each local .h5 after converting (default: delete). Bands read remotely are never downloaded, so there is no .h5 to keep.",
    )
    parser.add_argument(
        "--out-dir",
        "--outpath",
        dest="out_dir",
        default=NISAR_GEOTIFF_PATH,
        help=f"Folder to write GeoTIFFs to; created if missing. Defaults to {NISAR_GEOTIFF_PATH}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reconvert even if the GeoTIFF already exists (e.g. after changing an AOI's bbox or buffer_km).",
    )
    parser.add_argument(
        "--crs",
        type=epsg_crs,
        help="Output CRS as an EPSG code, e.g. EPSG:4326 or EPSG:3857 (default: the scene's own, usually UTM).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Scenes processed in parallel (default: 4). Speeds up remote reads near-linearly; whole-file "
        "downloads share bandwidth, so gain little and each needs ~6 GB of disk at once.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    # download_scene doesn't check the token itself, so fail before searching
    if not EARTHDATA_TOKEN:
        raise SystemExit("EARTHDATA_TOKEN is not set; add it to .env")

    bbox_str = SEARCH_AOIS[args.aoi]["bbox"]
    # Converter expects (min_lon, min_lat, max_lon, max_lat) as floats
    crop_bbox = [float(v) for v in bbox_str.split(",")] if args.crop else None

    # Read just the needed band(s) from the remote .h5 when that's small, otherwise download the whole file.
    # Remote reads are many small sequential requests, so their time grows with the pixels read (output size x
    # polarizations), while a whole-file download costs the same however many bands are used.
    polarizations = args.polarization or [None]
    if args.product == "GCOV":
        pixel_size_m = NISAR_GCOV_PIXEL_SIZE_M[args.frequency]
        band_pixels = (
            crop_pixels(crop_bbox, pixel_size_m)
            if args.crop
            else round(NISAR_GCOV_SCENE_SIZE_M / pixel_size_m) ** 2
        )
        n_pixels = band_pixels * len(polarizations)
        remote = n_pixels <= NISAR_REMOTE_MAX_PIXELS
        reason = (
            f"~{band_pixels / 1e6:.1f} Mpx x {len(polarizations)} polarization(s) = {n_pixels / 1e6:.1f} Mpx, "
            f"limit {NISAR_REMOTE_MAX_PIXELS / 1e6:.0f} Mpx"
        )
    else:
        # Only GCOV has been timed; other products read crops remotely and download full scenes
        remote = args.crop
        reason = f"{args.product} {'crop' if args.crop else 'full scene'}"
    print(
        f"{'Reading only the needed band(s) remotely' if remote else 'Downloading each whole .h5'} ({reason})"
    )

    granules = construct_api_search(
        bbox=bbox_str,
        start=args.start,
        end=args.end,
        processing_level=args.product,
        location=args.aoi,
    )
    scenes = granules.unique(
        subset="Granule Name", keep="first", maintain_order=True
    ).select("Granule Name", "URL")
    print(
        f"\n{scenes.height} scenes found for {args.aoi} ({args.product}, {args.start} to {args.end})"
    )

    # Scenes run in parallel worker processes (h5py serializes threads); each prints when it finishes,
    # above a progress bar of scenes done (bar.write keeps the bar at the bottom)
    tifs: list = []
    failures: list[tuple[str, str]] = []
    with (
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        tqdm(
            total=scenes.height,
            desc=f"Scenes ({args.workers} workers)",
            unit="scene",
        ) as bar,
    ):
        futures = {
            pool.submit(
                process_scene, granule_name, url, remote, crop_bbox, args
            ): granule_name
            for granule_name, url in scenes.iter_rows()
        }
        for future in as_completed(futures):
            granule_name = futures[future]
            scene_tifs, error, output = future.result()
            bar.write(f"\n[{bar.n + 1}/{scenes.height}] {granule_name}")
            if output:
                bar.write(output.rstrip("\n"))
            tifs += scene_tifs
            if error:
                bar.write(f"  FAILED: {error}")
                failures.append((granule_name, error))
            bar.update()

    print(f"\nDone. {len(tifs)} GeoTIFFs written, {len(failures)} scenes failed.")
    for name, err in failures:
        print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
