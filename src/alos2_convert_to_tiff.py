import argparse
import re
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform

from src.alos2_destripe import scene_stripes, window_stripes
from src.nisar_convert_to_tiff import is_url, reproject_tif, to_output
from utils.geometry import epsg_crs
from utils.variables import (
    ALOS2_ATTRIBUTION,
    ALOS2_CAL_DB,
    ALOS2_GEOTIFF_PATH,
    ALOS2_MASK_DROP,
    ALOS2_TIF_PATH,
    JAXA_L22_BUCKET_URL,
    JAXA_L22_PREFIX,
)

# e.g. ALOS2585563700-250327_WBDR2.2GUD_HH_SLP -> scene ID, polarization
L22_FILE_RE = re.compile(r"(ALOS2\d{9}-\d{6}_\w{3}[RL]2\.2GU[AD])_(HH|HV)_SLP")
CAL_LINEAR = np.float32(10 ** (ALOS2_CAL_DB / 10))

# GDAL settings for reading JAXA's Cloud-Optimized GeoTIFFs over HTTP: skip directory listings,
# and merge/multiplex the tile range requests
REMOTE_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
}


def dn_to_gamma0(block):
    """uint16 DN -> linear gamma0 (DN^2 * 10^-8.3); DN 0 is nodata."""
    block = block.astype("float32")
    gamma0 = block * block * CAL_LINEAR
    gamma0[block == 0] = np.nan
    return gamma0


def resolve_tif(source):
    """A URL as given; otherwise a local .tif path, or a file name looked up in ALOS2_TIF_PATH."""
    if is_url(source):
        return source
    path = Path(source)
    if not path.exists():
        name = path.name if path.name.endswith(".tif") else f"{path.name}.tif"
        path = ALOS2_TIF_PATH / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; download it first")
    return path


def convert_scene_to_geotiff(
    tif_path: Path | str,
    remove_tif: bool = False,
    db: bool = False,
    mask: bool = False,
    bbox: list[float] | None = None,
    aoi_name: str = "crop",
    out_dir: Path | str = ALOS2_GEOTIFF_PATH,
    overwrite: bool = False,
    dst_crs: str | None = None,
    destripe: bool = False,
) -> Path:
    """
    Convert a single, specified PALSAR-2 L2.2 HH/HV GeoTIFF (DN) to a calibrated gamma0 GeoTIFF.
    tif_path can also be a JAXA download URL, read remotely without downloading the whole file
    (worthwhile with a bbox, since only the cropped tiles are fetched). The mask is always a local file.
    dst_crs (e.g. "EPSG:4326") reprojects the output; by default it stays on the scene's UTM grid.
    destripe removes the ScanSAR burst stripes (see src/alos2_destripe.py); they're estimated from the
    whole scene at low resolution, so a remote crop also reads the whole scene at that resolution.
    """

    remote = is_url(tif_path)
    tif_path = resolve_tif(tif_path)

    # Determine scene and polarization from the file name, local or in the URL
    stem = PurePosixPath(urlparse(str(tif_path)).path).stem
    m = L22_FILE_RE.fullmatch(stem)
    if m is None:
        raise ValueError(f"{stem} is not a PALSAR-2 L2.2 HH/HV file")
    scene_id, polarization = m.groups()

    # Mask sits next to a local tif; for a remote tif it's downloaded to ALOS2_TIF_PATH
    mask_dir = ALOS2_TIF_PATH if remote else tif_path.parent
    mask_path = mask_dir / f"{scene_id}_MSK.tif"
    if mask and not mask_path.exists():
        raise FileNotFoundError(
            f"{mask_path.name} not found; download it with alos2_dl --mask"
        )

    # Name the output after the input so different scenes never overwrite each other
    suffix = "gamma0"
    if db:
        suffix += "_dB"
    if mask:
        suffix += "_masked"
    if destripe:
        suffix += "_destriped"
    # CRS before the AOI name: the viewer notebook finds crops by the name ending in _<aoi>
    if dst_crs:
        suffix += f"_epsg{CRS.from_user_input(dst_crs).to_epsg()}"
    if bbox:
        suffix += f"_{aoi_name}"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{suffix}.tif"

    if out_path.exists() and not overwrite:
        print(f"    Skipping {out_path.name} (already exists)")
        return out_path

    # Write to a partial file so a failed conversion never looks complete
    part = out_path.with_suffix(".tif.part")
    # Scene-grid copy, used only when reprojecting: written first, then warped into `part`
    native = out_path.with_suffix(".native.tif.part")
    try:
        with (
            rasterio.Env(**REMOTE_GDAL_ENV) if remote else nullcontext(),
            rasterio.open(f"/vsicurl/{tif_path}" if remote else tif_path) as src,
            rasterio.open(mask_path) if mask else nullcontext() as msk,
        ):
            # Full grid, or just the bbox (reprojected into the scene's UTM CRS)
            full = Window(0, 0, src.width, src.height)
            if bbox:
                b = transform_bounds("EPSG:4326", src.crs, *bbox)
                try:
                    win = (
                        from_bounds(*b, src.transform)
                        .round_offsets()
                        .round_lengths()
                        .intersection(full)
                    )
                except rasterio.errors.WindowError:
                    raise ValueError(f"bbox {bbox} does not overlap {stem}")
            else:
                win = full
            h, w = int(win.height), int(win.width)
            r_off, c_off = int(win.row_off), int(win.col_off)

            # Reproject only when asked for a CRS other than the scene's own
            warp = dst_crs is not None and CRS.from_user_input(dst_crs) != src.crs

            stripes = None
            if destripe:
                # The ocean flag the steps across the swath are measured on: the local mask if
                # there is one, otherwise read from JAXA (~1 MB)
                msk_source = (
                    mask_path
                    if mask_path.exists()
                    else f"/vsicurl/{JAXA_L22_BUCKET_URL}/{JAXA_L22_PREFIX}{scene_id}_MSK.tif"
                )
                with (
                    rasterio.Env(**REMOTE_GDAL_ENV),
                    rasterio.open(msk_source) as ocean_msk,
                ):
                    stripes = scene_stripes(src, ocean_msk)

            with rasterio.open(
                native if warp else part,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype="float32",
                crs=src.crs,
                transform=window_transform(win, src.transform),
                nodata=float("nan"),
                tiled="YES",
                compress="DEFLATE",
                interleave="BAND",
                predictor=3,
                BIGTIFF="yes",
            ) as dst:
                # Write in row blocks so the full array never has to fit in memory
                for r in range(0, h, 2048):
                    n = min(2048, h - r)
                    read_win = Window(c_off, r_off + r, w, n)
                    block = dn_to_gamma0(src.read(1, window=read_win))
                    if stripes is not None:
                        block /= 10 ** (
                            window_stripes(stripes, src.shape, read_win) / 10
                        )
                    if msk is not None:
                        block[
                            np.isin(msk.read(1, window=read_win), ALOS2_MASK_DROP)
                        ] = np.nan
                    # When reprojecting, keep values linear here; reproject_tif converts to dB after resampling
                    dst.write(
                        to_output(block, db and not warp), 1, window=Window(0, r, w, n)
                    )

                dst.update_tags(
                    ATTRIBUTION=ALOS2_ATTRIBUTION,
                    SCENE_ID=scene_id,
                    POLARIZATION=polarization,
                    UNITS="gamma0 (dB)" if db else "gamma0 (linear power)",
                    CALIBRATION="gamma0 = DN^2 * 10^(-8.3)",
                )
                if destripe:
                    dst.update_tags(
                        DESTRIPED="burst stripes removed (src/alos2_destripe.py)"
                    )

            # reproject_tif copies these tags over; UNITS follows db, so it's right after reprojecting too
            if warp:
                reproject_tif(native, part, dst_crs, db)
                native.unlink()

            # Overviews so full scenes render quickly when zoomed out (skipped for small crops)
            with (
                rasterio.Env(COMPRESS_OVERVIEW="DEFLATE"),
                rasterio.open(part, "r+") as dst,
            ):
                factors = [2**i for i in range(1, 10) if max(dst.shape) / 2**i >= 256]
                if factors:
                    dst.build_overviews(factors, Resampling.average)
                    dst.update_tags(ns="rio_overview", resampling="average")

    except Exception:
        part.unlink(missing_ok=True)
        native.unlink(missing_ok=True)
        raise

    part.replace(out_path)

    if remove_tif and not remote:
        tif_path.unlink()

    return out_path


def convert_scenes_to_geotiff(
    tif_paths: list[Path],
    remove_tif: bool = False,
    db: bool = False,
    mask: bool = False,
    bbox: list[float] | None = None,
    aoi_name: str = "crop",
    out_dir: Path | str = ALOS2_GEOTIFF_PATH,
    overwrite: bool = False,
    dst_crs: str | None = None,
    destripe: bool = False,
) -> list[Path]:
    """
    Convert a list of PALSAR-2 L2.2 tifs to calibrated GeoTIFFs.
    """
    tifs: list[Path] = []
    failures: list[tuple[str, str]] = []
    for i, tif_path in enumerate(tif_paths, start=1):
        name = Path(tif_path).name
        print(f"Converting {i}/{len(tif_paths)} {name}")
        try:
            tifs.append(
                convert_scene_to_geotiff(
                    tif_path,
                    remove_tif,
                    db,
                    mask,
                    bbox,
                    aoi_name,
                    out_dir,
                    overwrite,
                    dst_crs,
                    destripe,
                )
            )

        # OSError: missing/unreadable tif or tif write failure (includes RasterioIOError)
        # ValueError: not an L2.2 HH/HV file name, or bbox outside the scene
        except (OSError, ValueError) as exc:
            print(f"  FAILED: {exc}")
            failures.append((name, str(exc)))

    print(f"\nDone. {len(tifs)} converted, {len(failures)} failed.")
    for name, err in failures:
        print(f"  {name}: {err}")
    return tifs


def main():
    parser = argparse.ArgumentParser(
        description="Convert downloaded PALSAR-2 L2.2 GeoTIFFs (DN) to calibrated gamma0 GeoTIFFs."
    )
    parser.add_argument(
        "--tif",
        "--infile",
        dest="tif",
        nargs="+",
        required=True,
        help="One or more _HH_SLP/_HV_SLP .tif files (looked up in data/alos2_tif_scenes/).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the tif's profile and tags instead of converting.",
    )
    parser.add_argument(
        "--remove-tif",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove each source .tif after a successful conversion (default: False).",
    )
    parser.add_argument(
        "--db", action="store_true", help="Write gamma0 in dB (10*log10)."
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Set layover, shadow and invalid pixels to NaN (needs the scene's _MSK.tif).",
    )
    parser.add_argument(
        "--destripe",
        action="store_true",
        help="Remove the ScanSAR burst stripes, estimated from the whole scene (adds _destriped to the name).",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Crop to this lon/lat box at conversion time.",
    )
    parser.add_argument(
        "--aoi-name",
        help="Appended to the output name; required with --bbox so different crops don't share a file name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reconvert even if the GeoTIFF already exists (e.g. after changing --bbox).",
    )
    parser.add_argument(
        "--crs",
        type=epsg_crs,
        help="Output CRS as an EPSG code, e.g. EPSG:4326 or EPSG:3857 (default: the scene's own UTM zone).",
    )
    parser.add_argument(
        "--out-dir",
        "--outdir",
        "--outpath",
        dest="out_dir",
        default=ALOS2_GEOTIFF_PATH,
        help=f"Folder to write GeoTIFFs to; created if missing. Defaults to {ALOS2_GEOTIFF_PATH}.",
    )

    args = parser.parse_args()

    if args.bbox and not args.aoi_name:
        parser.error("--aoi-name is required with --bbox")

    if args.inspect:
        for tif_file in args.tif:
            source = resolve_tif(tif_file)
            remote = is_url(source)
            with (
                rasterio.Env(**REMOTE_GDAL_ENV) if remote else nullcontext(),
                rasterio.open(f"/vsicurl/{source}" if remote else source) as src,
            ):
                print(src.profile)
                print(src.tags())
    else:
        convert_scenes_to_geotiff(
            args.tif,
            args.remove_tif,
            args.db,
            args.mask,
            args.bbox,
            args.aoi_name,
            args.out_dir,
            args.overwrite,
            args.crs,
            args.destripe,
        )


if __name__ == "__main__":
    main()
