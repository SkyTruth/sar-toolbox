import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from src.s1wv_dl import read_annotation
from utils.geometry import epsg_crs
from utils.variables import (
    S1_GRD_BETA0_CALIBRATION,
    S1WV_ATTRIBUTION,
    S1WV_GEOTIFF_PATH,
    S1WV_GRD_RESOLUTION_M,
    S1WV_LEE_ENL_FRACTION,
    S1WV_LEE_WINDOW,
    S1WV_PIXEL_SIZE_M,
    S1WV_SAFE_PATH,
)


def resolve_tiff(source) -> Path:
    """A local measurement .tiff path, or a file name looked up in S1WV_SAFE_PATH/<product>/measurement/."""
    path = Path(source)
    if not path.exists():
        name = path.name if path.name.endswith(".tiff") else f"{path.name}.tiff"
        matches = list(S1WV_SAFE_PATH.glob(f"*/measurement/{name}"))
        if matches:
            path = matches[0]
    if not path.exists():
        raise FileNotFoundError(f"{source} not found; download it first")
    return path


def sidecar_paths(tiff_path: Path) -> dict[str, Path]:
    """The annotation and calibration XMLs s1wv_dl saves next to a measurement file (.SAFE layout)."""
    stem = tiff_path.name.removesuffix(".tiff")
    annotation = tiff_path.parent.parent / "annotation"
    paths = {
        "annotation": annotation / f"{stem}.xml",
        "calibration": annotation / "calibration" / f"calibration-{stem}.xml",
    }
    for kind, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{kind} file {path} not found; download the vignette with s1wv_dl"
            )
    return paths


def interp_lut(path: Path, vector_tag: str, value_tag: str, rows, cols):
    """
    Bilinear interpolation of a calibration lookup table at the given image rows and columns (clamped at the
    table's edges).
    """
    vectors = ET.parse(path).getroot().findall(f".//{vector_tag}")
    if not vectors:
        raise ValueError(f"no {vector_tag} in {path.name}")
    pixels = vectors[0].findtext("pixel").split()
    if any(v.findtext("pixel").split() != pixels for v in vectors):
        raise ValueError(
            f"{vector_tag}s in {path.name} use different pixel positions; can't interpolate on a grid"
        )
    lines = np.array([int(v.findtext("line")) for v in vectors], dtype=float)
    pixels = np.array(pixels, dtype=float)
    values = np.array(
        [[float(x) for x in v.findtext(value_tag).split()] for v in vectors]
    )

    # Along each vector (columns), then between vectors (rows)
    along = np.array([np.interp(cols, pixels, v) for v in values])
    i = np.clip(np.searchsorted(lines, rows) - 1, 0, len(lines) - 2)
    t = np.clip((rows - lines[i]) / (lines[i + 1] - lines[i]), 0, 1)[:, None]
    return along[i] * (1 - t) + along[i + 1] * t


def box_mean(a, size: int):
    """Mean over a size x size window centred on each pixel (edges reflected), via a summed-area table."""
    r = size // 2
    c = np.pad(a.astype("float64"), r, mode="reflect").cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (
        c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]
    ) / size**2


def lee_filter(intensity, window: int, enl: float):
    """
    Lee speckle filter on linear intensity. Each pixel moves from its window mean towards its own value by
    k = (local variance not explained by speckle) / (local variance), so flat sea is smoothed towards the
    mean while edges and bright targets, where variance is well above speckle's, are mostly kept.
    Speckle's squared coefficient of variation for intensity is 1 / ENL (equivalent number of looks).
    """
    cu2 = 1 / enl
    mean = box_mean(intensity, window)
    var = np.maximum(
        box_mean(intensity * intensity.astype("float64"), window) - mean**2, 0
    )
    var_x = np.maximum((var - mean**2 * cu2) / (1 + cu2), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.clip(np.where(var > 0, var_x / var, 0), 0, 1)
    return mean + k * (intensity - mean)


def multilook(tiff_path: Path, ann: dict):
    """
    |DN|^2 of the SLC averaged over blocks of about IW GRDH resolution (S1WV_GRD_RESOLUTION_M), plus the
    (azimuth, range) looks and the block centres' image rows and columns (where lookup tables are evaluated).
    Looks come from the vignette's pixel spacing, so WV1 (~23 deg) gets fewer range looks than WV2 (~37).
    """
    ground_range_m = ann["range_spacing_m"] / np.sin(np.deg2rad(ann["incidence_deg"]))
    looks = (
        max(1, round(S1WV_GRD_RESOLUTION_M[1] / ann["azimuth_spacing_m"])),
        max(1, round(S1WV_GRD_RESOLUTION_M[0] / ground_range_m)),
    )

    with rasterio.open(tiff_path) as src:
        dn = src.read(1, out_dtype="complex64")
    intensity = dn.real**2 + dn.imag**2
    del dn
    h = intensity.shape[0] // looks[0] * looks[0]
    w = intensity.shape[1] // looks[1] * looks[1]
    intensity = (
        intensity[:h, :w]
        .reshape(h // looks[0], looks[0], w // looks[1], looks[1])
        .mean(axis=(1, 3))
    )
    rows = np.arange(intensity.shape[0]) * looks[0] + (looks[0] - 1) / 2
    cols = np.arange(intensity.shape[1]) * looks[1] + (looks[1] - 1) / 2
    return intensity, looks, rows, cols


def grd_dn(tiff_path: Path, paths: dict[str, Path], ann: dict, lee_window=None):
    """
    One vignette as GRD-like DN in radar geometry (float32), with the looks and the ENL the Lee filter used
    (None if not filtered).

    GRD-like DN is the amplitude an IW GRDH product would store, so the vignette matches what Cerulean's model
    reads. Both products convert DN to beta0 (radar brightness) with a constant (beta0 = DN^2 / A^2), so
    DN_GRD = sqrt(multilooked |DN_SLC|^2) / A_SLC * A_GRD, with A_GRD = S1_GRD_BETA0_CALIBRATION and A_SLC
    from the vignette's calibration file. As in GRD, there's no calibration, noise removal or speckle
    filtering; lee_window optionally applies a Lee speckle filter (on intensity, before taking the amplitude).
    """
    intensity, looks, rows, cols = multilook(tiff_path, ann)

    enl = None
    if lee_window:
        enl = S1WV_LEE_ENL_FRACTION * looks[0] * looks[1]
        intensity = lee_filter(intensity, lee_window, enl)

    # The vignette's beta0 calibration constant (ESA's XML calls it betaNought)
    a_beta0 = interp_lut(
        paths["calibration"], "calibrationVector", "betaNought", rows, cols
    )
    dn = np.sqrt(np.maximum(intensity, 0)) / a_beta0 * S1_GRD_BETA0_CALIBRATION
    return dn.astype("float32"), looks, enl


def geocode(data, ann: dict, looks: tuple[int, int], dst_crs: str | None = None):
    """
    Warp radar-geometry data onto a north-up grid, using a thin-plate spline through the annotation's tie
    points (sea level, so no terrain correction is needed). By default the grid is the vignette's UTM zone
    with S1WV_PIXEL_SIZE_M pixels; dst_crs (e.g. "EPSG:4326") warps to that CRS instead, at about the same
    resolution. Tie points mark pixel centres; GCP rows/cols count from the image's top-left corner.
    """
    gcps = [
        GroundControlPoint(
            row=(p["line"] + 0.5) / looks[0],
            col=(p["pixel"] + 0.5) / looks[1],
            x=p["longitude"],
            y=p["latitude"],
            z=p["height"],
        )
        for p in ann["tie_points"]
    ]
    zone = int((ann["lon"] + 180) // 6) + 1
    utm = CRS.from_epsg((32700 if ann["lat"] < 0 else 32600) + zone)
    crs = CRS.from_user_input(dst_crs) if dst_crs else utm
    # Pixel size is only given in metres for a projected CRS; otherwise GDAL picks about the same resolution
    resolution = {"resolution": S1WV_PIXEL_SIZE_M} if crs.is_projected else {}
    transform, width, height = calculate_default_transform(
        "EPSG:4326",
        crs,
        data.shape[1],
        data.shape[0],
        gcps=gcps,
        SRC_METHOD="GCP_TPS",
        **resolution,
    )
    out = np.full((height, width), np.nan, dtype="float32")
    reproject(
        data,
        out,
        gcps=gcps,
        src_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_transform=transform,
        dst_crs=crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
        SRC_METHOD="GCP_TPS",
    )
    return out, transform, crs


def output_path(
    tiff_name: str,
    out_dir: Path | str = S1WV_GEOTIFF_PATH,
    lee_window: int | None = None,
    dst_crs: str | None = None,
) -> Path:
    """
    GeoTIFF path for a measurement file: named after it (unique per vignette), plus a tag for each option, so
    vignettes and options never collide, e.g. <vignette>.tif or <vignette>_lee5_epsg4326.tif.
    """
    tags = ""
    if lee_window:
        tags += f"_lee{lee_window}"
    if dst_crs:
        tags += f"_epsg{CRS.from_user_input(dst_crs).to_epsg()}"
    return Path(out_dir) / f"{tiff_name.removesuffix('.tiff')}{tags}.tif"


def convert_vignette_to_geotiff(
    tiff_path: Path | str,
    remove_slc: bool = False,
    out_dir: Path | str = S1WV_GEOTIFF_PATH,
    overwrite: bool = False,
    dst_crs: str | None = None,
    lee_window: int | None = None,
) -> Path:
    """
    Convert a single downloaded WV vignette (complex SLC measurement .tiff, radar geometry) to a geocoded
    GeoTIFF of GRD-like DN (uint16, 0 = nodata), what Cerulean's model reads from IW GRDH. The annotation and
    calibration XMLs must sit beside it in the .SAFE layout s1wv_dl writes.
    lee_window (odd, e.g. 5) applies a Lee speckle filter. dst_crs (e.g. "EPSG:4326") warps the output to
    that CRS; by default it is on the vignette's UTM zone.
    """
    if lee_window is not None and (lee_window < 3 or lee_window % 2 == 0):
        raise ValueError(f"Lee window must be odd and at least 3, not {lee_window}")
    tiff_path = resolve_tiff(tiff_path)
    paths = sidecar_paths(tiff_path)
    ann = read_annotation(paths["annotation"])
    product_name = tiff_path.parent.parent.name
    stem = tiff_path.name.removesuffix(".tiff")
    out_path = output_path(tiff_path.name, out_dir, lee_window, dst_crs)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f"    Skipping {out_path.name} (already exists)")
        return out_path

    # Write to a partial file so a failed conversion never looks complete
    part = out_path.with_suffix(".tif.part")
    try:
        dn, looks, enl = grd_dn(tiff_path, paths, ann, lee_window)
        dn, transform, crs = geocode(dn, ann, looks, dst_crs)
        # Integer DN like GRD's; 0 is nodata, so real pixels are at least 1
        dn = np.where(np.isnan(dn), 0, np.clip(np.round(dn), 1, 65535)).astype("uint16")

        with rasterio.open(
            part,
            "w",
            driver="GTiff",
            height=dn.shape[0],
            width=dn.shape[1],
            count=1,
            dtype="uint16",
            crs=crs,
            transform=transform,
            nodata=0,
            tiled="YES",
            compress="DEFLATE",
            interleave="BAND",
            predictor=2,
        ) as dst:
            dst.write(dn, 1)
            dst.update_tags(
                ATTRIBUTION=f"{S1WV_ATTRIBUTION} {ann['start'][:4]}",
                SCENE_ID=product_name,
                VIGNETTE=stem,
                SWATH=ann["swath"],
                POLARIZATION=ann["polarisation"],
                START_TIME=ann["start"],
                INCIDENCE_ANGLE_DEG=f"{ann['incidence_deg']:.2f}",
                LOOKS_RANGE_AZIMUTH=f"{looks[1]}x{looks[0]}",
                UNITS=f"GRD-like DN (amplitude): beta0 = DN^2 / {S1_GRD_BETA0_CALIBRATION:g}^2, as in IW GRDH",
                SPECKLE_FILTER=f"Lee {lee_window}x{lee_window}, ENL {enl:.1f}"
                if lee_window
                else "none",
            )

    except Exception:
        part.unlink(missing_ok=True)
        raise

    part.replace(out_path)

    # Only the ~105 MB measurement file goes; the small XMLs stay so the vignette can be re-resolved cheaply
    if remove_slc:
        tiff_path.unlink()

    return out_path


def convert_vignettes_to_geotiff(
    tiff_paths: list[Path],
    remove_slc: bool = False,
    out_dir: Path | str = S1WV_GEOTIFF_PATH,
    overwrite: bool = False,
    dst_crs: str | None = None,
    lee_window: int | None = None,
) -> list[Path]:
    """
    Convert a list of WV vignettes to GeoTIFFs.
    """
    tifs: list[Path] = []
    failures: list[tuple[str, str]] = []
    for i, tiff_path in enumerate(tiff_paths, start=1):
        name = Path(tiff_path).name
        print(f"Converting {i}/{len(tiff_paths)} {name}")
        try:
            tifs.append(
                convert_vignette_to_geotiff(
                    tiff_path, remove_slc, out_dir, overwrite, dst_crs, lee_window
                )
            )

        # OSError: missing/unreadable tiff or XML, or tif write failure (includes RasterioIOError)
        # ValueError: lookup table that can't be interpolated on a grid
        # ET.ParseError: truncated or invalid XML
        except (OSError, ValueError, ET.ParseError) as exc:
            print(f"  FAILED: {exc}")
            failures.append((name, str(exc)))

    print(f"\nDone. {len(tifs)} converted, {len(failures)} failed.")
    for name, err in failures:
        print(f"  {name}: {err}")
    return tifs


def odd_window(value: str) -> int:
    """argparse type for --lee: an odd window size of at least 3."""
    n = int(value)
    if n < 3 or n % 2 == 0:
        raise argparse.ArgumentTypeError(f"must be odd and at least 3, not {n}")
    return n


def add_lee_argument(parser: argparse.ArgumentParser) -> None:
    """--lee, shared by s1wv_convert_to_tiff and s1wv_access."""
    parser.add_argument(
        "--lee",
        nargs="?",
        const=S1WV_LEE_WINDOW,
        type=odd_window,
        metavar="WINDOW",
        help=f"Apply a Lee speckle filter (GRD and Cerulean don't); optional odd window size (default "
        f"{S1WV_LEE_WINDOW}, i.e. {S1WV_LEE_WINDOW}x{S1WV_LEE_WINDOW}). Adds _lee<WINDOW> to the file name.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert downloaded Sentinel-1 WV vignettes (SLC) to geocoded GeoTIFFs of GRD-like DN, "
        "the values Cerulean's model reads from IW GRDH."
    )
    parser.add_argument(
        "--tiff",
        "--infile",
        dest="tiff",
        nargs="+",
        required=True,
        help="One or more measurement .tiff files (looked up in data/s1wv_safe_files/*/measurement/).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the vignette's annotation summary instead of converting.",
    )
    parser.add_argument(
        "--remove-slc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove each measurement .tiff after a successful conversion (default: False).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reconvert even if the GeoTIFF already exists.",
    )
    parser.add_argument(
        "--crs",
        type=epsg_crs,
        help="Output CRS as an EPSG code, e.g. EPSG:4326 or EPSG:3857 (default: the vignette's UTM zone).",
    )
    parser.add_argument(
        "--out-dir",
        "--outdir",
        "--outpath",
        dest="out_dir",
        default=S1WV_GEOTIFF_PATH,
        help=f"Folder to write GeoTIFFs to; created if missing. Defaults to {S1WV_GEOTIFF_PATH}.",
    )
    add_lee_argument(parser)

    args = parser.parse_args()

    if args.inspect:
        for tiff in args.tiff:
            ann = read_annotation(sidecar_paths(resolve_tiff(tiff))["annotation"])
            print({k: v for k, v in ann.items() if k != "tie_points"})
    else:
        convert_vignettes_to_geotiff(
            args.tiff,
            args.remove_slc,
            args.out_dir,
            args.overwrite,
            args.crs,
            args.lee,
        )


if __name__ == "__main__":
    main()
