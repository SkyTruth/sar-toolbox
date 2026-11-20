import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from src.nisar_convert_to_tiff import reproject_tif, to_output
from src.s1wv_dl import read_annotation
from utils.geometry import epsg_crs
from utils.variables import (
    S1WV_ATTRIBUTION,
    S1WV_GEOTIFF_PATH,
    S1WV_GRD_RESOLUTION_M,
    S1WV_PIXEL_SIZE_M,
    S1WV_SAFE_PATH,
    S1WV_SIGMA0_FLOOR,
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
    """The annotation, calibration and noise XMLs s1wv_dl saves next to a measurement file (.SAFE layout)."""
    stem = tiff_path.name.removesuffix(".tiff")
    annotation = tiff_path.parent.parent / "annotation"
    paths = {
        "annotation": annotation / f"{stem}.xml",
        "calibration": annotation / "calibration" / f"calibration-{stem}.xml",
        "noise": annotation / "calibration" / f"noise-{stem}.xml",
    }
    for kind, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{kind} file {path} not found; download the vignette with s1wv_dl"
            )
    return paths


def interp_lut(
    path: Path, vector_tags: tuple[str, ...], value_tags: tuple[str, ...], rows, cols
):
    """
    Bilinear interpolation of a calibration or noise lookup table at the given image rows and columns
    (clamped at the table's edges). The first of vector_tags/value_tags present is used, so older IPF
    noise files (noiseVector/noiseLut) work as well as newer ones (noiseRangeVector/noiseRangeLut).
    """
    root = ET.parse(path).getroot()
    for vector_tag, value_tag in zip(vector_tags, value_tags):
        vectors = root.findall(f".//{vector_tag}")
        if vectors:
            break
    else:
        raise ValueError(f"no {' or '.join(vector_tags)} in {path.name}")
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


def load_sigma0(tiff_path: Path, paths: dict[str, Path], ann: dict):
    """
    Noise-subtracted sigma0 (linear), multilooked to about IW GRDH resolution, and the (azimuth, range) looks.
    sigma0 = (|DN|^2 - N) / A^2 with N the range noise LUT and A the sigmaNought LUT, as in Earth Engine's
    S1_GRD. Looks come from the vignette's pixel spacing, so WV1 (~23 deg) gets fewer range looks than WV2 (~37).
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

    # Both lookup tables vary smoothly, so they're evaluated at block centres
    rows = np.arange(intensity.shape[0]) * looks[0] + (looks[0] - 1) / 2
    cols = np.arange(intensity.shape[1]) * looks[1] + (looks[1] - 1) / 2
    a_sigma = interp_lut(
        paths["calibration"], ("calibrationVector",), ("sigmaNought",), rows, cols
    )
    noise = interp_lut(
        paths["noise"],
        ("noiseRangeVector", "noiseVector"),
        ("noiseRangeLut", "noiseLut"),
        rows,
        cols,
    )
    if ET.parse(paths["noise"]).getroot().find(".//noiseAzimuthVector") is not None:
        print(
            "    This product has azimuth noise vectors too; only range noise is subtracted."
        )
    sigma0 = (intensity - noise) / a_sigma**2
    return np.maximum(sigma0, S1WV_SIGMA0_FLOOR).astype("float32"), looks


def geocode(sigma0, ann: dict, looks: tuple[int, int]):
    """
    Warp radar-geometry sigma0 onto a north-up UTM grid with S1WV_PIXEL_SIZE_M pixels, using a thin-plate
    spline through the annotation's tie points (sea level, so no terrain correction is needed).
    Tie points mark pixel centres; GCP rows/cols count from the image's top-left corner.
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
    dst_crs = CRS.from_epsg((32700 if ann["lat"] < 0 else 32600) + zone)
    transform, width, height = calculate_default_transform(
        "EPSG:4326",
        dst_crs,
        sigma0.shape[1],
        sigma0.shape[0],
        gcps=gcps,
        resolution=S1WV_PIXEL_SIZE_M,
        SRC_METHOD="GCP_TPS",
    )
    out = np.full((height, width), np.nan, dtype="float32")
    reproject(
        sigma0,
        out,
        gcps=gcps,
        src_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_transform=transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
        SRC_METHOD="GCP_TPS",
    )
    return out, transform, dst_crs


def output_path(
    tiff_name: str,
    db: bool = False,
    out_dir: Path | str = S1WV_GEOTIFF_PATH,
    dst_crs: str | None = None,
) -> Path:
    """GeoTIFF path for a measurement file: named after it (unique per vignette), so vignettes never collide."""
    suffix = "sigma0"
    if db:
        suffix += "_dB"
    if dst_crs:
        suffix += f"_epsg{CRS.from_user_input(dst_crs).to_epsg()}"
    return Path(out_dir) / f"{tiff_name.removesuffix('.tiff')}_{suffix}.tif"


def convert_vignette_to_geotiff(
    tiff_path: Path | str,
    remove_slc: bool = False,
    db: bool = False,
    out_dir: Path | str = S1WV_GEOTIFF_PATH,
    overwrite: bool = False,
    dst_crs: str | None = None,
) -> Path:
    """
    Convert a single downloaded WV vignette (complex SLC measurement .tiff, radar geometry) to a calibrated,
    noise-subtracted, multilooked and geocoded sigma0 GeoTIFF. The annotation, calibration and noise XMLs
    must sit beside it in the .SAFE layout s1wv_dl writes.
    dst_crs (e.g. "EPSG:4326") reprojects the output; by default it is on the vignette's UTM zone.
    """
    tiff_path = resolve_tiff(tiff_path)
    paths = sidecar_paths(tiff_path)
    ann = read_annotation(paths["annotation"])
    product_name = tiff_path.parent.parent.name
    stem = tiff_path.name.removesuffix(".tiff")
    out_path = output_path(tiff_path.name, db, out_dir, dst_crs)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f"    Skipping {out_path.name} (already exists)")
        return out_path

    # Write to a partial file so a failed conversion never looks complete
    part = out_path.with_suffix(".tif.part")
    # UTM copy, used only when reprojecting: written first, then warped into `part`
    native = out_path.with_suffix(".native.tif.part")
    try:
        sigma0, looks = load_sigma0(tiff_path, paths, ann)
        sigma0_geo, transform, utm_crs = geocode(sigma0, ann, looks)
        del sigma0

        # Reproject only when asked for a CRS other than the vignette's own UTM zone
        warp = dst_crs is not None and CRS.from_user_input(dst_crs) != utm_crs

        with rasterio.open(
            native if warp else part,
            "w",
            driver="GTiff",
            height=sigma0_geo.shape[0],
            width=sigma0_geo.shape[1],
            count=1,
            dtype="float32",
            crs=utm_crs,
            transform=transform,
            nodata=float("nan"),
            tiled="YES",
            compress="DEFLATE",
            interleave="BAND",
            predictor=3,
        ) as dst:
            # When reprojecting, keep values linear here; reproject_tif converts to dB after resampling
            dst.write(to_output(sigma0_geo, db and not warp), 1)
            dst.update_tags(
                ATTRIBUTION=f"{S1WV_ATTRIBUTION} {ann['start'][:4]}",
                SCENE_ID=product_name,
                VIGNETTE=stem,
                SWATH=ann["swath"],
                POLARIZATION=ann["polarisation"],
                START_TIME=ann["start"],
                INCIDENCE_ANGLE_DEG=f"{ann['incidence_deg']:.2f}",
                LOOKS_RANGE_AZIMUTH=f"{looks[1]}x{looks[0]}",
                UNITS="sigma0 (dB)" if db else "sigma0 (linear power)",
                CALIBRATION=f"sigma0 = (|DN|^2 - noise) / sigmaNought^2, floored at {S1WV_SIGMA0_FLOOR:g}",
            )

        # reproject_tif copies these tags over; UNITS follows db, so it's right after reprojecting too
        if warp:
            reproject_tif(native, part, dst_crs, db)
            native.unlink()

    except Exception:
        part.unlink(missing_ok=True)
        native.unlink(missing_ok=True)
        raise

    part.replace(out_path)

    # Only the ~105 MB measurement file goes; the small XMLs stay so the vignette can be re-resolved cheaply
    if remove_slc:
        tiff_path.unlink()

    return out_path


def convert_vignettes_to_geotiff(
    tiff_paths: list[Path],
    remove_slc: bool = False,
    db: bool = False,
    out_dir: Path | str = S1WV_GEOTIFF_PATH,
    overwrite: bool = False,
    dst_crs: str | None = None,
) -> list[Path]:
    """
    Convert a list of WV vignettes to sigma0 GeoTIFFs.
    """
    tifs: list[Path] = []
    failures: list[tuple[str, str]] = []
    for i, tiff_path in enumerate(tiff_paths, start=1):
        name = Path(tiff_path).name
        print(f"Converting {i}/{len(tiff_paths)} {name}")
        try:
            tifs.append(
                convert_vignette_to_geotiff(
                    tiff_path, remove_slc, db, out_dir, overwrite, dst_crs
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert downloaded Sentinel-1 WV vignettes (SLC) to calibrated sigma0 GeoTIFFs."
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
        "--db", action="store_true", help="Write sigma0 in dB (10*log10)."
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

    args = parser.parse_args()

    if args.inspect:
        for tiff in args.tiff:
            ann = read_annotation(sidecar_paths(resolve_tiff(tiff))["annotation"])
            print({k: v for k, v in ann.items() if k != "tie_points"})
    else:
        convert_vignettes_to_geotiff(
            args.tiff, args.remove_slc, args.db, args.out_dir, args.overwrite, args.crs
        )


if __name__ == "__main__":
    main()
