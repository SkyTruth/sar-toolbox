"""
Lee speckle filtering of NISAR and PALSAR-2 backscatter.

    uv run python -m src.lee --scene SCENE [SCENE ...]

A scene is a GeoTIFF written by nisar_convert_to_tiff / alos2_convert_to_tiff (a bare file name is looked up in
the sensor's geotiff folder). The filtered copy is written next to it in the same units, with the settings in its
name, e.g. <name>_ml1x1_refinedLee_enlAuto.tif or <name>_ml3x3_lee_win9_enl4.5.tif.

--multilook N averages N x N blocks of pixels before filtering: much less speckle, on pixels N times larger.
"""

import argparse
import math
from functools import partial
from pathlib import Path

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window

from src.cfar import scene_geotiff
from src.nisar_convert_to_tiff import to_output
from utils.variables import NISAR_PROCESSING_INFO, SEARCH_AOIS

# Pixels per row block, which bounds memory on full scenes (the refined filter holds ~15 float64 arrays of it)
LEE_BLOCK_PIXELS = 4_000_000
# Scene ENL estimate: tile width in pixels, strips of tiles sampled down the scene, and the percentile of the
# tiles' variance / mean^2 that is taken as speckle alone
ENL_TILE = 16
ENL_STRIPS = 24
ENL_PERCENTILE = 10
# File name suffix per method, so a scene's refined and plain outputs sit side by side
METHOD_SUFFIXES = {"refined": "refinedLee", "lee": "lee"}

# The 8 sides a 7x7 window can be split toward, as (row, col) steps: N/S, W/E, NW/SE, NE/SW. Each pair is the
# two sides of one edge orientation.
SIDES = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
_DY, _DX = np.mgrid[-3:4, -3:4]
# Per side, the half of the 7x7 window lying toward it, center line included (28 pixels)
HALF_WINDOWS = np.stack([sy * _DY + sx * _DX >= 0 for sy, sx in SIDES])


def box_sum(a, half: int):
    """Sum of `a` over the (2*half+1)^2 window around each pixel; outside the array counts as 0."""
    h, w = a.shape
    p = np.pad(a, half)
    rows = sum(p[i : i + h] for i in range(2 * half + 1))
    return sum(rows[:, j : j + w] for j in range(2 * half + 1))


def shifted(padded, dy: int, dx: int, pad: int):
    """View of an array padded by `pad` on every side, moved so each pixel sees its neighbor at (dy, dx)."""
    h, w = padded.shape[0] - 2 * pad, padded.shape[1] - 2 * pad
    return padded[pad + dy : pad + dy + h, pad + dx : pad + dx + w]


def mmse(power, mean, var, sigma_v2):
    """
    Lee's estimate of the speckle-free value from a pixel and its window's mean and variance:
    mean + b * (pixel - mean), where b is the share of the window's variance that speckle (relative variance
    sigma_v2 = 1 / ENL) can't explain. Flat areas get the mean, edges and targets keep the pixel.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        var_x = (var - mean**2 * sigma_v2) / (1 + sigma_v2)
        b = np.clip(np.where(var > 0, var_x / var, 0), 0, 1)
    out = mean + b * (power - mean)
    # Pixels whose window has no usable statistics pass through unchanged; NaN stays NaN
    return np.where(np.isfinite(out), out, power).astype("float32")


def lee(power, enl, half: int = 3):
    """Lee (1980) filter over a square window of half-width `half` pixels. Fast, but leaves edges noisy."""
    power = power.astype("float64")
    valid = np.isfinite(power)
    x = np.where(valid, power, 0)
    n = box_sum(valid.astype("float64"), half)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = box_sum(x, half) / n
        var = (box_sum(x * x, half) / n - mean**2) * n / (n - 1)
    return mmse(power, mean, var, 1 / np.asarray(enl))


def refined_lee(power, enl=None):
    """
    Refined Lee (1981) filter on a 2-D array of linear power (NaN = no data), after the Earth Engine version by
    Guido Lemoine: find the strongest edge through each pixel's 7x7 window from the means of nine 3x3
    sub-windows, then filter with the statistics of the half window on the pixel's own side of that edge.

    enl is the equivalent number of looks, a number or a per-pixel array. With None the speckle level is
    estimated per pixel from the sub-windows, as the Earth Engine version does.
    """
    power = power.astype("float64")
    valid = np.isfinite(power)
    x = np.where(valid, power, 0)
    h, w = x.shape

    n3 = box_sum(valid.astype("float64"), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean3 = box_sum(x, 1) / n3
        var3 = box_sum(x * x, 1) / n3 - mean3**2
    sub_mean = np.pad(mean3, 2, constant_values=np.nan)

    # Of the four orientations, the one whose opposite sub-windows differ most; then the side whose
    # sub-window is closer to the center one, which is the side of the edge the pixel is on
    direction = np.zeros((h, w), "uint8")
    steepest = np.full((h, w), -1.0)
    for k in range(4):
        a, b = (
            shifted(sub_mean, 2 * sy, 2 * sx, 2) for sy, sx in SIDES[2 * k : 2 * k + 2]
        )
        gradient = np.abs(a - b)
        toward_b = np.abs(b - mean3) < np.abs(a - mean3)
        # False where a sub-window is missing (scene edge, no data)
        better = gradient > steepest
        direction[better] = (2 * k + toward_b)[better]
        steepest[better] = gradient[better]

    # Mean and variance over each pixel's own half window, one neighbor offset at a time
    xp, vp = np.pad(x, 3), np.pad(valid, 3)
    n, s, s2 = (np.zeros((h, w)) for _ in range(3))
    for oy in range(7):
        for ox in range(7):
            use = HALF_WINDOWS[:, oy, ox][direction]
            xs = np.where(use, xp[oy : oy + h, ox : ox + w], 0)
            n += use & vp[oy : oy + h, ox : ox + w]
            s += xs
            s2 += xs * xs
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s / n
        var = (s2 / n - mean**2) * n / (n - 1)

    if enl is not None:
        sigma_v2 = 1 / np.asarray(enl)
    else:
        # Mean of the 5 smallest variance / mean^2 among the nine sub-windows
        with np.errstate(divide="ignore", invalid="ignore"):
            cv2 = np.pad((var3 / mean3**2).astype("float32"), 2, constant_values=np.nan)
        stack = np.stack(
            [shifted(cv2, i, j, 2) for i in (-2, 0, 2) for j in (-2, 0, 2)]
        )
        stack.sort(axis=0)  # NaN sorts last
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_v2 = np.nansum(stack[:5], axis=0) / np.isfinite(stack[:5]).sum(axis=0)
    return mmse(power, mean, var, sigma_v2)


def read_power(src, window: Window, in_db: bool):
    """A window of band 1 as linear power, NaN where there is no data."""
    block = src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
    return 10 ** (block / 10) if in_db else block


def multilook_grid(src, multilook: int):
    """
    Where the multilook x multilook blocks fall on an open scene: the (row, col) of the first block's corner in
    the scene, then the multilooked height, width and transform.

    Blocks sit on multiples of the new pixel size in map coordinates (like gdalwarp -tap) rather than starting at
    the scene's own corner, so multilooked scenes and crops of the same area line up pixel for pixel. The first
    block can therefore start up to multilook - 1 pixels before the scene.
    """
    t = src.transform
    # Scene corner in whole pixels from the map origin. Rounded first so float noise can't move it across a
    # pixel, then floored so corners at half pixels (PALSAR-2) all fall the same way.
    row0 = -(math.floor(round(t.f / t.e, 6)) % multilook)
    col0 = -(math.floor(round(t.c / t.a, 6)) % multilook)
    height = math.ceil((src.height - row0) / multilook)
    width = math.ceil((src.width - col0) / multilook)
    transform = t * Affine.translation(col0, row0) * Affine.scale(multilook)
    return row0, col0, height, width, transform


def multilook_blocks(power, multilook: int):
    """
    Mean of each multilook x multilook block of linear power (both sides of the array must be multiples of it).
    A block with less than half of its pixels valid becomes NaN; otherwise it is the mean of the valid ones.
    """
    h, w = power.shape
    blocks = power.reshape(h // multilook, multilook, w // multilook, multilook)
    valid = np.isfinite(blocks)
    n = valid.sum(axis=(1, 3))
    total = np.where(valid, blocks, 0).sum(axis=(1, 3), dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(2 * n >= multilook**2, total / n, np.nan).astype("float32")


def read_rows(src, top: int, bottom: int, in_db: bool, multilook: int = 1):
    """
    Rows top..bottom of the scene at full width, as linear power (NaN = no data). With multilook > 1 they are
    rows of the multilooked scene (see multilook_grid).
    """
    if multilook == 1:
        return read_power(src, Window(0, top, src.width, bottom - top), in_db)
    row0, col0, _, width, _ = multilook_grid(src, multilook)
    # Scene rows and columns the blocks cover; the parts of edge blocks that lie outside the scene are NaN
    r0, r1 = row0 + top * multilook, row0 + bottom * multilook
    c1 = col0 + width * multilook
    first, last = max(r0, 0), min(r1, src.height)
    block = read_power(src, Window(0, first, src.width, last - first), in_db)
    block = np.pad(
        block,
        ((first - r0, r1 - last), (-col0, c1 - src.width)),
        constant_values=np.nan,
    )
    return multilook_blocks(block, multilook)


def estimate_enl(src, in_db: bool, multilook: int = 1) -> float:
    """
    Equivalent number of looks of an open scene: mean^2 / variance of its pixels where the ground is uniform.
    Measured on ENL_TILE-pixel tiles in strips sampled down the scene. Texture, edges and targets only ever add
    variance, so the speckle level is read from the most uniform tiles rather than the typical one.

    With multilook > 1 it is measured on the multilooked scene, since that is what gets filtered.
    """
    _, _, height, width, _ = multilook_grid(src, multilook)
    rows = min(height, 4 * ENL_TILE) // ENL_TILE * ENL_TILE
    cols = width // ENL_TILE * ENL_TILE
    starts = np.unique(np.linspace(0, height - rows, ENL_STRIPS).astype(int))
    cv2 = []
    for r in starts if rows and cols else []:
        strip = read_rows(src, int(r), int(r) + rows, in_db, multilook)[:, :cols]
        strip = strip.astype("float64")
        tiles = strip.reshape(rows // ENL_TILE, ENL_TILE, cols // ENL_TILE, ENL_TILE)
        tiles = tiles.swapaxes(1, 2).reshape(-1, ENL_TILE**2)
        tiles = tiles[np.isfinite(tiles).all(axis=1)]
        mean = tiles.mean(axis=1)
        cv2.append(tiles[mean > 0].var(axis=1, ddof=1) / mean[mean > 0] ** 2)
    cv2 = np.concatenate(cv2) if cv2 else np.empty(0)
    cv2 = cv2[cv2 > 0]
    if cv2.size < 100:
        raise ValueError(
            "too little data to estimate the number of looks; give it with --enl"
        )
    return float(1 / np.percentile(cv2, ENL_PERCENTILE))


def filter_dataset(src, dst, filt, halo: int, in_db: bool, multilook: int = 1) -> None:
    """
    Filter band 1 of src into dst in row blocks, so a full scene never has to fit in memory. Each block is read
    with `halo` extra rows above and below, enough that the rows kept match filtering the scene in one piece.
    With multilook > 1 the rows are those of the multilooked scene, which is dst's grid.
    """
    step = max(256, LEE_BLOCK_PIXELS // (src.width * multilook))
    for r0 in range(0, dst.height, step):
        r1 = min(r0 + step, dst.height)
        top, bottom = max(0, r0 - halo), min(dst.height, r1 + halo)
        block = read_rows(src, top, bottom, in_db, multilook)
        out = filt(block)[r0 - top : r1 - top]
        dst.write(to_output(out, in_db), 1, window=Window(0, r0, dst.width, r1 - r0))


def output_name(
    stem: str, method: str, enl: float | str | None, window: int, multilook: int
) -> str:
    """
    <stem>_<settings>, with a crop's _<aoi> kept last: the viewer notebook finds crops by the name ending in it.

    The settings are the arguments as given, so runs of a scene with different settings never share a file name:
    the multilook block (ml1x1 is none), the method, the window for the plain filter (the refined one is always
    7x7), then the number of looks as enl<number>, enlLocal, or enlAuto when it is measured (the measured value
    is in the file's tags).
    """
    if enl is None:
        enl_text = "Auto"
    elif enl == "local":
        enl_text = "Local"
    else:
        enl_text = f"{enl:g}"
    settings = f"ml{multilook}x{multilook}_{METHOD_SUFFIXES[method]}"
    if method == "lee":
        settings += f"_win{window}"
    settings += f"_enl{enl_text}"

    aoi = next((a for a in SEARCH_AOIS if stem.endswith(f"_{a}")), None)
    if aoi:
        return f"{stem.removesuffix(f'_{aoi}')}_{settings}_{aoi}"
    return f"{stem}_{settings}"


def filter_geotiff(
    tif_path: Path | str,
    method: str = "refined",
    enl: float | str | None = None,
    window: int = 7,
    multilook: int = 1,
    out_dir: Path | str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Speckle filter a converted backscatter GeoTIFF (NISAR GCOV/GSLC or PALSAR-2 gamma0, linear or dB) and write
    it next to the input unless out_dir is given, with the settings in its name (see output_name), e.g.
    <name>_ml1x1_refinedLee_enlAuto.tif.

    method is "refined" (7x7, keeps edges) or "lee" (square window of `window` pixels). enl is the scene's
    equivalent number of looks; None measures it from the scene, and "local" makes the refined filter estimate
    the speckle level per pixel instead. multilook > 1 first averages blocks of that many pixels a side, so the
    output's pixels are that many times larger (see multilook_grid for where they fall).
    """
    tif_path = Path(tif_path)
    stem = tif_path.stem

    out_dir = Path(out_dir) if out_dir else tif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{output_name(stem, method, enl, window, multilook)}.tif"
    if out_path.exists() and not overwrite:
        print(f"    Skipping {out_path.name} (already exists)")
        return out_path

    # NISAR products other than GSLC/GCOV (coherence, soil moisture) aren't backscatter
    if stem.startswith("NISAR_"):
        product = stem.split("_")[3]
        if not NISAR_PROCESSING_INFO.get(product, {}).get("db_accepted"):
            raise ValueError(
                f"speckle filtering needs backscatter (GSLC or GCOV), not {product}"
            )
    if enl == "local" and method != "refined":
        raise ValueError("--enl local is only for the refined filter")
    in_db = "_dB" in stem  # same test the viewer notebook uses

    # Write to a partial file so a failed run never looks complete
    part = out_path.with_suffix(".tif.part")
    try:
        with rasterio.open(tif_path) as src:
            if enl is None:
                enl = estimate_enl(src, in_db, multilook)
                print(f"    ENL measured at {enl:.1f}")
            if method == "refined":
                halo = 3
                filt = partial(refined_lee, enl=None if enl == "local" else enl)
            else:
                halo = window // 2
                filt = partial(lee, enl=enl, half=halo)

            _, _, height, width, transform = multilook_grid(src, multilook)
            profile = src.profile | {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "transform": transform,
                "count": 1,
                "dtype": "float32",
                "tiled": True,
                "compress": "DEFLATE",
                "predictor": 3,
                "BIGTIFF": "yes",
            }
            with rasterio.open(part, "w", **profile) as dst:
                filter_dataset(src, dst, filt, halo, in_db, multilook)
                # Source tags first (e.g. the JAXA credit on PALSAR-2 GeoTIFFs), then how this file was made
                dst.update_tags(
                    **src.tags(),
                    SPECKLE_FILTER=method,
                    SPECKLE_FILTER_ENL=str(enl),
                    SPECKLE_FILTER_MULTILOOK=str(multilook),
                )

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
        raise

    part.replace(out_path)
    return out_path


def enl_value(value: str) -> float | str:
    """argparse type for --enl."""
    if value == "local":
        return value
    enl = float(value)
    if enl <= 0:
        raise argparse.ArgumentTypeError("must be a positive number, or 'local'")
    return enl


def window_value(value: str) -> int:
    """argparse type for --window."""
    window = int(value)
    if window < 3 or window % 2 == 0:
        raise argparse.ArgumentTypeError("must be an odd number of pixels, 3 or more")
    return window


def multilook_value(value: str) -> int:
    """argparse type for --multilook."""
    multilook = int(value)
    if multilook < 1:
        raise argparse.ArgumentTypeError("must be a whole number, 1 or more")
    return multilook


def main():
    parser = argparse.ArgumentParser(
        description="Speckle filter NISAR / PALSAR-2 backscatter GeoTIFFs with a Lee filter."
    )
    parser.add_argument(
        "--scene",
        "--infile",
        dest="scene",
        nargs="+",
        required=True,
        help="One or more converted GeoTIFFs (looked up in data/nisar_geotiffs/ or data/alos2_geotiffs/).",
    )
    parser.add_argument(
        "--method",
        choices=list(METHOD_SUFFIXES),
        default="refined",
        help="refined: 7x7 edge-aligned windows, keeps edges sharp. lee: plain square window, faster "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--enl",
        type=enl_value,
        help="Equivalent number of looks of the scene, e.g. 1 for GSLC. 'local' estimates the speckle level per "
        "pixel like the Earth Engine version (refined only). Default: measured from the scene.",
    )
    parser.add_argument(
        "--window",
        type=window_value,
        default=7,
        help="Window width in pixels for --method lee (default: %(default)s).",
    )
    parser.add_argument(
        "--multilook",
        "--ml",
        type=multilook_value,
        default=1,
        metavar="N",
        help="Average N x N blocks of pixels before filtering: much less speckle, but pixels N times larger and "
        "features narrower than N pixels fainter (default: %(default)s, no averaging).",
    )
    parser.add_argument(
        "--out-dir",
        "--outdir",
        "--outpath",
        dest="out_dir",
        help="Folder to write the filtered GeoTIFFs to; created if missing. Defaults to each input's own folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Filter again even if the output already exists (e.g. after reconverting the scene).",
    )
    args = parser.parse_args()

    outputs: list[Path] = []
    failures: list[tuple[str, str]] = []
    for i, scene in enumerate(args.scene, start=1):
        name = Path(scene).name
        print(f"Filtering {i}/{len(args.scene)} {name}")
        try:
            tif = scene_geotiff(scene)
            outputs.append(
                filter_geotiff(
                    tif,
                    args.method,
                    args.enl,
                    args.window,
                    args.multilook,
                    args.out_dir,
                    args.overwrite,
                )
            )
        # OSError: missing/unreadable GeoTIFF or write failure; ValueError: not backscatter, or too little data
        except (OSError, ValueError) as exc:
            print(f"  FAILED: {exc}")
            failures.append((name, str(exc)))

    print(f"\nDone. {len(outputs)} scenes filtered, {len(failures)} failed.")
    for name, err in failures:
        print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
