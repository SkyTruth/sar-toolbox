"""
CFAR detection of bright targets (vessels, offshore infrastructure) in NISAR and PALSAR-2 backscatter.

    uv run python -m src.cfar --scene SCENE [SCENE ...]

A scene is a GeoTIFF written by nisar_convert_to_tiff / alos2_convert_to_tiff, or a downloaded NISAR .h5 or
PALSAR-2 L2.2 source .tif, which is converted first. nisar_access and alos2_access run the same detection
with --cfar.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform

from src import alos2_convert_to_tiff, nisar_convert_to_tiff
from utils.variables import (
    ALOS2_GEOTIFF_PATH,
    CFAR_BACKGROUND_M,
    CFAR_BLOCK_PIXELS,
    CFAR_GUARD_M,
    CFAR_MAX_HIT_PIXELS,
    CFAR_PFA,
    NISAR_GEOTIFF_PATH,
    NISAR_PROCESSING_INFO,
)

# Passes over each block; every pass after the first leaves the previous pass's hits out of the background
PASSES = 2
# With coarse pixels (NISAR frequency B is 80 m) keep the background ring at least this many pixels thick
MIN_RING_PX = 4
# Pixels with less than this share of their ring valid (scene edge, coast) are not tested
MIN_RING_FRACTION = 0.5


def ring_sum(a, guard: int, bg: int):
    """
    Sum of `a` over each pixel's background ring: the (2*bg+1)^2 box around it minus the (2*guard+1)^2 guard box.
    Uses an integral image, so the cost doesn't grow with the window size; outside the array counts as 0.
    """
    h, w = a.shape
    ii = np.zeros((h + 2 * bg + 1, w + 2 * bg + 1))
    ii[bg + 1 : bg + 1 + h, bg + 1 : bg + 1 + w] = a
    np.cumsum(ii, axis=0, out=ii)
    np.cumsum(ii, axis=1, out=ii)

    def box(half):
        n, trim = 2 * half + 1, bg - half
        v = ii[trim : ii.shape[0] - trim, trim : ii.shape[1] - trim]
        return v[n:, n:] - v[:-n, n:] - v[n:, :-n] + v[:-n, :-n]

    return box(bg) - box(guard)


def cfar(power, guard: int, bg: int, k: float):
    """
    Two-parameter CFAR on a 2-D array of linear power (NaN = no data): a pixel is a hit when it is more than k
    standard deviations above the mean of its background ring. guard and bg are half-widths in pixels.

    The test runs on the cube root of power. Speckled intensity is gamma-distributed, with a long bright tail
    that a mean + k*std test on the power itself keeps tripping over. Its cube root is close to normal whatever
    the number of looks (Wilson-Hilferty), so k maps to a false alarm rate through the normal quantile.

    A second pass leaves the first pass's hits out of the background, so a target isn't hidden by a brighter
    neighbor in its ring, or by its own far end.

    Returns the hit mask, each pixel's score (standard deviations above its background) and background power.
    """
    amp = np.cbrt(power)
    valid = np.isfinite(amp)
    clutter = valid
    min_count = MIN_RING_FRACTION * ((2 * bg + 1) ** 2 - (2 * guard + 1) ** 2)
    for _ in range(PASSES):
        x = np.where(clutter, amp, 0)
        n = ring_sum(clutter, guard, bg)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = ring_sum(x, guard, bg) / n
            var = ring_sum(x * x, guard, bg) / n - mean**2
            std = np.sqrt(np.clip(var, 0, None))
            score = (amp - mean) / std
        hits = valid & (n >= min_count) & (score > k)
        if not hits.any():
            break
        clutter = valid & ~hits

    # Mean power from the cube-root statistics: for gamma clutter E[x^(1/3)] sits (std/mean)^2 below mean^(1/3)
    with np.errstate(divide="ignore", invalid="ignore"):
        background = (mean + std**2 / mean) ** 3
    return hits, score, background


def label_pixels(rows, cols):
    """Group hit pixels that touch (8-connected) into targets; returns each pixel's target number, from 0."""
    index = {rc: i for i, rc in enumerate(zip(rows.tolist(), cols.tolist()))}
    parent = list(range(len(index)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (r, c), i in index.items():
        # The four neighbors after this pixel in scan order; the other four link back on their own turn
        for neighbor in ((r, c + 1), (r + 1, c - 1), (r + 1, c), (r + 1, c + 1)):
            j = index.get(neighbor)
            if j is not None:
                parent[find(i)] = find(j)

    roots = [find(i) for i in range(len(parent))]
    return np.unique(roots, return_inverse=True)[1]


def load_land(path: Path, crs, bounds, buffer_m: float):
    """
    Land polygons from a local vector file (anything geopandas reads: .shp, .gpkg, .geojson, ...) as a GeoSeries
    in the scene's CRS, limited to the scene's bounds and grown by buffer_m. None if no land is in the scene.
    """
    # geopandas is slow to import, so only load it when a land mask is used
    import geopandas as gpd
    from shapely.geometry import box

    # Padded so polygons just outside the scene still reach in once buffered
    area = gpd.GeoSeries([box(*bounds).buffer(buffer_m + 1000)], crs=crs)
    land = gpd.read_file(path, bbox=area)
    # Clip before reprojecting: a continent-sized polygon doesn't survive being put into one UTM zone
    land = land.clip(tuple(area.to_crs(land.crs).total_bounds)).geometry.to_crs(crs)
    if buffer_m:
        land = land.buffer(buffer_m)
    land = land[~land.is_empty]
    return None if land.empty else land


def to_db(power: float) -> float | None:
    """Power in dB for the GeoJSON properties; None where it's undefined, since JSON has no NaN."""
    return round(10 * math.log10(power), 2) if power > 0 else None


def detect_targets(
    tif_path: Path | str,
    pfa: float = CFAR_PFA,
    guard_m: float = CFAR_GUARD_M,
    background_m: float = CFAR_BACKGROUND_M,
    land_mask: Path | str | None = None,
    land_buffer_m: float = 0.0,
    out_dir: Path | str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Run CFAR over a converted backscatter GeoTIFF (NISAR GCOV/GSLC or PALSAR-2 gamma0, linear or dB, on the
    scene's own grid) and write the targets found as lon/lat points to <name>_cfar.geojson, next to the GeoTIFF
    unless out_dir is given.

    guard_m and background_m are full window widths in meters, converted with the GeoTIFF's pixel size.
    land_mask is a local vector file of land polygons; pixels inside them (grown by land_buffer_m) are ignored.
    """
    tif_path = Path(tif_path)
    stem = tif_path.stem

    out_dir = Path(out_dir) if out_dir else tif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_cfar.geojson"
    if out_path.exists() and not overwrite:
        print(f"    Skipping {out_path.name} (already exists)")
        return out_path

    # NISAR products other than GSLC/GCOV (coherence, soil moisture) aren't backscatter
    if stem.startswith("NISAR_"):
        product = stem.split("_")[3]
        if not NISAR_PROCESSING_INFO.get(product, {}).get("db_accepted"):
            raise ValueError(f"CFAR needs backscatter (GSLC or GCOV), not {product}")

    # Standard deviations above background that a normal variable exceeds with probability pfa
    k = -NormalDist().inv_cdf(pfa)
    in_db = "_dB" in stem  # same test the viewer notebook uses

    found = []  # per block: rows, cols, power, score, background of the hit pixels
    n_hits = 0
    with rasterio.open(tif_path) as src:
        if src.crs is None or not src.crs.is_projected:
            raise ValueError(
                f"{tif_path.name} is not on a projected (meter) grid; run CFAR on a GeoTIFF converted without --crs"
            )
        crs, transform, tags = src.crs, src.transform, src.tags()
        pixel_m = max(src.res)
        guard = max(1, round(guard_m / 2 / pixel_m))
        bg = max(guard + MIN_RING_PX, round(background_m / 2 / pixel_m))
        land = (
            load_land(Path(land_mask), crs, src.bounds, land_buffer_m)
            if land_mask
            else None
        )

        # Row blocks so a full scene never has to fit in memory, each read with enough extra rows above and
        # below that the rows kept come out the same as if the whole scene were processed at once
        halo = PASSES * bg
        step = max(256, CFAR_BLOCK_PIXELS // src.width)
        for r0 in range(0, src.height, step):
            r1 = min(r0 + step, src.height)
            top, bottom = max(0, r0 - halo), min(src.height, r1 + halo)
            win = Window(0, top, src.width, bottom - top)
            block = src.read(1, window=win, masked=True).astype("float32")
            block = block.filled(np.nan)
            if in_db:
                block = 10 ** (block / 10)
            if land is not None:
                west, south, east, north = window_bounds(win, transform)
                shapes = land.cx[west:east, south:north]
                if len(shapes):
                    on_land = rasterize(
                        shapes,
                        out_shape=block.shape,
                        transform=window_transform(win, transform),
                        dtype="uint8",
                    )
                    block[on_land == 1] = np.nan
            if not np.isfinite(block).any():
                continue

            hits, score, background = cfar(block, guard, bg, k)
            hits[: r0 - top] = False
            hits[r1 - top :] = False
            rr, cc = np.nonzero(hits)
            n_hits += rr.size
            if n_hits > CFAR_MAX_HIT_PIXELS:
                raise ValueError(
                    f"over {CFAR_MAX_HIT_PIXELS:,} pixels passed the test in {tif_path.name}, so this isn't "
                    "sea clutter; mask land with --land-mask or lower --pfa"
                )
            found.append(
                (rr + top, cc, block[rr, cc], score[rr, cc], background[rr, cc])
            )

    features = []
    if n_hits:
        rows, cols, power, score, background = (np.concatenate(a) for a in zip(*found))
        labels = label_pixels(rows, cols)
        count = np.bincount(labels)
        # Power-weighted centroid in pixel coordinates (+0.5 for pixel centers), then lon/lat
        weight = np.bincount(labels, weights=power)
        col_c = np.bincount(labels, weights=power * (cols + 0.5)) / weight
        row_c = np.bincount(labels, weights=power * (rows + 0.5)) / weight
        lons, lats = warp_transform(crs, "EPSG:4326", *(transform * (col_c, row_c)))
        # Brightest pixel of each target: last of its group when sorted by target, then power
        peak = np.lexsort((power, labels))[np.cumsum(count) - 1]
        pixel_area = abs(transform.a * transform.e)

        for i in np.argsort(-score[peak]):
            p = peak[i]
            peak_db, background_db = to_db(power[p]), to_db(background[p])
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(lons[i], 6), round(lats[i], 6)],
                    },
                    "properties": {
                        "n_pixels": int(count[i]),
                        "area_m2": round(float(count[i] * pixel_area)),
                        "peak_db": peak_db,
                        "background_db": background_db,
                        "contrast_db": None
                        if None in (peak_db, background_db)
                        else round(peak_db - background_db, 2),
                        "sigma": round(float(score[p]), 1)
                        if math.isfinite(score[p])
                        else None,
                    },
                }
            )

    collection = {
        "type": "FeatureCollection",
        # How the file was made; kept with it so detections can be traced to their settings
        "cfar": {
            "source": tif_path.name,
            "pfa": pfa,
            "guard_m": guard_m,
            "background_m": background_m,
            "guard_px": 2 * guard + 1,
            "background_px": 2 * bg + 1,
            "pixel_m": pixel_m,
            "land_mask": Path(land_mask).name if land_mask else None,
            "land_buffer_m": land_buffer_m,
        },
        "features": features,
    }
    # e.g. the JAXA credit on PALSAR-2 GeoTIFFs
    if "ATTRIBUTION" in tags:
        collection["cfar"]["attribution"] = tags["ATTRIBUTION"]
    out_path.write_text(json.dumps(collection))

    print(f"    {len(features)} targets -> {out_path.name}")
    return out_path


def scene_geotiff(
    source: Path | str, frequency: str = "B", polarization: str | None = None
) -> Path:
    """
    The GeoTIFF to run CFAR on: a converted GeoTIFF as given (a bare file name is looked up in the sensor's
    geotiff folder), or a downloaded NISAR .h5 / PALSAR-2 L2.2 source .tif converted first (whole scene, linear,
    on its own grid; skipped if that GeoTIFF already exists). frequency and polarization apply to .h5 only.
    """
    name = Path(source).name
    is_alos2 = name.startswith("ALOS2")

    # PALSAR-2 source tifs and converted GeoTIFFs are both .tif; only the source matches the L2.2 file name
    source_tif = alos2_convert_to_tiff.L22_FILE_RE.fullmatch(name.removesuffix(".tif"))
    if name.endswith(".tif") and not source_tif:
        path = Path(source)
        if not path.exists():
            path = (ALOS2_GEOTIFF_PATH if is_alos2 else NISAR_GEOTIFF_PATH) / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        return path

    if is_alos2:
        return alos2_convert_to_tiff.convert_scene_to_geotiff(source)
    return nisar_convert_to_tiff.convert_scene_to_geotiff(
        source, frequency, polarization
    )


def pfa_value(value: str) -> float:
    """argparse type for --pfa."""
    pfa = float(value)
    if not 0 < pfa < 0.5:
        raise argparse.ArgumentTypeError("must be between 0 and 0.5, e.g. 1e-6")
    return pfa


def existing_file(value: str) -> Path:
    """argparse type for --land-mask, so a wrong path fails before any scene is processed."""
    path = Path(value).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{value} not found")
    return path


def add_cfar_arguments(parser: argparse.ArgumentParser) -> None:
    """The detection settings, shared by this module's CLI and the --cfar option of nisar_access / alos2_access."""
    group = parser.add_argument_group("CFAR detection")
    group.add_argument(
        "--pfa",
        type=pfa_value,
        default=CFAR_PFA,
        help="False alarm rate per pixel the threshold is set for; lower finds fewer, surer targets (default: %(default)s).",
    )
    group.add_argument(
        "--guard-m",
        type=float,
        default=CFAR_GUARD_M,
        help="Width in meters of the guard window around each pixel, kept out of its background; should be "
        "wider than the largest target (default: %(default)s).",
    )
    group.add_argument(
        "--background-m",
        type=float,
        default=CFAR_BACKGROUND_M,
        help="Width in meters of the background window; the ring between it and the guard window gives the "
        "local sea clutter (default: %(default)s).",
    )
    group.add_argument(
        "--land-mask",
        type=existing_file,
        help="Local vector file of land polygons (.shp, .gpkg, .geojson, ...); pixels on land are ignored. "
        "Without it, land in a scene is searched too.",
    )
    group.add_argument(
        "--land-buffer-m",
        type=float,
        default=0.0,
        help="Grow the land mask by this many meters, to drop shoreline returns (default: %(default)s).",
    )


def cfar_options(args: argparse.Namespace) -> dict:
    """The add_cfar_arguments settings as keyword arguments for detect_targets."""
    return {
        "pfa": args.pfa,
        "guard_m": args.guard_m,
        "background_m": args.background_m,
        "land_mask": args.land_mask,
        "land_buffer_m": args.land_buffer_m,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect bright targets (vessels, infrastructure) in NISAR / PALSAR-2 scenes with CFAR."
    )
    parser.add_argument(
        "--scene",
        "--infile",
        dest="scene",
        nargs="+",
        required=True,
        help="One or more converted GeoTIFFs (looked up in data/nisar_geotiffs/ or data/alos2_geotiffs/), or "
        "downloaded scenes to convert first: NISAR .h5 files or granule names (data/nisar_h5_scenes/), "
        "PALSAR-2 _HH_SLP/_HV_SLP .tif files (data/alos2_tif_scenes/).",
    )
    parser.add_argument(
        "--frequency",
        "--freq",
        type=str.upper,
        choices=["A", "B"],
        default="B",
        help="For .h5 scenes. A: higher resolution, larger files. B: lower resolution, smaller (default: %(default)s).",
    )
    parser.add_argument(
        "--polarization",
        "--pol",
        type=str.upper,
        help="For .h5 scenes. GCOV: HHHH, HVHV or VVVV. GSLC: HH or HV. Defaults per product.",
    )
    parser.add_argument(
        "--out-dir",
        "--outdir",
        "--outpath",
        dest="out_dir",
        help="Folder to write the detections to; created if missing. Defaults to the GeoTIFF's own folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Detect again even if the .geojson already exists (e.g. after changing --pfa).",
    )
    add_cfar_arguments(parser)
    args = parser.parse_args()

    outputs: list[Path] = []
    failures: list[tuple[str, str]] = []
    for i, scene in enumerate(args.scene, start=1):
        name = Path(scene).name
        print(f"Detecting {i}/{len(args.scene)} {name}")
        try:
            tif = scene_geotiff(scene, args.frequency, args.polarization)
            outputs.append(
                detect_targets(
                    tif,
                    **cfar_options(args),
                    out_dir=args.out_dir,
                    overwrite=args.overwrite,
                )
            )

        # Same failure types the converters' batch loops catch (a scene may be converted here first), plus
        # ValueError for a GeoTIFF CFAR can't use (wrong product, reprojected, mostly land) and RuntimeError
        # for an unreadable land mask
        except (OSError, KeyError, ValueError, IndexError, RuntimeError) as exc:
            print(f"  FAILED: {exc}")
            failures.append((name, str(exc)))

    print(f"\nDone. {len(outputs)} scenes searched, {len(failures)} failed.")
    for name, err in failures:
        print(f"  {name}: {err}")


if __name__ == "__main__":
    main()