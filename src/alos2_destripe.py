"""
Estimate and remove the burst stripes (ScanSAR scalloping) in a PALSAR-2 L2.2 scene.

L2.2 scenes are orthorectified and carry no beam or burst layout, so everything comes from the image:
the track direction from the swath's two long edges, and the burst period from the along-track spectrum.
The scene is resampled so the track runs down the columns; each column is then one range, and within
narrow range strips the stripe pattern is the part of the along-track median's short-scale wiggle
that repeats every burst.
"""

import warnings

import numpy as np
from rasterio.enums import Resampling
from scipy import ndimage as ndi

from utils.variables import (
    ALOS2_DESTRIPE_CLIP_DB,
    ALOS2_DESTRIPE_COMB_PERIODS,
    ALOS2_DESTRIPE_DECIMATION,
    ALOS2_DESTRIPE_PERIOD_KM,
    ALOS2_DESTRIPE_STRIP_PX,
)


def track_angle(valid: np.ndarray) -> float:
    """Angle (radians) of the satellite track from the raster's columns, fitted to the swath's long edges."""
    rows = np.flatnonzero(valid.any(axis=1))
    # Stay clear of the staggered top and bottom edges, where each beam starts and ends
    cut = len(rows) // 5
    rows = rows[cut:-cut]
    first = valid[rows].argmax(axis=1)
    last = valid.shape[1] - 1 - valid[rows, ::-1].argmax(axis=1)
    slopes = [np.polyfit(rows, edge, 1)[0] for edge in (first, last)]
    return float(np.arctan(np.mean(slopes)))


def to_track_grid(shape: tuple[int, int], angle: float):
    """
    Grid rotated so the track runs down its columns, covering the whole scene.
    Returns the (row, col) scene coordinates of every grid cell, and a function mapping
    scene (row, col) arrays to grid (row, col), for sampling back.
    """
    cos, sin = np.cos(angle), np.sin(angle)
    h, w = shape
    corners = np.array([[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]], dtype=float)
    # u runs along track, v across it
    u = corners[:, 0] * cos + corners[:, 1] * sin
    v = -corners[:, 0] * sin + corners[:, 1] * cos
    u0, v0 = np.floor(u.min()), np.floor(v.min())
    uu = np.arange(u0, np.ceil(u.max()) + 1, dtype="float32")[:, None]
    vv = np.arange(v0, np.ceil(v.max()) + 1, dtype="float32")[None, :]
    scene_coords = (uu * cos - vv * sin, uu * sin + vv * cos)

    def to_grid(r, c):
        return (r * cos + c * sin - u0, -r * sin + c * cos - v0)

    return scene_coords, to_grid


def burst_period(profiles: np.ndarray, pixel_m: float) -> float:
    """Burst period (pixels) from the strongest peak in the strips' mean along-track spectrum."""
    lo_km, hi_km = ALOS2_DESTRIPE_PERIOD_KM
    hi_px = hi_km * 1000 / pixel_m
    # Remove anything longer than the longest period searched before taking the spectrum
    detrended = profiles - ndi.median_filter(
        profiles, size=(int(2 * hi_px) | 1, 1), mode="nearest"
    )
    # Zero-padded 8x for a finer frequency grid: the stripe estimate compares bursts up to
    # ALOS2_DESTRIPE_COMB_PERIODS away, so the period has to be good to a fraction of a pixel
    n = 8 * len(profiles)
    freq = np.fft.rfftfreq(n)
    with np.errstate(divide="ignore"):
        period_km = 1 / freq * pixel_m / 1000
    search = (period_km >= lo_km) & (period_km <= hi_km)
    power = (np.abs(np.fft.rfft(detrended, n=n, axis=0)[search]) ** 2).mean(axis=1)
    # Peak relative to the background around it, which rises towards long periods (land, weather)
    # and can otherwise outweigh faint stripes, as in HV
    power /= ndi.median_filter(power, size=201, mode="nearest")
    return float(1 / freq[search][power.argmax()])


def burst_stripes(db: np.ndarray, pixel_m: float) -> np.ndarray:
    """
    Stripe pattern (dB) of a downsampled whole-scene dB array (NaN = nodata), on the same grid.
    Subtracting it in dB (dividing by 10**(stripes / 10) in linear power) removes the stripes.
    """
    valid = np.isfinite(db)
    # Edge pixels mix data with nodata when downsampled
    valid = ndi.binary_erosion(valid, iterations=2)
    angle = track_angle(valid)
    scene_coords, to_grid = to_track_grid(db.shape, angle)

    # Onto the track grid, dropping cells that are partly outside the swath
    rot = ndi.map_coordinates(np.where(valid, db, 0), scene_coords, order=1, cval=0)
    inside = ndi.map_coordinates(valid.astype("float32"), scene_coords, order=1, cval=0)
    rot[inside < 0.99] = np.nan

    # Each column is now a range line; flatten it to its median, and leave slicks, ships and land out
    # Columns and strips outside the swath are all NaN; nanmedian warns about each one
    with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
        rel = rot - np.nanmedian(rot, axis=0, keepdims=True)
    masked = np.where(np.abs(rel) > ALOS2_DESTRIPE_CLIP_DB, np.nan, rel)

    # Along-track median profile of each range strip. Rows where the strip is mostly masked
    # (land, big slicks) are left as gaps, so a few edge pixels can't pass for a stripe.
    strip = ALOS2_DESTRIPE_STRIP_PX
    starts = np.arange(0, masked.shape[1], strip)
    profiles, centers = [], []
    for s in starts:
        block = masked[:, s : s + strip]
        with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
            profile = np.nanmedian(block, axis=1)
        profile[np.isfinite(block).mean(axis=1) < 0.5] = np.nan
        if np.isfinite(profile).sum() > 1:
            profiles.append(profile)
            centers.append(s + block.shape[1] / 2 - 0.5)
    profiles = np.stack(profiles, axis=1)
    # Gaps bridged linearly, only for the period search and the running median
    rows = np.arange(len(profiles))
    filled = np.stack(
        [np.interp(rows, rows[np.isfinite(p)], p[np.isfinite(p)]) for p in profiles.T],
        axis=1,
    )

    period = burst_period(filled, pixel_m)
    print(
        f"    Destripe: track {np.degrees(angle):.1f} deg from columns, "
        f"burst period {period * pixel_m / 1000:.1f} km"
    )
    # Short-scale wiggle: what's left after a running median over two burst periods
    wiggle = profiles - ndi.median_filter(
        filled, size=(int(2 * period) | 1, 1), mode="nearest"
    )
    # Stripes repeat every burst period and real features don't, so the stripe at each row is the
    # median of the wiggle at that row and the same point in the bursts on either side
    k = ALOS2_DESTRIPE_COMB_PERIODS
    shifted = []
    for shift in np.arange(-k, k + 1) * period:
        at = rows + shift
        s = _lerp(wiggle, at, axis=0)
        s[(at < 0) | (at > len(rows) - 1)] = np.nan
        shifted.append(s)
    with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
        stripes = np.nan_to_num(np.nanmedian(shifted, axis=0))

    # Interpolate between strip centers across range, so there's no step at each strip edge
    cols = np.arange(masked.shape[1])
    grid = _lerp(stripes, np.interp(cols, centers, np.arange(len(centers))), axis=1)

    # Back onto the scene grid; 0 (no correction) outside the swath
    rows, cols = np.indices(db.shape, dtype="float32")
    out = ndi.map_coordinates(grid, to_grid(rows, cols), order=1, cval=0)
    out[~np.isfinite(db)] = 0
    return out.astype("float32")


def scene_stripes(src) -> np.ndarray:
    """
    Stripe pattern (dB) of a whole L2.2 scene (an open DN GeoTIFF), on the scene grid downsampled by
    ALOS2_DESTRIPE_DECIMATION. Always the whole scene, even for a crop: a crop is too short along
    track, and may not reach the swath edges, to estimate the stripes from.
    """
    shape = (
        src.height // ALOS2_DESTRIPE_DECIMATION,
        src.width // ALOS2_DESTRIPE_DECIMATION,
    )
    dn = src.read(1, out_shape=shape, resampling=Resampling.average).astype("float32")
    # dB up to the calibration constant, which doesn't change the stripe pattern; DN 0 is nodata
    with np.errstate(divide="ignore"):
        db = np.where(dn > 0, 20 * np.log10(dn), np.nan)
    pixel_m = abs(src.transform.a) * src.width / shape[1]
    return burst_stripes(db, pixel_m)


def _lerp(a: np.ndarray, coords: np.ndarray, axis: int) -> np.ndarray:
    """Linearly interpolate a along one axis at fractional indices (clamped to its ends)."""
    n = a.shape[axis]
    coords = np.clip(coords, 0, n - 1)
    i0 = np.minimum(np.floor(coords).astype(int), max(n - 2, 0))
    f = (coords - i0).astype("float32")
    lo, hi = np.take(a, i0, axis=axis), np.take(a, np.minimum(i0 + 1, n - 1), axis=axis)
    f = f[:, None] if axis == 0 else f[None, :]
    return lo * (1 - f) + hi * f


def window_stripes(
    stripes: np.ndarray, scene_shape: tuple[int, int], window
) -> np.ndarray:
    """Stripe pattern (dB) upsampled from scene_stripes' grid onto one full-resolution window of the scene."""
    sy = scene_shape[0] / stripes.shape[0]
    sx = scene_shape[1] / stripes.shape[1]
    # Centers of the window's pixels, in coarse-grid pixel indices
    rows = (window.row_off + np.arange(window.height) + 0.5) / sy - 0.5
    cols = (window.col_off + np.arange(window.width) + 0.5) / sx - 0.5
    return _lerp(_lerp(stripes, rows, axis=0), cols, axis=1)
