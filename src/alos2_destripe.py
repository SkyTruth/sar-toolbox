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
    ALOS2_DESTRIPE_MAX_GAIN,
    ALOS2_DESTRIPE_MIN_OCEAN,
    ALOS2_DESTRIPE_PERIOD_KM,
    ALOS2_DESTRIPE_STRIP_KM,
    ALOS2_MASK_OCEAN,
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


def burst_stripes(
    db: np.ndarray, pixel_m: float, ocean: np.ndarray | None = None
) -> np.ndarray:
    """
    Stripe pattern (dB) of a downsampled whole-scene dB array (NaN = nodata), on the same grid.
    Subtracting it in dB (dividing by 10**(stripes / 10) in linear power) removes the stripes.
    With ocean (JAXA's ocean flag on the same grid), the steps and arches across the swath, where
    the beams meet, are included too.
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

    # Each column is now a range line; flatten it to its median (removes the steps between beams)
    # Columns and strips outside the swath are all NaN; nanmedian warns about each one
    with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
        rel = rot - np.nanmedian(rot, axis=0, keepdims=True)

    # Narrow range strips, and where each column sits between their centers
    strip = round(ALOS2_DESTRIPE_STRIP_KM * 1000 / pixel_m)
    starts = np.arange(0, rel.shape[1], strip)
    centers = starts + np.minimum(strip, rel.shape[1] - starts) / 2 - 0.5
    strip_pos = np.interp(np.arange(rel.shape[1]), centers, np.arange(len(starts)))
    rows = np.arange(rel.shape[0])

    def strip_profiles(a):
        """Along-track median of each strip; rows where the strip is mostly NaN are gaps."""
        profiles = []
        for s in starts:
            block = a[:, s : s + strip]
            with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
                profile = np.nanmedian(block, axis=1)
            profile[np.isfinite(block).mean(axis=1) < 0.5] = np.nan
            profiles.append(profile)
        return np.stack(profiles, axis=1)

    def bridged(profiles):
        """Gaps bridged linearly (all-gap strips set to 0), for filtering."""
        out = np.zeros_like(profiles)
        for j, p in enumerate(profiles.T):
            ok = np.isfinite(p)
            if ok.sum() > 1:
                out[:, j] = np.interp(rows, rows[ok], p[ok])
        return out

    # First pass, nothing left out: the burst period, and each strip's local background
    # (a running median over two burst periods)
    first = bridged(strip_profiles(rel))
    period = burst_period(first, pixel_m)
    print(
        f"    Destripe: track {np.degrees(angle):.1f} deg from columns, "
        f"burst period {period * pixel_m / 1000:.1f} km"
    )
    window = (int(2 * period) | 1, 1)
    background = _lerp(
        ndi.median_filter(first, size=window, mode="nearest"), strip_pos, axis=1
    )

    # Second pass: leave out pixels far from their local background (slicks, ships, coastlines).
    # Local, so water that's brighter or darker than the open sea, like shallows, still counts.
    masked = np.where(np.abs(rel - background) > ALOS2_DESTRIPE_CLIP_DB, np.nan, rel)
    profiles = strip_profiles(masked)
    # Short-scale wiggle: what's left after a running median over two burst periods
    wiggle = profiles - ndi.median_filter(
        bridged(profiles), size=window, mode="nearest"
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
    grid = _lerp(stripes, strip_pos, axis=1)

    # That's the stripes' shape, but their strength varies from place to place (stronger on brighter
    # water, as near shore), so scale it to how much of it each neighbourhood actually contains:
    # a least-squares gain over two burst periods along track and one strip across
    used = np.isfinite(masked)
    resid = np.where(used, masked - background, 0)
    shape = np.where(used, grid, 0)
    size = (int(2 * period) | 1, strip)
    num = ndi.uniform_filter(resid * shape, size)
    den = ndi.uniform_filter(shape * shape, size)
    # Where too little is left to fit (land, big slicks), keep the shape as estimated
    enough = ndi.uniform_filter(used.astype("float32"), size) > 0.3
    gain = np.where(enough, num / np.maximum(den, 1e-6), 1.0)
    grid *= np.clip(gain, 0, ALOS2_DESTRIPE_MAX_GAIN)

    # Across the swath: the steps where the beams meet, and the arch within each beam (strongest in
    # HV). They're what the ocean's median along each range line has on top of a smooth trend; the
    # trend, the real fall-off with incidence angle, stays. Only range lines that are mostly ocean
    # count, since water near shore alone isn't typical of the whole line.
    if ocean is not None:
        sea = ndi.map_coordinates(ocean.astype("float32"), scene_coords, order=1) > 0.99
        sea &= np.isfinite(rot)
        with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
            profile = np.nanmedian(np.where(sea, rot, np.nan), axis=0)
        n_sea = sea.sum(axis=0)
        ok = (n_sea > 0) & (
            n_sea >= ALOS2_DESTRIPE_MIN_OCEAN * np.isfinite(rot).sum(axis=0)
        )
        if ok.sum() > 10:
            x = np.linspace(0, 1, len(profile))
            trend = np.polyval(np.polyfit(x[ok], profile[ok], 3), x)
            # Range lines that aren't mostly ocean take the nearest ocean line's value
            grid += np.interp(x, x[ok], (profile - trend)[ok])[None, :]

    # Back onto the scene grid; 0 (no correction) outside the swath
    rows, cols = np.indices(db.shape, dtype="float32")
    out = ndi.map_coordinates(grid, to_grid(rows, cols), order=1, cval=0)
    out[~np.isfinite(db)] = 0
    return out.astype("float32")


def scene_stripes(
    src, msk=None, decimation: int = ALOS2_DESTRIPE_DECIMATION
) -> np.ndarray:
    """
    Stripe pattern (dB) of a whole L2.2 scene (an open DN GeoTIFF), on the scene grid downsampled by
    decimation. Always the whole scene, even for a crop: a crop is too short along track, and may not
    reach the swath edges, to estimate the stripes from. msk is the scene's open _MSK.tif, whose
    ocean flag the stripes across the swath are measured on (without it, they're left).
    """
    shape = (src.height // decimation, src.width // decimation)
    dn = src.read(1, out_shape=shape, resampling=Resampling.average).astype("float32")
    # dB up to the calibration constant, which doesn't change the stripe pattern; DN 0 is nodata
    with np.errstate(divide="ignore"):
        db = np.where(dn > 0, 20 * np.log10(dn), np.nan)
    ocean = None
    if msk is not None:
        flags = msk.read(1, out_shape=shape, resampling=Resampling.nearest)
        ocean = flags == ALOS2_MASK_OCEAN
    pixel_m = abs(src.transform.a) * src.width / shape[1]
    return burst_stripes(db, pixel_m, ocean)


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
