"""
Speckle filtering for any sensor's SAR imagery.

Functions take a 2D numpy array of linear intensity (power: |SLC|^2, sigma0, gamma0, beta0, or GRD DN^2), not
amplitude or dB, so convert before filtering and take dB afterwards. NaN marks no data: it's left out of every
window's statistics and stays NaN in the output, so scene edges don't bleed into valid pixels.

Large scenes (e.g. a full NISAR frame) can be filtered block by block: give each block a margin of
window // 2 pixels on every side, filter it, and keep only the block's interior.
"""

import numpy as np


def box_mean(a, size: int):
    """
    Mean over a size x size window centred on each pixel, via summed-area tables. NaNs are left out (a window
    of only NaNs gives NaN); edges are reflected.
    """
    r = size // 2
    valid = np.isfinite(a)
    values = np.pad(np.where(valid, a, 0).astype("float64"), r, mode="reflect")
    counts = np.pad(valid.astype("float64"), r, mode="reflect")

    def window_sums(x):
        c = np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]

    n = window_sums(counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, window_sums(values) / n, np.nan)


def lee_filter(intensity, window: int, enl: float):
    """
    Lee speckle filter. Each pixel moves from its window mean towards its own value by
    k = (local variance not explained by speckle) / (local variance), so uniform areas (e.g. flat sea) are
    smoothed towards the mean while edges and bright targets, where variance is well above speckle's, are
    mostly kept. The window mean is preserved, so calibration is unaffected.

    window: odd window size in pixels (e.g. 5 for 5x5).
    enl: the image's equivalent number of looks, i.e. how strong its speckle is (speckle's squared coefficient
    of variation for intensity is 1 / ENL); lower smooths more. It's ~1 for single-look data and below the
    nominal look count for multilooked data whose pixels are correlated, so it's best measured per product,
    e.g. as mean^2 / variance over uniform areas.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError(f"window must be odd and at least 3, not {window}")
    if enl <= 0:
        raise ValueError(f"enl must be positive, not {enl}")
    cu2 = 1 / enl
    intensity = np.asarray(intensity, dtype="float64")
    mean = box_mean(intensity, window)
    var = np.maximum(box_mean(intensity**2, window) - mean**2, 0)
    var_x = np.maximum((var - mean**2 * cu2) / (1 + cu2), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.clip(np.where(var > 0, var_x / var, 0), 0, 1)
    return mean + k * (intensity - mean)
