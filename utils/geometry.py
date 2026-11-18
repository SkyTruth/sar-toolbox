import argparse

from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.warp import transform_bounds


def epsg_crs(value: str) -> str:
    """argparse type for --crs: an EPSG code (EPSG:4326, epsg:4326 or 4326), returned as 'EPSG:<code>'."""
    try:
        code = CRS.from_user_input(int(value) if value.isdigit() else value).to_epsg()
    except CRSError:
        code = None
    if code is None:
        raise argparse.ArgumentTypeError(
            f"not an EPSG code: {value!r} (e.g. EPSG:4326)"
        )
    return f"EPSG:{code}"


def crop_pixels(bbox: list[float], pixel_size_m: float) -> int:
    """Approximate output pixels for a lon/lat bbox, measured in the UTM zone of its center (as NISAR grids are)."""
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
    west, south, east, north = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bbox)
    return round((east - west) / pixel_size_m) * round((north - south) / pixel_size_m)
