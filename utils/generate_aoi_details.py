import geopandas as gpd
from shapely.geometry import Point

# Generate an AOI in the format accepted by SEARCH_AOIS (utils/aois.py)


def point_to_bbox(lon: float, lat: float, buffer_km: float) -> str:
    """Square buffer of buffer_km around a lon/lat point, as a "min_lon,min_lat,max_lon,max_lat" bbox."""
    # Point in WGS84, reprojected to its local UTM zone so the buffer is in metres
    gdf = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    utm_crs = gdf.estimate_utm_crs()
    buffered = gdf.to_crs(utm_crs).buffer(buffer_km * 1000, cap_style="square")

    # Back to lon/lat and take the bounding box
    min_lon, min_lat, max_lon, max_lat = buffered.to_crs(
        "EPSG:4326"
    ).total_bounds.round(6)
    return f"{min_lon},{min_lat},{max_lon},{max_lat}"


def define_aoi_from_bounds() -> dict:
    ll_coordinate = input("Define lower left coord (min lon, min lat): ").replace(
        " ", ""
    )
    ur_coordinate = input("Define upper right coord (max lon, max lat): ").replace(
        " ", ""
    )
    return {"bbox": f"{ll_coordinate},{ur_coordinate}"}


def define_aoi_from_point() -> dict:
    point = input("Define center point (lon, lat): ").replace(" ", "")
    lon, lat = (float(v) for v in point.split(","))
    buffer_km = float(input("Buffer distance from point (km): "))
    print(f"\nbbox: {point_to_bbox(lon, lat, buffer_km)}")
    return {"point": point, "buffer_km": buffer_km}


def main() -> None:
    aoi_name = input("Descriptive AOI Name:  ").strip().lower().replace(" ", "-")

    method = input("Define AOI from (1) bounds or (2) point + buffer? [1/2]: ").strip()
    if method == "2":
        aoi = define_aoi_from_point()
    else:
        aoi = define_aoi_from_bounds()

    print("\nAdd this to SEARCH_AOIS in utils/aois.py:\n")
    if "point" in aoi:
        print(
            f'    "{aoi_name}": {{"point": "{aoi["point"]}", "buffer_km": {aoi["buffer_km"]:g}}},  # lon, lat'
        )
    else:
        print(f'    "{aoi_name}": {{"bbox": "{aoi["bbox"]}"}},')


if __name__ == "__main__":
    main()
