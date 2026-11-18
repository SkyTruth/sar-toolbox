"""
One-time repo setup. Run from the repo root:

    uv run python -m utils.setup

Creates any files or directories that are missing (or needed .git-ignores). Will NOT overwrite any pre-existing versions.
- .env (copied from .env.default)
- utils/aois.py (empty SEARCH_AOIS template)
- archive/, data/nisar_granule_files/, data/nisar_h5_scenes/, data/nisar_geotiffs/
"""

import shutil

from utils.variables import (
    ALOS2_GEOTIFF_PATH,
    ALOS2_GRANULE_CSV_PATH,
    ALOS2_TIF_PATH,
    ARCHIVE_PATH,
    GRANULE_CSV_PATH,
    NISAR_GEOTIFF_PATH,
    NISAR_H5_PATH,
    REPO_ROOT,
)

AOIS_TEMPLATE = """\
# AOIs. These are git-ignored.
# Format: "aoi-name": {"bbox": "min_lon,min_lat,max_lon,max_lat"},
#     or: "aoi-name": {"point": "lon,lat", "buffer_km": 10},  (bbox computed from the point)

SEARCH_AOIS = {
}
"""


def make_env() -> None:
    env = REPO_ROOT / ".env"
    if env.exists():
        print("exists:  .env (left unchanged)")
    else:
        shutil.copy(REPO_ROOT / ".env.default", env)
        print("created: .env (add your EARTHDATA_TOKEN)")


def make_aois() -> None:
    aois = REPO_ROOT / "utils" / "aois.py"
    if aois.exists():
        print("exists:  utils/aois.py (left unchanged)")
    else:
        aois.write_text(AOIS_TEMPLATE)
        print("created: utils/aois.py (add your AOIs)")


def make_dirs() -> None:
    for folder in (
        ARCHIVE_PATH,
        GRANULE_CSV_PATH,
        NISAR_H5_PATH,
        NISAR_GEOTIFF_PATH,
        ALOS2_GRANULE_CSV_PATH,
        ALOS2_TIF_PATH,
        ALOS2_GEOTIFF_PATH,
    ):
        if folder.exists():
            print(f"exists:  {folder.relative_to(REPO_ROOT)}")
        else:
            folder.mkdir(parents=True)
            print(f"created: {folder.relative_to(REPO_ROOT)}")


def main() -> None:
    make_env()
    make_aois()
    make_dirs()


if __name__ == "__main__":
    main()
