import os
from datetime import UTC, date, datetime
from pathlib import Path

from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# TOKEN FOR EARTH DATA LOGIN
EARTHDATA_TOKEN = os.getenv("EARTHDATA_TOKEN")
EARTHDATA_TOKEN_DATE = os.getenv("EARTHDATA_TOKEN_DATE")
EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD")

# ASF VARIABLES
ASF_API_BASE = "https://api.daac.asf.alaska.edu/services/search/param"
ASF_DATASET = "NISAR"
ASF_DEFAULT_PROCESSING_LEVEL = "GCOV"
ASF_DATA_MATURITY = "PROVISIONAL"
ASF_OUT = "csv"

# DATE AND TIME (UTC) OF DATA REQUEST
REQUEST_DATE = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")

# REPO STRUCTURE
REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_PATH = REPO_ROOT / "archive"
DATA_DIRECTORY = REPO_ROOT / "data"
GRANULE_CSV_PATH = DATA_DIRECTORY / "nisar_granule_files"
NISAR_GEOTIFF_PATH = DATA_DIRECTORY / "nisar_geotiffs"
NISAR_H5_PATH = DATA_DIRECTORY / "nisar_h5_scenes"

ALOS2_GRANULE_CSV_PATH = DATA_DIRECTORY / "alos2_granule_files"
ALOS2_TIF_PATH = DATA_DIRECTORY / "alos2_tif_scenes"
ALOS2_GEOTIFF_PATH = DATA_DIRECTORY / "alos2_geotiffs"

# TESTING VARIABLES
TEST_URL = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GCOV_PROVISIONAL_V1/NISAR_L2_PR_GCOV_029_026_D_069_4005_DHDH_A_20260827T001756_20260827T001830_P05023_N_F_J_001/NISAR_L2_PR_GCOV_029_026_D_069_4005_DHDH_A_20260827T001756_20260827T001830_P05023_N_F_J_001.h5"
TEST_XML_URL = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GCOV_PROVISIONAL_V1/NISAR_L2_PR_GCOV_029_026_D_069_4005_DHDH_A_20260827T001756_20260827T001830_P05023_N_F_J_001/NISAR_L2_PR_GCOV_029_026_D_069_4005_DHDH_A_20260827T001756_20260827T001830_P05023_N_F_J_001.h5.iso.xml"

# GCOV grid pixel size per frequency (from the product files)
NISAR_GCOV_PIXEL_SIZE_M = {"A": 10, "B": 80}
# Nominal GCOV frame width/height, for sizing uncropped reads
NISAR_GCOV_SCENE_SIZE_M = 335_000
# Maximum number of pixels to read remotely from NISAR's ASF bucket (per polarization file)
NISAR_REMOTE_MAX_PIXELS = 60_000_000

# NISAR PROCESSING DATA (where each product's data lives inside the .h5, how to name the output, and its nodata value)
NISAR_PROCESSING_INFO = {
    "GSLC": {
        "grid": "/science/LSAR/GSLC/grids/frequency{frequency}",
        "dataset": "{polarization}",
        "suffix": "freq{frequency}_{polarization}_intensity",
        "default_polarization": "HH",  # HH or HV
        "nodata": float("nan"),
        "db_accepted": True,
    },
    "GCOV": {
        "grid": "/science/LSAR/GCOV/grids/frequency{frequency}",
        "dataset": "{polarization}",
        "suffix": "freq{frequency}_{polarization}",
        "default_polarization": "HHHH",  # HHHH or HVHV
        "nodata": float("nan"),
        "db_accepted": True,
    },
    "GUNW": {
        "grid": "/science/LSAR/GUNW/grids/frequency{frequency}/wrappedInterferogram",
        "dataset": "{polarization}/coherenceMagnitude",
        "suffix": "freq{frequency}_{polarization}_coherence",
        "default_polarization": "HH",  # HH or HV
        "nodata": float("nan"),
        "db_accepted": False,
    },
    "SME2": {
        "grid": "/science/LSAR/SME2/grids",
        "dataset": "soilMoisture",
        "suffix": "soilMoisture",
        "default_polarization": None,
        "nodata": -9999.0,
        "db_accepted": False,
    },
}

# Define Test AOIs, always available
SEARCH_AOIS = {
    "wva-test-area": {"bbox": "-82.233966,37.994819,-81.8204,38.254771"},
    "drc-test-area": {"bbox": "11.173938,-6.499376,11.718339,-5.954592"},
    "indian-ocean-test-area": {"bbox": "87.547812,5.546815,88.452932,6.45293"},
    "leaky-infrastructure-70111": {"point": "5.312,4.803", "buffer_km": 20},
    # Timor Sea, off NW Australia: a Sentinel-1 WV1 vignette repeats here every 12 days (S1A, 2025)
    "s1wv-timor-sea-test-area": {"point": "116.596,-12.772", "buffer_km": 10},
}

# Import AOIs from git-ignored file (utils/aois.py); same names override the test ones
try:
    from utils.aois import SEARCH_AOIS as LOCAL_AOIS

    SEARCH_AOIS |= LOCAL_AOIS
except ImportError:
    pass

# AOIs given as "point" (lon,lat) + "buffer_km" get their bbox from the point; it takes precedence over any "bbox"
if any("point" in aoi for aoi in SEARCH_AOIS.values()):
    # geopandas is slow to import, so only load it when an AOI needs it
    from utils.generate_aoi_details import point_to_bbox

    for name, aoi in SEARCH_AOIS.items():
        if "point" in aoi:
            if "buffer_km" not in aoi:
                raise ValueError(f'AOI "{name}" has a "point" but no "buffer_km"')
            lon, lat = (float(v) for v in aoi["point"].split(","))
            aoi["bbox"] = point_to_bbox(lon, lat, aoi["buffer_km"])

for name, aoi in SEARCH_AOIS.items():
    if "bbox" not in aoi:
        raise ValueError(f'AOI "{name}" needs a "bbox", or a "point" and "buffer_km"')


# JAXA VARIABLES
# PALSAR-2 ScanSAR L2.2 on JAXA's public S3 bucket (no credentials needed)
JAXA_L22_BUCKET_URL = "https://jaxaalos2.s3.us-west-2.amazonaws.com"
JAXA_L22_PREFIX = "palsar2/L2.2/2025/"  # full global archive, Aug 2014 to present
# Orbit number grows linearly with time (207 revolutions per 14-day repeat); fit is within +/-9 orbits
ALOS2_ORBIT_REF_DATE = date(2014, 8, 4)
ALOS2_ORBIT_REF = 1069
ALOS2_ORBITS_PER_DAY = 207 / 14
ALOS2_ORBIT_PAD = 15
ALOS2_CAL_DB = -83.0  # gamma0 dB = 10*log10(DN^2) - 83
ALOS2_MODES = ["WBD", "WWD", "WBS", "WWS", "VBD"]
ALOS2_HH_ONLY_MODES = {"WBS", "WWS"}
ALOS2_MASK_DROP = (2, 3, 5)  # MSK values: 2 layover, 3 shadow, 5 invalid
ALOS2_ATTRIBUTION = "©JAXA"
# L2.2 GeoTIFF pixel size
ALOS2_PIXEL_SIZE_M = 25
# Largest crop (output pixels per HH/HV file) that alos2_access reads remotely; bigger crops download whole files.
# Each polarization is its own file, so both routes scale with the number of polarizations and the limit is per file.
# Timed at ~24 MB/s from JAXA's bucket: remote ~1.3 s + 0.42 s per Mpx vs ~17 s download + 0.18 s per Mpx convert,
# so they break even near 65 Mpx per file. 50 Mpx (~175 km box at 25 m) leaves margin for faster downloads.
ALOS2_REMOTE_MAX_PIXELS = 50_000_000


# SENTINEL-1 WAVE MODE (WV) VARIABLES
# WV SLC vignettes from the Copernicus Data Space Ecosystem (CDSE)
CDSE_USERNAME = os.getenv("CDSE_USERNAME")
CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")
CDSE_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
CDSE_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
# CDSE allows at most 4 concurrent downloads per user
CDSE_MAX_DOWNLOADS = 4

S1WV_GRANULE_CSV_PATH = DATA_DIRECTORY / "s1wv_granule_files"
S1WV_SAFE_PATH = (
    DATA_DIRECTORY / "s1wv_safe_files"
)  # per-product folders, laid out like the .SAFE
S1WV_GEOTIFF_PATH = DATA_DIRECTORY / "s1wv_geotiffs"

S1WV_PRODUCT_TYPE = "WV_SLC__1S"  # WV is only distributed as SLC (and L2 OCN), not GRD
# Multilook to about the resolution of IW GRDH (ground range, azimuth), so vignettes look like S1 GRD
S1WV_GRD_RESOLUTION_M = (20, 22)
# Pixel size of the geocoded (UTM) output; about the resolution after multilooking
S1WV_PIXEL_SIZE_M = 20
# Calibration constant IW GRDH products use to convert their DN to beta0 (radar brightness):
# beta0 = DN^2 / 474^2, the same everywhere in the image (from a GRDH calibration file, 2025). WV vignettes are
# scaled with it to "GRD-like DN", the values Cerulean's model reads from GRD.
S1_GRD_BETA0_CALIBRATION = 474.0
# Noise removal (--remove-noise) can leave values at or below zero; sigma0 is floored here (-50 dB) so dB
# output has no holes
S1WV_SIGMA0_FLOOR = 1e-5
# Optional Lee speckle filter (--lee): default window, and its equivalent number of looks (ENL) as a fraction of
# the multilook count. Neighbouring SLC pixels are correlated, so 5x5 looks act like ~9 independent ones
# (measured on a WV1 vignette in uniform areas: ENL ~6-10 against a nominal 25)
S1WV_LEE_WINDOW = 5
S1WV_LEE_ENL_FRACTION = 0.35
# A vignette's footprint (catalogue) and its annotation centre further apart than this mean the footprint
# order and the file order disagree; every vignette is then located from its annotation instead
S1WV_MAX_PAIRING_OFFSET_KM = 50
S1WV_ATTRIBUTION = "Contains modified Copernicus Sentinel data"
