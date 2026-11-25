from __future__ import annotations  # self-referencing type hints (CdseAuth) on Python < 3.14

import argparse
import json
import math
import threading
import time
import xml.etree.ElementTree as ET
from datetime import timedelta
from pathlib import Path

import polars as pl
from pyreqwest.client import SyncClientBuilder
from pyreqwest.exceptions import RequestError

from sar_toolbox.utils.variables import (
    CDSE_CATALOGUE_URL,
    CDSE_DOWNLOAD_URL,
    CDSE_PASSWORD,
    CDSE_TOKEN_URL,
    CDSE_USERNAME,
    S1WV_GRANULE_CSV_PATH,
    S1WV_MAX_PAIRING_OFFSET_KM,
    S1WV_SAFE_PATH,
)

REDIRECTS = (301, 302, 303, 307, 308)


class CdseAuth:
    """
    One CDSE login, shared by every download. CDSE locks an account for a while after many logins in quick
    succession, so the access token (~10 min) is reused and renewed with its refresh token rather than
    logging in again for each file. A failed login isn't retried, so a bad password costs one attempt.
    state()/from_state() hand the tokens to worker processes, which then only ever refresh.
    """

    def __init__(self, tokens: dict | None = None):
        self.tokens = tokens or {
            "access": None,
            "access_expires": 0.0,
            "refresh": None,
            "refresh_expires": 0.0,
        }
        self.login_error: str | None = None
        self.lock = threading.Lock()

    def state(self) -> dict:
        self.token()  # make sure there's a token to hand over
        return dict(self.tokens)

    @classmethod
    def from_state(cls, tokens: dict) -> CdseAuth:
        return cls(dict(tokens))

    def _request(self, client, data: dict) -> bool:
        resp = (
            client.post(CDSE_TOKEN_URL)
            .form({"client_id": "cdse-public", **data})
            .build()
            .send()
        )
        if resp.status != 200:
            # Only a failed password login blocks retries; a failed refresh just falls back to logging in
            if data["grant_type"] == "password":
                self.login_error = (
                    f"CDSE login failed ({resp.status}): {resp.text()[:200]}"
                )
            return False
        t, now = resp.json(), time.time()
        self.tokens = {
            "access": t["access_token"],
            "access_expires": now + t["expires_in"],
            "refresh": t.get("refresh_token"),
            "refresh_expires": now + t.get("refresh_expires_in", 0),
        }
        return True

    def token(self) -> str:
        with self.lock:
            now = time.time()
            if self.tokens["access"] and now < self.tokens["access_expires"] - 60:
                return self.tokens["access"]
            with SyncClientBuilder().timeout(timedelta(seconds=60)).build() as client:
                refresh = {
                    "grant_type": "refresh_token",
                    "refresh_token": self.tokens["refresh"],
                }
                if (
                    self.tokens["refresh"]
                    and now < self.tokens["refresh_expires"] - 60
                    and self._request(client, refresh)
                ):
                    return self.tokens["access"]
                if self.login_error:
                    raise RuntimeError(self.login_error)
                if not CDSE_USERNAME or not CDSE_PASSWORD:
                    raise RuntimeError(
                        "CDSE_USERNAME / CDSE_PASSWORD are not set; add them to .env"
                    )
                if not self._request(
                    client,
                    {
                        "grant_type": "password",
                        "username": CDSE_USERNAME,
                        "password": CDSE_PASSWORD,
                    },
                ):
                    raise RuntimeError(self.login_error)
            return self.tokens["access"]


def read_df(granule_csv: str):
    # Use path as given, look in S1WV_GRANULE_CSV_PATH otherwise
    granule_path = Path(granule_csv)
    if not granule_path.exists():
        granule_path = S1WV_GRANULE_CSV_PATH / granule_path
    return pl.read_csv(granule_path)


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def node_url(product_id: str, product_name: str, *path: str) -> str:
    """URL of a file or folder inside a product's .SAFE."""
    return (
        f"{CDSE_DOWNLOAD_URL}/Products({product_id})/Nodes({product_name}.SAFE)"
        + "".join(f"/Nodes({p})" for p in path)
    )


def product_dir(product_name: str) -> Path:
    return S1WV_SAFE_PATH / product_name


def download_node(
    client, auth: CdseAuth, product_id: str, product_name: str, path: list[str]
) -> Path:
    """
    Download one file from inside the product to the same path under S1WV_SAFE_PATH/<product>, skipping it
    if already on disk. CDSE redirects downloads to another host, and HTTP clients drop the Authorization
    header on cross-host redirects, so redirects are followed by hand (the client must not follow them).
    """
    out_path = product_dir(product_name).joinpath(*path)
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")

    url = node_url(product_id, product_name, *path) + "/$value"
    for _ in range(10):
        with client.get(url).bearer_auth(auth.token()).build_streamed() as resp:
            if resp.status in REDIRECTS:
                url = resp.get_header("location")
                continue
            if resp.status != 200:
                raise RuntimeError(f"{resp.status} for {product_name}/{'/'.join(path)}")
            expected = int(resp.get_header("content-length") or 0)
            reader = resp.body_reader
            with open(part, "wb") as f:
                while (chunk := reader.read(1 << 20)) is not None:
                    f.write(chunk)
            break
    else:
        raise RuntimeError(f"too many redirects for {product_name}/{'/'.join(path)}")

    if expected and part.stat().st_size != expected:
        part.unlink()
        raise RuntimeError(f"size mismatch for {product_name}/{'/'.join(path)}")
    part.replace(out_path)
    return out_path


def read_annotation(path: Path) -> dict:
    """What the converter needs from a vignette's annotation XML: centre, swath, spacing and tie points."""
    root = ET.parse(path).getroot()
    points = root.findall(".//geolocationGridPoint")
    info = root.find(".//imageAnnotation/imageInformation")
    return {
        "lon": sum(float(p.findtext("longitude")) for p in points) / len(points),
        "lat": sum(float(p.findtext("latitude")) for p in points) / len(points),
        "swath": root.findtext(".//adsHeader/swath"),
        "polarisation": root.findtext(".//adsHeader/polarisation"),
        "start": root.findtext(".//adsHeader/startTime"),
        "range_spacing_m": float(info.findtext("rangePixelSpacing")),
        "azimuth_spacing_m": float(info.findtext("azimuthPixelSpacing")),
        "incidence_deg": float(info.findtext("incidenceAngleMidSwath")),
        # Tie points: image line and pixel, and where they are on the ground
        "tie_points": [
            {
                k: float(p.findtext(k))
                for k in ("line", "pixel", "latitude", "longitude", "height")
            }
            for p in points
        ],
    }


def annotation_path(
    client, auth, product_id: str, product_name: str, tiff_name: str
) -> Path:
    return download_node(
        client,
        auth,
        product_id,
        product_name,
        ["annotation", tiff_name.removesuffix(".tiff") + ".xml"],
    )


def check_online(client, product_name: str, product_id: str) -> None:
    resp = client.get(f"{CDSE_CATALOGUE_URL}/Products({product_id})").build().send()
    if resp.status != 200:
        raise RuntimeError(
            f"{resp.status} looking up {product_name} in the CDSE catalogue"
        )
    if not resp.json()["Online"]:
        raise RuntimeError(f"{product_name} is offline at CDSE; try again later")


def resolve_vignette(client, auth: CdseAuth, row: dict) -> str:
    """
    Measurement file name for one vignette (one row of the search CSV), cached as vignette_<n>.json in the
    product folder. Measurement files sorted by start time line up with the footprint's vignettes; the
    vignette's annotation confirms the pairing, and if it's off, every vignette is located from its annotation.
    """
    name, product_id, vignette = row["Granule Name"], row["Product Id"], row["Vignette"]
    cache = product_dir(name) / f"vignette_{vignette:03d}.json"
    if cache.exists():
        return json.loads(cache.read_text())["file"]

    check_online(client, name, product_id)
    resp = (
        client.get(node_url(product_id, name, "measurement") + "/Nodes").build().send()
    )
    if resp.status != 200:
        raise RuntimeError(f"{resp.status} listing {name}/measurement")
    # File names are <mission>-<swath>-slc-<pol>-<start>-<stop>-..., so the 5th field is the start time
    files = sorted(
        (n["Name"] for n in resp.json()["result"] if n["Name"].endswith(".tiff")),
        key=lambda f: f.split("-")[4],
    )

    paired = vignette < len(files)
    if paired:
        file = files[vignette]
        ann = read_annotation(annotation_path(client, auth, product_id, name, file))
        paired = (
            haversine_km(row["Center Lon"], row["Center Lat"], ann["lon"], ann["lat"])
            <= S1WV_MAX_PAIRING_OFFSET_KM
        )
    if not paired:
        print(
            f"    Footprint and file order disagree; locating all {len(files)} vignettes from their annotations"
        )
        dists = []
        for f in files:
            ann = read_annotation(annotation_path(client, auth, product_id, name, f))
            dists.append(
                haversine_km(
                    row["Center Lon"], row["Center Lat"], ann["lon"], ann["lat"]
                )
            )
        file = files[dists.index(min(dists))]

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"product_id": product_id, "file": file}))
    return file


def download_vignette(client, auth: CdseAuth, row: dict) -> Path:
    """
    Download one vignette's measurement GeoTIFF (complex int16, radar geometry, ~85-110 MB) and its
    annotation and calibration XMLs, rather than the whole multi-GB product. Returns the measurement file's
    path; the XMLs sit beside it in the .SAFE layout, where s1wv_convert_to_tiff finds them.
    """
    name, product_id = row["Granule Name"], row["Product Id"]
    tiff = resolve_vignette(client, auth, row)
    stem = tiff.removesuffix(".tiff")
    annotation_path(client, auth, product_id, name, tiff)
    download_node(
        client,
        auth,
        product_id,
        name,
        ["annotation", "calibration", f"calibration-{stem}.xml"],
    )
    measurement = product_dir(name) / "measurement" / tiff
    if measurement.exists():
        print(f"    Skipping {tiff} (already exists)")
        return measurement
    return download_node(client, auth, product_id, name, ["measurement", tiff])


def cdse_client():
    # Redirects are followed by hand in download_node, so the bearer token reaches the redirected host
    return (
        SyncClientBuilder()
        .timeout(timedelta(seconds=600))
        .follow_redirects(False)
        .build()
    )


def download_vignettes(granules: pl.DataFrame) -> list[Path]:
    # Ensure no duplicates are included
    vignettes = granules.unique(
        subset=["Granule Name", "Vignette"], keep="first", maintain_order=True
    )
    print(f"{vignettes.height} vignettes requested")

    auth = CdseAuth()
    paths: list[Path] = []
    failures: list[tuple[str, str]] = []
    with cdse_client() as client:
        for i, row in enumerate(vignettes.iter_rows(named=True), start=1):
            label = f"{row['Granule Name']} vignette {row['Vignette']}"
            print(f"Downloading {i}/{vignettes.height} {label}")
            try:
                paths.append(download_vignette(client, auth, row))

            # RequestError: connection, timeout, or other network failure from pyreqwest
            # RuntimeError: login failure, non-200 status, offline product or size mismatch
            # OSError: couldn't write the file to disk
            # ET.ParseError: truncated or invalid annotation XML
            except (RequestError, RuntimeError, OSError, ET.ParseError) as exc:
                print(f"  FAILED: {exc}")
                failures.append((label, str(exc)))

    print(f"\nDone. {len(paths)} downloaded, {len(failures)} failed.")
    for label, err in failures:
        print(f"  {label}: {err}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-1 WV vignettes from CDSE based on the vignettes listed in specified csv."
    )
    parser.add_argument(
        "--granule-file",
        "--gf",
        "--granule_file",
        dest="granule_file",
        required=True,
        help=(
            "CSV listing the vignettes to download; a bare filename is looked up in data/s1wv_granule_files/."
        ),
    )
    parser.add_argument(
        "--granule",
        nargs="+",
        help="Only download vignettes from these granule (product) names in the CSV.",
    )
    args = parser.parse_args()

    granules = read_df(args.granule_file)
    if args.granule:
        granules = granules.filter(pl.col("Granule Name").is_in(args.granule))
        missing = set(args.granule) - set(granules["Granule Name"])
        if missing:
            print(f"Not in CSV: {', '.join(sorted(missing))}")
    download_vignettes(granules)


if __name__ == "__main__":
    main()
