import argparse
from pathlib import Path

import polars as pl
from pyreqwest.client import SyncClientBuilder
from pyreqwest.exceptions import RequestError

from utils.variables import ALOS2_GRANULE_CSV_PATH, ALOS2_TIF_PATH

# Granule CSV column holding each file's URL
URL_COLUMNS = {"HH": "URL", "HV": "HV URL", "MSK": "MSK URL"}


def read_df(granule_csv: str):
    # Use path as given, look in ALOS2_GRANULE_CSV_PATH otherwise
    granule_path = Path(granule_csv)
    if not granule_path.exists():
        granule_path = ALOS2_GRANULE_CSV_PATH / granule_path
    return pl.read_csv(granule_path)


def file_name(download_url: str) -> str:
    """e.g. ALOS2585563700-250327_WBDR2.2GUD_HH_SLP (no .tif)."""
    return download_url.rsplit("/", 1)[-1].removesuffix(".tif")


def download_scene(client, granule_name: str, download_url: str) -> Path:
    outfile_path = Path(ALOS2_TIF_PATH) / f"{granule_name}.tif"

    if outfile_path.exists():
        print(f"    Skipping {granule_name} (already exists)")
        return outfile_path

    # Define path for partial file
    part = outfile_path.with_suffix(".tif.part")
    # JAXA's bucket is public, so no token is needed
    req = client.get(download_url).build_streamed()

    with req as resp:
        if resp.status != 200:
            raise RuntimeError(f"{resp.status} for {granule_name}")
        expected = int(resp.get_header("content-length") or 0)

        reader = resp.body_reader
        with open(part, "wb") as f:
            while (chunk := reader.read(1 << 20)) is not None:
                f.write(chunk)

    if expected and part.stat().st_size != expected:
        part.unlink()
        raise RuntimeError(f"size mismatch for {granule_name}")

    part.replace(outfile_path)
    return outfile_path


def download_scenes(
    granules: pl.DataFrame,
    polarizations: list[str] = ("HH", "HV"),
    mask: bool = False,
):
    # Ensure no duplicates are included
    scenes = granules.unique(subset="Granule Name", keep="first", maintain_order=True)

    # One file per scene per polarization (HH-only scenes have no HV URL), plus the mask if asked for
    files = list(polarizations) + (["MSK"] if mask else [])
    downloads = [
        (file_name(row[URL_COLUMNS[f]]), row[URL_COLUMNS[f]])
        for row in scenes.iter_rows(named=True)
        for f in files
        if row[URL_COLUMNS[f]]
    ]

    # Check for any files already downloaded, to avoid re-downloading
    existing_downloads = {p.stem for p in ALOS2_TIF_PATH.glob("*.tif")}
    pending_downloads = [d for d in downloads if d[0] not in existing_downloads]

    skipped = len(downloads) - len(pending_downloads)
    print(
        f"{scenes.height} scenes / {len(downloads)} files requested, {skipped} already on disk, {len(pending_downloads)} to download"
    )

    failures: list[tuple[str, str]] = []
    with SyncClientBuilder().build() as client:
        for i, (granule_name, url) in enumerate(pending_downloads, start=1):
            print(f"Downloading {i}/{len(pending_downloads)} {granule_name}")
            try:
                download_scene(client, granule_name, url)

            # RequestError: connection, timeout, or other network failure from pyreqwest
            # RuntimeError: non-200 status or size mismatch (raised in download_scene)
            # OSError: couldn't write the file to disk
            except (RequestError, RuntimeError, OSError) as exc:
                print(f"  FAILED: {exc}")
                failures.append((granule_name, str(exc)))

    print(
        f"\nDone. {len(pending_downloads) - len(failures)} downloaded, {len(failures)} failed."
    )
    for granule_name, err in failures:
        print(f"  {granule_name}: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PALSAR-2 L2.2 GeoTIFFs from JAXA's S3 bucket based on the granules listed in specified csv."
    )

    parser.add_argument(
        "--granule-file",
        "--gf",
        "--granule_file",
        dest="granule_file",
        required=True,
        help=(
            "CSV listing the granules to download; a bare filename is looked up in data/alos2_granule_files/."
        ),
    )
    parser.add_argument(
        "--granule",
        nargs="+",
        help="Only download these granule names from the CSV.",
    )
    parser.add_argument(
        "--polarization",
        "--pol",
        nargs="+",
        default=["HH", "HV"],
        type=str.upper,
        choices=["HH", "HV"],
        help="Which polarizations to download (default: HH HV). HV is skipped for HH-only scenes.",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Also download each scene's MSK (layover/shadow) file, needed for convert --mask.",
    )
    args = parser.parse_args()

    granules = read_df(args.granule_file)
    if args.granule:
        granules = granules.filter(pl.col("Granule Name").is_in(args.granule))
        missing = set(args.granule) - set(granules["Granule Name"])
        if missing:
            print(f"Not in CSV: {', '.join(sorted(missing))}")
    download_scenes(granules, args.polarization, args.mask)


if __name__ == "__main__":
    main()
