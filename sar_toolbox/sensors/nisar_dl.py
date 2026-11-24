import argparse
from pathlib import Path

import polars as pl
from pyreqwest.client import SyncClientBuilder
from pyreqwest.exceptions import RequestError

from sar_toolbox.utils.variables import (
    EARTHDATA_TOKEN,
    GRANULE_CSV_PATH,
    NISAR_H5_PATH,
)


def read_df(granule_csv: str):
    # Use path as given, look in GRANULE_CSV_PATH otherwise
    granule_path = Path(granule_csv)
    if not granule_path.exists():
        granule_path = GRANULE_CSV_PATH / granule_path
    return pl.read_csv(granule_path)


def download_scene(client, granule_name: str, download_url: str) -> Path:
    outfile_path = Path(NISAR_H5_PATH) / f"{granule_name}.h5"

    if outfile_path.exists():
        print(f"    Skipping {granule_name} (already exists)")
        return outfile_path

    # Define path for partial file
    part = outfile_path.with_suffix(".h5.part")
    req = client.get(download_url).bearer_auth(EARTHDATA_TOKEN).build_streamed()

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


def download_scenes(granules: pl.DataFrame):
    if not EARTHDATA_TOKEN:
        raise RuntimeError("EARTHDATA_TOKEN is not set; add it to .env")

    # Ensure no duplicates are included
    scene_ids = granules.unique(
        subset="Granule Name", keep="first", maintain_order=True
    ).select("Granule Name", "URL")

    # Check for any scenes already downloaded, to avoid re-downloading
    existing_downloads = {p.stem for p in NISAR_H5_PATH.glob("*.h5")}
    pending_downloads = scene_ids.filter(
        ~pl.col("Granule Name").is_in(existing_downloads)
    )

    skipped = scene_ids.height - pending_downloads.height
    print(
        f"{scene_ids.height} scenes requested, {skipped} already on disk, {pending_downloads.height} to download"
    )

    failures: list[tuple[str, str]] = []
    with SyncClientBuilder().build() as client:
        for i, (granule_name, url) in enumerate(pending_downloads.iter_rows(), start=1):
            print(f"Downloading {i}/{pending_downloads.height} {granule_name}")
            try:
                download_scene(client, granule_name, url)

            # RequestError: connection, timeout, or other network failure from pyreqwest
            # RuntimeError: non-200 status or size mismatch (raised in download_scene)
            # OSError: couldn't write the file to disk
            except (RequestError, RuntimeError, OSError) as exc:
                print(f"  FAILED: {exc}")
                failures.append((granule_name, str(exc)))

    print(
        f"\nDone. {pending_downloads.height - len(failures)} downloaded, {len(failures)} failed."
    )
    for granule_name, err in failures:
        print(f"  {granule_name}: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NISAR Scenes from ASF based on the granule names listed in specified csv."
    )

    parser.add_argument(
        "--granule-file",
        "--gf",
        "--granule_file",
        dest="granule_file",
        required=True,
        help=(
            "CSV listing the granules to download; a bare filename is looked up in data/nisar_granule_files/."
        ),
    )
    parser.add_argument(
        "--granule",
        nargs="+",
        help="Only download these granule names from the CSV.",
    )
    args = parser.parse_args()

    granules = read_df(args.granule_file)
    if args.granule:
        granules = granules.filter(pl.col("Granule Name").is_in(args.granule))
        missing = set(args.granule) - set(granules["Granule Name"])
        if missing:
            print(f"Not in CSV: {', '.join(sorted(missing))}")
    download_scenes(granules)


if __name__ == "__main__":
    main()
