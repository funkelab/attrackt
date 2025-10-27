import argparse
import logging
from pathlib import Path
from zipfile import ZipFile

import gdown
import requests

logging.basicConfig(level=logging.INFO)


def download_file(url: str, output_path: Path) -> None:
    """Download a file from a given URL (supports both generic and Google Drive links)."""
    if "drive.google.com" in url:
        logging.info("Detected Google Drive link, using gdown...")
        gdown.download(url, str(output_path), quiet=False)
    else:
        logging.info("Downloading file from external URL...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


def extract_data(zip_url: str, data_dir: str) -> None:
    """
    Downloads and extracts a zip file from an external or Google Drive URL.

    Parameters
    ----------
    zip_url : str
        URL of the zip file (Google Drive or any external source).
    data_dir : str
        Path to the directory where data will be stored.
    """
    target_path = Path(data_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    output_zip = target_path / "data.zip"

    try:
        download_file(zip_url, output_zip)

        # Extract zip
        with ZipFile(output_zip, "r") as zfile:
            zfile.extractall(target_path)
        logging.info(f"Downloaded and extracted data to {target_path}")

    except Exception as e:
        logging.error(f"Failed to download or extract data: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract a zip file from any external or Google Drive URL."
    )
    parser.add_argument(
        "--zip_url",
        type=str,
        required=True,
        help="URL of the zip file (Google Drive or any external link)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory where data will be stored (default: ./data)",
    )

    args = parser.parse_args()
    extract_data(args.zip_url, args.data_dir)


if __name__ == "__main__":
    main()
