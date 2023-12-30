import argparse
import logging
from pathlib import Path
from zipfile import ZipFile

import gdown

logging.basicConfig(level=logging.INFO)


def extract_data(zip_url: str, data_dir: str) -> None:
    """
    Downloads and extracts a zip file from Google Drive using gdown.

    Parameters
    ----------
    zip_url : str
        Google Drive URL of the zip file.
    data_dir : str
        Path to the directory where data will be stored.

    Returns
    -------
    None
    """
    target_path = Path(data_dir)

    if target_path.exists():
        logging.info(f"Directory already exists at {target_path}")
        return

    target_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directory {target_path}")

    try:
        output_zip = target_path / "data.zip"

        # Download with gdown
        gdown.download(zip_url, str(output_zip), quiet=False)

        # Extract zip
        with ZipFile(output_zip, "r") as zfile:
            zfile.extractall(target_path)
        logging.info(f"Downloaded and extracted data to {target_path}")

    except Exception as e:
        logging.error(f"Failed to download or extract data: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract a zip file from Google Drive."
    )
    parser.add_argument(
        "--zip_url",
        type=str,
        required=True,
        help="Google Drive file ID or URL of the zip file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory where data will be stored (default: ./data)",
    )

    args = parser.parse_args()
    extract_data(args.zip_url, args.target_path)


if __name__ == "__main__":
    main()
