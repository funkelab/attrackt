import logging
from argparse import ArgumentParser

import yaml

from scripts.create_csv import create_csv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def run_all_from_yaml(yaml_path: str):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    seg_dirs = config["segmentation_dir_names"]
    track_files = config["man_track_file_names"]
    output_files = config["output_csv_file_names"]

    assert len(seg_dirs) == len(track_files) == len(output_files), "Mismatched lengths."

    for seg_dir, track_file, output_file in zip(seg_dirs, track_files, output_files):
        logger.info(f"Processing: {output_file}")
        create_csv(
            segmentation_dir_name=seg_dir,
            zarr_container_name=None,
            zarr_dataset_name=None,
            man_track_file_name=track_file,
            man_track_json_file_name=None,
            output_csv_file_name=output_file
        )

def main():
    parser = ArgumentParser(description="Create CSV files from segmentation directories.")   
    parser.add_argument('--yaml_config_file_name')
    args = parser.parse_args()
    run_all_from_yaml(args.yaml_config_file_name)



if __name__ == "__main__":
    main()
