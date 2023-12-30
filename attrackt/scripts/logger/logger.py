import csv
import logging
from pathlib import Path
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


class CSVLogger:
    def __init__(self, keys: List[str], title: Union[str, Path]):
        self.keys = keys
        self.title = Path(title).with_suffix(".csv")
        self.data: Dict[str, List[float]] = {k: [] for k in keys}
        logger.info(f"Created CSVLogger with keys: {self.keys}")

        # Create file and write header if it doesn't exist
        if not self.title.exists():
            with self.title.open(mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.keys)
                writer.writeheader()

    def add(self, key: str, value: float) -> None:
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in logger. Valid keys: {self.keys}")
        self.data[key].append(value)

    def write(self, reset: bool = False) -> None:
        rows = [dict(zip(self.keys, values)) for values in zip(*self.data.values())]
        with self.title.open(mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.keys)
            writer.writerows(rows)

        if reset:
            self.data = {k: [] for k in self.keys}
            logger.debug("Logger data reset after writing")
