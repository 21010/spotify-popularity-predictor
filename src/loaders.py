import logging
from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class DataLoader(ABC):

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass


class LocalCSVDataLoader(DataLoader):

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)

    def load(self) -> pd.DataFrame:
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        if not self.filepath.is_file():
            raise ValueError(f"Not a file: {self.filepath}")
        if self.filepath.suffix != ".csv":
            raise ValueError(f"Unsupported file type: {self.filepath}")

        try:
            logger.info(f"Loading data from {self.filepath}")
            return pd.read_csv(self.filepath)
        except Exception as e:
            logger.error(f"Error loading data from {self.filepath}: {e}")
            raise e


class HuggingFaceCSVDataLoader(DataLoader):

    def __init__(self, hf_uri: str):
        self.hf_uri = hf_uri

    def load(self) -> pd.DataFrame:

        if not self.hf_uri.startswith("hf://"):
            logger.warning(f"URI {self.hf_uri} might not be a valid Hugging Face URI (should start with hf://)")
        if not self.hf_uri.endswith(".csv"):
            logger.warning(f"URI {self.hf_uri} does not end with .csv, checking content type is advised.")

        try:
            logger.info(f"Loading remote data from {self.hf_uri}")
            return pd.read_csv(self.hf_uri)
        except Exception as e:
            logger.error(f"Error loading remote data: {e}")
            raise e


class DataLoaderFactory:
    @staticmethod
    def get_loader(path: str) -> DataLoader:
        if path.startswith("hf://") or path.startswith("http"):
            return HuggingFaceCSVDataLoader(path)
        else:
            return LocalCSVDataLoader(path)
