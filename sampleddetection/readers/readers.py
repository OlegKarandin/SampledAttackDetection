from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Any

import pandas as pd

from ..utils import setup_logger


class AbstractTimeSeriesReader(ABC):
    """
    Abstract class for readers
    Readers are defined to be obejcts that return packet-like info
    """

    @abstractmethod
    def __getitem__(self, index) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Number of individual elements inside of this object.
        """
        pass

    @abstractmethod
    def getTimestamp(self, idx) -> float:
        pass

    @property
    @abstractmethod
    def init_time(self) -> float:
        """
        Initial Timestamp
        """
        pass

    @property
    @abstractmethod
    def fin_time(self) -> float:
        """
        Final Timestamp
        """
        pass


class CSVReader(AbstractTimeSeriesReader):
    """
    Interface for CSV
    Expectations:
        - csv must have timestamp column
    """

    def __init__(self, csv_path: Path):
        """
        Parameters
        ~~~~~~~~~~
            mdata_path : Path to store meta data after having a pass over pcap file and forming an index
        """
        # Parameters
        self.logger = setup_logger(__class__.__name__)
        self.csv_path = csv_path
        self.logger.info("Reading csv...")
        strt = time()
        self.csv_df = pd.read_csv(csv_path)
        self.logger.info(
            f"CSV loaded, took {time() - strt: 4.2f} seconds with {len(self.csv_df)} length"
        )

        # TODO: Check this to work properly
        self.first_sniff_time: float = self.csv_df.loc[0, "timestamp"].astype(float)
        self.last_sniff_time: float = self.csv_df.loc[
            self.csv_df.index[-1], "timestamp"
        ].astype(float)

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx) -> Any:
        return self.csv_df.iloc[idx]

    def getTimestamp(self, idx):
        return self.csv_df.iloc[idx]["timestamp"]

    @property
    def init_time(self) -> float:
        return self.csv_df.iloc[0]["timestamp"]

    @property
    def fin_time(self) -> float:
        """
        Final Timestamp
        """
        return self.csv_df.iloc[-1]["timestamp"]
