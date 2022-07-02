from abc import ABC, abstractmethod

import pandas as pd


class BaseCalcer(ABC):
    name = '_base'
    keys = None

    def __init__(self) -> None:
        """Init `BaseCalcer`."""

    @abstractmethod
    def compute(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Compute features from `input_data`.

        Args:
            input_data: initial dataframe to compute features
        Returns:
            pd.DataFrame with computed features
        """
