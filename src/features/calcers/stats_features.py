from scipy.stats import skew, kurtosis, zscore
import numpy as np
import pandas as pd
from ..calcers.base_calcer import BaseCalcer

from scipy.spatial.distance import cdist


class StatsFeaturesCalcer(BaseCalcer):
    """Class to compute statistical features for given dataset.

    Class to compute mean, variance, std, skewness, kurtosis,
    binned z_scores and binned value percentages.

    Args:
        percentages_step: step for percentages computing
    Returns:
        Computed statistical features
    """

    name = 'stats_features'
    keys = ['dataset_name']

    def __init__(self, percentages_step: float) -> None:
        """Init StatsFeaturesCalcer."""
        self.d_vector = np.ndarray(0)
        self.percentages_step = percentages_step
        self.vector_len = 0
        super().__init__()

    def compute(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Start computing features.

        Args:
            input_data: dataframe to compute features

        Returns:
            pd.DataFrame: computed statistical features
        """
        self.d_vector = self._compute_d_vector(
            input_data.drop('target', axis=1),
        )
        self.vector_len = len(self.d_vector)
        stats = {
            'md1_mean': self.d_vector.mean(),
            'md2_var': self.d_vector.var(),
            'md3_std': self.d_vector.std(),
            'md4_skew': skew(self.d_vector),
            'md5_kurtosis': kurtosis(self.d_vector),
        }

        stats.update(self._compute_percentages())
        stats.update(self._compute_z_scores())
        return pd.DataFrame([stats])

    def _compute_d_vector(self, input_data: pd.DataFrame) -> np.ndarray:
        """Compute vector of pairwise distances.

        Args:
            input_data: dataframe to compute features

        Returns:
            np.ndarray: vector of pairwise distances
        """
        return cdist(input_data, input_data).flatten()

    def _compute_percentages(self) -> dict:
        """Compute binned percentages by d-vector.

        Returns:
            dict: dict with the percentage of d-vector values
             that are contained in the given range
        """
        percentages = {}
        percentiles = np.linspace(0, 1, 21)
        for left_bound in percentiles[:-1]:
            left_bound = round(left_bound, 2)
            right_bound = round(left_bound + self.percentages_step, 2)
            condition = np.where(
                (self.d_vector < right_bound) &
                (self.d_vector >= left_bound),
            )
            percentages[
                f'md6_d_percentage_{left_bound}_{right_bound}'
            ] = len(self.d_vector[condition]) / self.vector_len
        return percentages

    def _compute_z_scores(self) -> dict:
        """Compute binned z-scores by d-vector.

        Returns:
            dict: dict with the percentage of d-vector z-scores values
             that are contained in the given range
        """
        z_scores_result = {}
        z_bins = list(range(-3, 4))
        z_scores = zscore(self.d_vector)
        for left_bound in z_bins[:-1]:
            left_bound = round(left_bound, 2)
            right_bound = round(left_bound + 1, 2)
            condition = np.where(
                (z_scores < right_bound) & (z_scores >= left_bound),
            )
            z_score = len(z_scores[condition]) / self.vector_len
            z_scores_result[
                f'md7_z_score_{left_bound}_{right_bound}'
            ] = z_score
        return z_scores_result
