import gtda.homology as hl
import pandas as pd

from ..calcers.base_calcer import BaseCalcer
from ..tda_feature_engineering import (
    num_relevant_holes,
    average_lifetime,
    calculate_amplitude_feature,
    length_betti,
    sum_length,
    average_length,
    onset_longest_betti,
    polynomial_feature_1,
    polynomial_feature_2,
    polynomial_feature_3,
    polynomial_feature_4,
)


class TDACalcer(BaseCalcer):
    """Class to compute TDA (topological data analysis) features.

    Args:
        num_relevant_holes_theta: value between 0 and 1 to be used to
        calculate the threshold in num_relevant_holes feature
    Returns:
        Computed TDA features
    """

    name = 'tda_features'
    keys = ['dataset_name']

    def __init__(self, num_relevant_holes_theta: float) -> None:
        """Init TDACalcer."""
        self.num_relevant_holes_theta = num_relevant_holes_theta

        super().__init__()

    def compute(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Start computing features.

        Args:
            input_data: dataframe to compute features

        Returns:
            pd.DataFrame: computed statistical features
        """
        features = {}
        homology_dimensions = [0, 1, 2]
        persistence = hl.VietorisRipsPersistence(
            metric='euclidean',
            homology_dimensions=homology_dimensions,
        )
        persistence_diagram = persistence.fit_transform(
            input_data.drop('target', axis=1).values[None, :, :],
        )

        for homology_dim in homology_dimensions:
            features[
                f'avg_lifetime_hom_dim_{homology_dim}',
            ] = average_lifetime(
                persistence_diagram,
                homology_dim,
            )[0]
            features[
                f'num_relevant_holes_hom_dim_{homology_dim}'
            ] = num_relevant_holes(
                persistence_diagram,
                homology_dim,
                theta=self.num_relevant_holes_theta,
            )[0]
            features[
                f'length_Betti_hom_dim_{homology_dim}'
            ] = length_betti(
                persistence_diagram[0],
                homology_dim=homology_dim,
            )
            features[
                f'sum_length_hom_dim_{homology_dim}'
            ] = sum_length(
                persistence_diagram[0],
                homology_dim=homology_dim,
            )
            features[
                f'average_length_hom_dim_{homology_dim}'
            ] = average_length(
                persistence_diagram[0],
                homology_dim=homology_dim,
            )
            features[
                f'onset_longest_Betti_hom_dim_{homology_dim}'
            ] = onset_longest_betti(
                persistence_diagram[0],
                homology_dim=homology_dim,
            )
            features[
                f'polynomial_feature_1_hom_dim_{homology_dim}'
            ] = polynomial_feature_1(
                persistence_diagram[0],
                homology_dim=homology_dim,
            )
            features[
                f'polynomial_feature_2_hom_dim_{homology_dim}'
            ] = polynomial_feature_2(
                persistence_diagram[0],
                homology_dim=homology_dim,
            )
            features[
                f'polynomial_feature_3_hom_dim_{homology_dim}'
            ] = polynomial_feature_3(
                persistence_diagram[0],
                homology_dim=homology_dim,
            )
            features[
                f'polynomial_feature_4_hom_dim_{homology_dim}'
            ] = polynomial_feature_4(
                persistence_diagram[0],
                homology_dim=homology_dim,
            )

        features['amplitude_feature'] = calculate_amplitude_feature(
            persistence_diagram,
        )[0][0]
        return pd.DataFrame([features])
