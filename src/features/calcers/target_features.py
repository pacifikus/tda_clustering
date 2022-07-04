import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (
    rand_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from ..calcers.base_calcer import BaseCalcer
from typing import Tuple, List


class TargetCalcer(BaseCalcer):
    """Class to compute targets.

    Class to compute optimal metric and clustering algorithm.

    Args:
        clustering_algos: list of available clustering algorithms
        random_state: random value to init clustering
    Returns:
        Computed statistical features
    """

    name = 'meta_features'
    keys = ['dataset_name']

    def __init__(self, clustering_algos: List[str], random_state: int) -> None:
        """Init TargetCalcer."""
        self.clustering_algos = clustering_algos
        self.random_state = random_state
        super().__init__()

    def compute(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Start computing features.

        Args:
            input_data: dataframe to compute features

        Returns:
            pd.DataFrame: computed targets
        """
        n_clusters = len(np.unique(input_data['target']))
        X = input_data.drop('target', axis=1)
        y = input_data['target']
        target_df = {
            'algo': [],
            'rand': [],
            'silhouette': [],
            'CH': [],
            'DB': [],
        }

        start = n_clusters - 3 if n_clusters - 3 > 1 else 2
        for n_cluster in range(start, n_clusters + 3):
            for algo in self.clustering_algos:
                if algo == 'KMeans':
                    clustering = KMeans(
                        n_clusters=n_cluster,
                        random_state=self.random_state,
                    )
                elif algo == 'AgglomerativeClustering':
                    clustering = AgglomerativeClustering(n_clusters=n_cluster)
                else:
                    clustering = SpectralClustering(
                        n_clusters=n_cluster,
                        random_state=self.random_state,
                    )

                preds = clustering.fit_predict(X)

                rand = rand_score(y, preds)
                silhouette, calinski_harabasz, db = \
                    self._compute_inner_metrics(X, preds)
                target_df['algo'].append(algo)
                target_df['silhouette'].append(silhouette)
                target_df['rand'].append(rand)
                target_df['CH'].append(calinski_harabasz)
                target_df['DB'].append(db)
        target_df = pd.DataFrame(target_df)
        opt_algo = self._get_opt_algo(target_df)
        ideal_inner_metric = self._get_opt_metric(target_df, opt_algo)

        return pd.DataFrame(
            [{'algo': opt_algo, 'ideal_inner_metric': ideal_inner_metric}],
        )

    def _get_opt_algo(self, input_data: pd.DataFrame) -> str:
        """Get optimal algo as algo with max Rand value.

        Args:
            input_data: dataframe to compute features

        Returns:
            str: optimal clustering algorithm name
        """
        return input_data.sort_values(
            'rand',
            ascending=False,
        )['algo'].values[0]

    def _get_opt_metric(self, input_data: pd.DataFrame, algo: str) -> str:
        """Get optimal metric.

         Compute optimal metric as metric with the most strong
         positive correlation with Rand metric.

        Args:
            input_data: dataframe to compute features
            algo: current clustering algorithm to filter out results
            and compute metric
        Returns:
            optimal metric name
        """
        metrics = ['silhouette', 'CH', 'DB']
        input_data['DB'] = 1 / input_data['DB']
        opt_metrics = input_data[
            input_data['algo'] == algo
        ].corrwith(input_data['rand'])[metrics]
        return opt_metrics[['silhouette', 'CH', 'DB']].idxmax(axis=0)

    def _compute_inner_metrics(
        self,
        X: pd.DataFrame,
        preds: np.ndarray,
    ) -> Tuple[str, str, str]:
        """Compute inner metrics.

        Compute silhouette, calinski-harabasz index, davies-bouldin index.

        Args:
            X: dataframe to compute features
            preds: predictions from algorithm
        Returns:
            Tuple with computed metrics
        """
        silhouette = silhouette_score(X, preds)
        calinski_harabasz = calinski_harabasz_score(X, preds)
        db = davies_bouldin_score(X, preds)
        return silhouette, calinski_harabasz, db
