import networkx as nx
import pandas as pd
from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
)
from networkx.algorithms.approximation.clique import large_clique_size
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from features.calcers.base_calcer import BaseCalcer


def compute_graph_features(graph: nx.Graph) -> dict:
    """Compute Mapper features for given dataset.

    Args:
        graph: graph from Mapper
    Returns:
        Computed graph features
    """
    return {
        'average_clustering': nx.average_clustering(graph),
        'large_clique_size': large_clique_size(graph),
        'node_connectivity': nx.node_connectivity(graph),
        'number_connected_components': nx.number_connected_components(graph),
        'degree_assortativity_coefficient':
            nx.degree_assortativity_coefficient(graph),
    }


class MapperFeatureCalcer(BaseCalcer):
    """Class to compute features with Mapper algorithm.

    Class to computing features from Mapper connected component graph.
    Uses networkx methods for computing features based on graph.

    Args:
        num_pca_components: Number of PCA components of Mapper
        random_state: value which implies the selection of a random
        cubical_cover_intervals: number of intervals in the covers of
        CubicalCover feature dimensions
        overlap_frac: fractional overlap between consecutive intervals
        in the covers of CubicalCover feature dimensions
    """

    name = 'mapper_features'
    keys = ['dataset_name']

    def __init__(
        self,
        num_pca_components: int,
        random_state: int,
        cubical_cover_intervals: int,
        overlap_frac: float,
    ) -> None:
        """Init MapperCalcer."""
        self.num_pca_components = num_pca_components
        self.random_state = random_state
        self.cubical_cover_intervals = cubical_cover_intervals
        self.overlap_frac = overlap_frac

        super().__init__()

    def compute(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Start computing features.

        Args:
            input_data: dataframe to compute features

        Returns:
            pd.DataFrame: computed graph features
            (average_clustering, large_clique_size, node_connectivity,
            number_connected_components, degree_assortativity_coefficient)
        """
        graph = self._get_mapper_graph(input_data.drop('target', axis=1))
        features = compute_graph_features(graph)
        return pd.DataFrame([features])

    def _get_mapper_graph(self, input_data: pd.DataFrame) -> nx.Graph:
        """Get graph of connected components from initial dataframe.

        Args:
            input_data: dataframe with single dataset data to compute graph

        Returns:
            nx.Graph: graph of connected components
        """
        filter_func = PCA(
            n_components=self.num_pca_components,
            random_state=self.random_state,
        )
        cover = CubicalCover(
            n_intervals=self.cubical_cover_intervals,
            overlap_frac=self.overlap_frac,
        )

        pipe = make_mapper_pipeline(
            filter_func=filter_func,
            cover=cover,
            clusterer=DBSCAN(),
            verbose=False,
            n_jobs=-1,
        )

        graph = pipe.fit_transform(input_data)
        edgelist = graph.get_edgelist()
        return nx.Graph(edgelist)
