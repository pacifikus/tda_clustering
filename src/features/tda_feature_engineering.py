"""TDA features computing."""

from typing import List

import gtda.diagrams as diag
import numpy as np
import pandas as pd


def num_relevant_holes(
    X_scaled: np.ndarray,
    homology_dim: int,
    theta: float = 0.5
) -> List[int]:
    n_rel_holes = []

    for i in range(X_scaled.shape[0]):
        persistence_table = pd.DataFrame(
            X_scaled[i],
            columns=['birth', 'death', 'homology']
        )
        persistence_table['lifetime'] = \
            persistence_table['death'] - persistence_table['birth']
        threshold = persistence_table[
            persistence_table['homology'] == homology_dim
        ]['lifetime'].max() * theta
        n_rel_holes.append(
            persistence_table[
                (persistence_table['lifetime'] > threshold) &
                (persistence_table['homology'] == homology_dim)
            ].shape[0]
        )
    return n_rel_holes


def average_lifetime(X_scaled: np.ndarray, homology_dim: int) -> List[float]:
    avg_lifetime_list = []

    for i in range(X_scaled.shape[0]):
        persistence_table = pd.DataFrame(
            X_scaled[i],
            columns=['birth', 'death', 'homology'],
        )
        persistence_table['lifetime'] = \
            persistence_table['death'] - persistence_table['birth']
        avg_lifetime_list.append(
            persistence_table[
                persistence_table['homology'] == homology_dim
            ]['lifetime'].mean(),
        )

    return avg_lifetime_list


def calculate_amplitude_feature(
    X_scaled: np.ndarray,
    metric: str = 'wasserstein',
    order: int = 2,
) -> np.ndarray:
    amplitude = diag.Amplitude(metric=metric, order=order)
    return amplitude.fit_transform(X_scaled)


def length_betti(
    persistence_diagram: np.ndarray,
    n: int = 2,
    homology_dim: int = 0,
) -> float:
    try:
        return np.partition(
            np.diff(
                persistence_diagram[:, :2][
                    persistence_diagram[:, -1] == homology_dim
                ],
            ).flatten(), -1 * n,
        )[-1 * n]
    except:
        return -1


def sum_length(
    persistence_diagram: np.ndarray,
) -> float:
    return np.diff(
        persistence_diagram[:, :2][persistence_diagram[:, -1] == 0],
    ).sum()


def average_length(
    persistence_diagram: np.ndarray,
    homology_dim: int = 0,
) -> float:
    return np.diff(
        persistence_diagram[:, :2][persistence_diagram[:, -1] == homology_dim],
    ).mean()


def onset_longest_betti(
    persistence_diagram: np.ndarray,
    homology_dim=1,
) -> float:
    persistence_hom_dim = persistence_diagram[:, :2][
        persistence_diagram[:, -1] == homology_dim
    ]
    return persistence_hom_dim[
        np.diff(persistence_hom_dim).argmax()
    ][0]


def smallest_onset(persistence_diagram, homology_dim=1, cutoff=1.0):
    try:
        return persistence_diagram[:, :2][
            (np.diff(persistence_diagram[:, :2]) >= cutoff).flatten(), :
        ].flatten()[0]
    except:
        return -1


def average_middle_point(
    persistence_diagram: np.ndarray,
    cutoff: float,
) -> float:
    return persistence_diagram[:, :2][
        (np.diff(persistence_diagram[:, :2]) >= cutoff).flatten(), :
    ].mean(axis=1)


def polynomial_feature_1(
    persistence_diagram: np.ndarray,
    homology_dim: int,
) -> float:
    # Polynomial of barcode features
    if homology_dim in persistence_diagram[:, -1]:
        num = persistence_diagram[
            persistence_diagram[:, -1] == homology_dim
        ].shape[0]
        persistence_hom_dim = persistence_diagram[
            persistence_diagram[:, -1] == homology_dim
        ]
        return (
            persistence_hom_dim[:, 0] *
            (persistence_hom_dim[:, 1] - persistence_hom_dim[:, 0])
        ).sum() / num
    return np.inf


def polynomial_feature_2(
    persistence_diagram: np.ndarray,
    homology_dim: int,
) -> float:
    # Polynomial of barcode features
    if homology_dim in persistence_diagram[:, -1]:
        num = persistence_diagram[
            persistence_diagram[:, -1] == homology_dim
        ].shape[0]
        persistence_hom_dim = persistence_diagram[
            persistence_diagram[:, -1] == homology_dim
        ]
        return (
            (persistence_hom_dim[:, 1].max() - persistence_hom_dim[:, 1])
            * (persistence_hom_dim[:, 1] - persistence_hom_dim[:, 0])
        ).sum() / num
    return np.inf


def polynomial_feature_3(
    persistence_diagram: np.ndarray,
    homology_dim: int,
) -> float:
    # Polynomial of barcode features
    if homology_dim in persistence_diagram[:, -1]:
        num = persistence_diagram[
            persistence_diagram[:, -1] == homology_dim
        ].shape[0]
        persistence_hom_dim = persistence_diagram[
            persistence_diagram[:, -1] == homology_dim
        ]
        return (
            persistence_hom_dim[:, 0] ** 2
            * (persistence_hom_dim[:, 1] - persistence_hom_dim[:, 0]) ** 4
        ).sum() / num
    return np.inf


def polynomial_feature_4(
    persistence_diagram: np.ndarray,
    homology_dim: int,
) -> float:
    # Polynomial of barcode features
    if homology_dim in persistence_diagram[:, -1]:
        num = persistence_diagram[
            persistence_diagram[:, -1] == homology_dim
        ].shape[0]
        persistence_hom_dim = persistence_diagram[
            persistence_diagram[:, -1] == homology_dim
        ]
        return (
            (persistence_hom_dim[:, 1].max() - persistence_hom_dim[:, 1]) ** 2
            * (persistence_hom_dim[:, 1] - persistence_hom_dim[:, 0]) ** 4
        ).sum() / num
    return np.inf
