import argparse
import sys
from collections import defaultdict
from typing import List

import numpy as np
import openml
import pandas as pd
import path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from logging import Logger
from typing import Generator

folder = path.Path(__file__).abspath()
sys.path.append(folder.parent.parent)

from utils import read_config, get_logger
from features.calcers.base_calcer import BaseCalcer
from features.calcers.stats_features import StatsFeaturesCalcer
from features.calcers.mapper_features import MapperFeatureCalcer
from features.calcers.tda_features import TDACalcer
from features.calcers.target_features import TargetCalcer

CALCER_REFERENCE = {}


def register_calcer(calcer_class) -> None:
    """Register new calcer in calcer dict.

    Args:
        calcer_class: Calcer class
    """
    CALCER_REFERENCE[calcer_class.name] = calcer_class


def create_calcer(name: str, **kwargs) -> BaseCalcer:
    """Create new calcer with given params.

    Args:
        name: Calcer name
        kwargs: keyword arguments of calcer class
    Returns:
        BaseCalcer object
    """
    return CALCER_REFERENCE[name](**kwargs)


def join_dataframes(
    tables: List[pd.DataFrame],
    on: List[str],
    how: str,
) -> pd.DataFrame:
    """Join features datasets to single.

    Args:
        tables: features datasets
        on: Column for merging by
        how: Merging type
    Returns:
        United dataframe
    """
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on=on, how=how)
    return merged


def get_full_datasets(
    ids_list: List[int],
    logger: Logger,
) -> Generator[pd.DataFrame, pd.Series, str]:
    """Download datasets content from OpenML.

    Args:
        ids_list: list of datasets indexes to download
        logger: Logger object to log state
    Returns:
        Generator for X, y, dataset.name values
    """
    for ids in ids_list:
        dataset = openml.datasets.get_dataset(ids)
        try:
            X, y, _, _ = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format='dataframe',
            )
            yield X, y, dataset.name
        except:
            logger.error(f'Dataset {dataset.name} loading error!')


def label_encode(input_data: pd.DataFrame) -> pd.DataFrame:
    """Label encode categorical features.

    Args:
        input_data: dataframe to label encode
    Returns:
        dataframe with computed features
    """
    if type(input_data) is pd.DataFrame:
        le_dict = defaultdict(LabelEncoder)
        non_numeric_cols = input_data.select_dtypes(
            exclude=[np.number],
        ).columns.tolist()
        cat_features = non_numeric_cols
        cols = input_data.loc[:, cat_features].columns
        input_data[cols] = input_data[cols].apply(
            lambda x: le_dict[x.name].fit_transform(x),
        )
    else:
        input_data = LabelEncoder().fit_transform(input_data)
    return input_data


def compute_features(
    input_data: pd.DataFrame,
    config: dict,
    dataset_name: str,
) -> pd.DataFrame:
    """Build calcers list and compute features.

    Args:
        input_data: dataframe to compute features
        config: dict with a configuration
        dataset_name: name of dataset to store as column
    Returns:
        dataframe with united computed features
    """
    calcers = []
    keys = None

    for calcer_name in config['features_calcers'].keys():
        calcer_args = config['features_calcers'][calcer_name]
        calcer = create_calcer(calcer_name, **calcer_args)
        if keys is None:
            keys = set(calcer.keys)
        elif set(calcer.keys) != keys:
            raise KeyError(f'{calcer.keys}')

        calcers.append(calcer)

    computation_results = []
    for calc in calcers:
        df = calc.compute(input_data)
        df['dataset_name'] = dataset_name
        computation_results.append(df)

    return join_dataframes(computation_results, on=list(keys), how='outer')


def drop_single_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """Drop features with one unique value.

    Args:
        input_data: dataframe to drop features
    Returns:
        dataframe without single features
    """
    return input_data.drop(
        [
            'onset_longest_Betti_hom_dim_0',
            'polynomial_feature_1_hom_dim_0',
            'polynomial_feature_3_hom_dim_0',
        ],
        axis=1,
    )


def main(config_path: str) -> None:
    """Run build features to turn raw data into processed data.

    Args:
       config_path: path to configuration file
    """
    config = read_config(config_path)

    logger = get_logger(
        'BUILD_FEATURES',
        log_level=config['base']['log_level'],
    )
    logger.info('Start compute features from OpenML datasets')

    openml_df = pd.read_csv(
       config['data']['filepaths']['raw_openml_data_file'],
    )
    dataset_ids = openml_df['did'].values.tolist()
    datasets_generator = get_full_datasets(dataset_ids, logger)

    features_df_list = []
    for X, y, name in tqdm(datasets_generator):
        X = label_encode(X)
        y = label_encode(y)
        X = X.fillna(0)
        X['target'] = y
        features = compute_features(X, config, name)
        features_df_list.append(features)

    features_df = pd.concat(features_df_list)
    features_df = drop_single_features(features_df)

    target_cols = ['ideal_inner_metric', 'algo']
    features_df[target_cols] = label_encode(features_df[target_cols])

    logger.info('Save meta-dataset with computed features')
    features_df.to_csv(
        config['data']['filepaths']['processed_data_file'],
        index=False,
    )


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    register_calcer(StatsFeaturesCalcer)
    register_calcer(MapperFeatureCalcer)
    register_calcer(TDACalcer)
    register_calcer(TargetCalcer)

    main(config_path=args.config)
