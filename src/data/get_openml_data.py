import sys

import openml
import pandas as pd
import path

folder = path.Path(__file__).abspath()
sys.path.append(folder.parent.parent)

import argparse

from utils import read_config, save_pandas_plot, get_logger


def get_openml_df(config: dict) -> pd.DataFrame:
    """Construct dataset by defined criteria.

    Args:
        config: dictionary with configuration params
    Returns:
        pd.DataFrame with meta-features of OpenML datasets
    """
    df_cols = config['data']['openml_settings']['cols_to_load']
    openml_df = openml.datasets.list_datasets(
        output_format="dataframe",
    )[df_cols]
    figure_folder = config['data']['filepaths']['figure_folder']
    config = config['data']['openml_settings']

    openml_df = openml_df[
        (openml_df['NumberOfClasses'] > config['min_num_classes']) &
        (openml_df['NumberOfClasses'] < config['max_num_classes']) &
        (openml_df['NumberOfInstances'] < config['max_number_of_instances']) &
        (openml_df['NumberOfInstances'] > config['min_number_of_instances']) &
        (openml_df['NumberOfFeatures'] < config['max_number_of_features'])
    ]
    openml_df = openml_df.drop_duplicates(subset=['name'], keep='last')

    df_to_drop = [
        'tr45.wc', 'tr21.wc', 'tr31.wc', 'oh15.wc', 'tr11.wc', 'tr23.wc', 'fbis.wc', 'new3s.wc', 're0.wc',
        'oh0.wc', 'la2s.wc', 'oh5.wc', 're1.wc', 'la1s.wc', 'tr12.wc', 'wap.wc', 'tr41.wc', 'oh10.wc', 'fl2000',
        'diggle_table_a2', 'gina_prior2', 'MyIris', 'iris_test_upload', 'iris-example', 'iris_test', 'iriiiiiis',
        'openml_df', 'hypothyroid', 'page-blocks', 'optdigits', 'waveform-5000', 'isolet', 'satimage',
        'UNIX_user_data', 'JapaneseVowels', 'ipums_la_99-small', 'credit-g', 'oil_spill', 'monks-problems-3',
        'analcatdata_election2000', 'monks-problems-1', 'backache', 'profb', 'fri_c3_1000_25', 'bodyfat',
        'analcatdata_olympic2000', 'sensory', 'autoMpg', 'pharynx', 'strikes', 'cleveland', 'cholesterol', 'pbcseq',
        'rmftsa_ctoarrivals', 'autoMpg', 'sensory', 'disclosure_x_bias', 'analcatdata_neavote', 'no2',
        'chscase_census6', 'chscase_census5', 'chscase_census3', 'humandevel', 'colleges_usnews', 'cars', 'heart-c',
        'analcatdata_marketing', 'collins', 'grub-damage', 'ar4', 'PizzaCutter1', 'CostaMadre1', 'CastMetal1',
        'KnuggetChase3', 'PieChart1', 'PieChart3', 'PieChart2', 'sa-heart', 'heart-switzerland', 'vertebra-column',
        'volcanoes-a2', 'volcanoes-a3', 'volcanoes-a4', 'volcanoes-e1', 'volcanoes-e4', 'autoUniv-au1-1000',
        'autoUniv-au6-400', 'autoUniv-au7-1100', 'autoUniv-au6-1000', 'heart-h', 'dresses-sales', 'mofn-3-7-10',
        'mux6', 'Australian', 'car', 'cleve', 'threeOf9', 'xd6', 'climate-model-simulation-crashes', 'DRSongsLyrics',
        'Zombies-Apocalypse', 'volcanoes-e2', 'ibm-employee-attrition', 'cars1', 'ibm-employee-performance',
        'fri_c3_100_50',
    ]
    openml_df = openml_df[~openml_df['name'].isin(df_to_drop)]

    cols = ['NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses']
    for col in cols:
        hist_filepath = f"{figure_folder}{col}_hist.png"
        ax = openml_df[col].hist()
        save_pandas_plot(
            ax,
            title=f'Distribution of {col}',
            filepath=hist_filepath,
        )
        ax.clear()

    return openml_df


def main(config_path: str) -> None:
    """Download and save data from OpenML (https://www.openml.org/).

    Args:
        config_path: path to configuration file
    """
    config = read_config(config_path)

    logger = get_logger('DATA_LOAD', log_level=config['base']['log_level'])
    logger.info('Start loading dataset info from OpenML')

    openml_data = get_openml_df(config)

    logger.info('Save dataset info from OpenML')
    openml_data.to_csv(
        config['data']['filepaths']['raw_openml_data_file'],
        index=False,
    )


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    main(config_path=args.config)
