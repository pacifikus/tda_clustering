import sys

import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import path
import os
from mlflow.tracking import MlflowClient
import mlflow
from dotenv import load_dotenv
from typing import Tuple

folder = path.Path(__file__).abspath()
sys.path.append(folder.parent.parent)

from utils import read_config, get_logger, plot_grid_search_results
from mlflow_utils import create_mlflow_experiment

load_dotenv()
target_cols = ['ideal_inner_metric', 'algo']


def split_data(
    input_data: pd.DataFrame,
    target_col: str,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Start computing features.

    Args:
        input_data: dataframe to compute features
        target_col: columns to process as target
        config: dictionary with configuration params
    Returns:
        Tuple with X_train, X_test, y_train, y_test
    """
    y = input_data[target_col]
    X = input_data.drop(
        ['dataset_name', 'true_n_clusters', 'rand'] + target_cols,
        axis=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config['train']['test_size'],
        random_state=config['base']['seed'],
    )
    return X_train, X_test, y_train, y_test


def search_hyperparams(
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: dict[str, str],
) -> GridSearchCV:
    """Search model hyperparams with GridSearchCV.

    Args:
        config: dictionary with configuration params
        X_train: train_data
        y_train: train labels
        scoring: scoring dict for multi-score GridSearchCV
    Returns:
        Tuple with X_train, X_test, y_train, y_test
    """
    clf_config = config['train']['rf']['param_grid']

    param_grid = {
        'n_estimators': clf_config['n_estimators'],
        'max_depth': clf_config['max_depth'] + [None],
        # 'min_samples_split': clf_config['min_samples_split'],
        # 'min_samples_leaf': clf_config['min_samples_leaf'],
        'random_state': [clf_config['random_state']],
    }

    clf = RandomForestClassifier()
    rf_grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=config['train']['cv'],
        verbose=config['train']['grid_search_verbose'],
        n_jobs=-1,
        scoring=scoring,
        refit='F1_macro',
    )

    rf_grid_search.fit(X_train, y_train)
    return rf_grid_search


def main(config_path: str) -> None:
    """Run model training and hyperparameter search.

    Args:
        config_path: path to configuration file
    """
    config = read_config(config_path)
    logger = get_logger('TRAIN', log_level=config['base']['log_level'])

    meta_dataset = pd.read_csv(
        config['data']['filepaths']['processed_data_file'],
    )

    client = MlflowClient(
        tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
        registry_uri=os.getenv('MLFLOW_STORAGE'),
    )

    for target in target_cols:
        exp_name = f'clf_{target}'
        exp_id = create_mlflow_experiment(exp_name)
        run = client.create_run(
            experiment_id=exp_id,
            start_time=None,
            tags=None,
        )
        logger.debug(
            f'MLflow experiment ID: {exp_id}, run ID: {run.info.run_id}',
        )

        with mlflow.start_run(
            run_id=run.info.run_id,
            experiment_id=exp_id,
        ):
            logger.info(f'Start hyperparameters search for {target} clf')
            X = meta_dataset[meta_dataset[target] != 3]  # TODO check, 3 = nans
            X_train, X_test, y_train, y_test = split_data(X, target, config)

            scoring = config['train']['grid_search_scoring']
            clf_type = f'clf_{target}'
            rf_grid_search = search_hyperparams(
                config,
                X_train,
                y_train,
                scoring,
            )
            model = rf_grid_search.best_estimator_
            f1_score = rf_grid_search.best_score_
            logger.info(f'Best grid search estimator: {model}')
            logger.info(f'Best grid search score: {f1_score}')

            mlflow.log_params(params=rf_grid_search.best_params_)
            mlflow.log_metric('f1_macro', f1_score)
            mlflow.sklearn.log_model(model, os.getenv('MLFLOW_STORAGE'))

            plot_grid_search_results(
                results=rf_grid_search.cv_results_,
                scoring=scoring,
                filepath=config['data']['filepaths']['figure_folder']
                + f'GridSearchCV_{clf_type}_results.png',
                best_params=rf_grid_search.best_params_,
            )

            logger.info(f'Save {clf_type} model')
            models_path = config['train']['model_path'] \
                + f'model__{clf_type}.joblib'
            joblib.dump(rf_grid_search, models_path)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    main(config_path=args.config)
