import argparse
import os
import sys
from typing import Tuple

import joblib
import mlflow
import pandas as pd
import numpy as np
import path
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, classification_report


folder = path.Path(__file__).abspath()
sys.path.append(folder.parent.parent)

from feature_importances import (
    get_feature_importances_mdi,
    get_feature_importances_permutation,
)
from utils import (
    read_config,
    get_logger,
    plot_grid_search_results,
    plot_feature_importances,
    plot_roc_curves,
    save_df_as_img,
)
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
        ['dataset_name', 'true_n_clusters', 'rand'] + target_cols + ['DB', 'CH', 'silhouette'],
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
    scoring: dict,
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
        'min_samples_split': clf_config['min_samples_split'],
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


def start_interpretation(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict,
    clf_name: str,
) -> None:
    """
    Compute feature importances for the given model.

    Args:
        model: fitted model object
        X_train: train data
        X_test: test data
        y_test: test labels
        config: dictionary with configuration params
        clf_name: classificator type (target)
    """
    mdi_importances, mdi_std = get_feature_importances_mdi(model, X_train)
    mdi_filepath = os.path.join(
        config['data']['filepaths']['figure_folder'],
        f'{clf_name}_' + config['train']['f_importances']['mdi']['plot_file_name']
    )
    perm_importances, perm_std = get_feature_importances_permutation(
        model,
        X_test,
        y_test,
        config,
    )
    perm_filepath = os.path.join(
        config['data']['filepaths']['figure_folder'],
        f'{clf_name}_' + config['train']['f_importances']['permutation']['plot_file_name']
    )
    plot_feature_importances(mdi_importances, mdi_std, mdi_filepath)
    plot_feature_importances(perm_importances, perm_std, perm_filepath, fi_type='Permutation')


def compute_roc_curves(
    clf: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    clf_type: str,
) -> None:
    """
    Compute roc curves with micro and macro avg for a given classifier.
    Args:
        clf: fitted model
        X_test: test split of the data
        y_test: test labels
        X_train: train  split of the data
        y_train: train labels
        config: dictionary with configuration params
        clf_type: type of classifier
    """
    n_classes = y_train.nunique()
    y_train = label_binarize(y_train, classes=[0, 1, 2])
    y_test = label_binarize(y_test, classes=[0, 1, 2])
    y_score = clf.fit(X_train, y_train).predict(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    img_filepath = os.path.join(
        config['data']['filepaths']['figure_folder'],
        f'{clf_type}_roc_auc_curves.png',
    )
    plot_roc_curves(
        fpr,
        tpr,
        roc_auc,
        n_classes,
        clf_type,
        filepath=img_filepath,
    )


def compute_classification_report(
    clf: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict,
    clf_type: str,
) -> None:
    """
    Compute and plot to file classification report by test split.
    Args:
        clf: fitted model
        X_test: test split of the data
        y_test: test labels
        config: dictionary with configuration params
        clf_type: type of classifier

    Returns:

    """
    y_preds = clf.predict(X_test)
    clf_report = classification_report(y_test, y_preds, output_dict=True)
    clf_report = pd.DataFrame(clf_report).transpose()
    img_filepath = os.path.join(
        config['data']['filepaths']['figure_folder'],
        f'{clf_type}_clf_report.png',
    )
    save_df_as_img(clf_report, img_filepath)


def find_best_run(target_col: str) -> mlflow.entities.Run:
    """
    Find best run from MLflow experiment.
    Args:
        target_col: target column to get best run

    Returns:
        best run of the experiment
    """
    exp_name = f'clf_{target_col}'
    client = MlflowClient(
        tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
        registry_uri=os.getenv('MLFLOW_STORAGE'),
    )
    experiment = client.get_experiment_by_name(exp_name)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.f1_macro DESC"]
    )[0]
    return best_run


def register_model(target_col: str) -> None:
    """
    Register model from the best MLflow run.

    Args:
        target_col: target column to get best run
    """
    best_run = find_best_run(target_col)
    mlflow.register_model(
        model_uri=f'runs:/{best_run.info.run_id}/model',
        name=f'{target_col}_rf'
    )


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
            X = meta_dataset.fillna(0)
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

            start_interpretation(model, X_train, X_test, y_test, config, clf_name=target)
            compute_classification_report(model, X_test, y_test, config, clf_type=target)
            compute_roc_curves(model, X_test, y_test, X_train, y_train, config, clf_type=target)

            logger.info(f'Save {clf_type} model')
            models_path = config['train']['model_path'] \
                + f'model__{clf_type}.joblib'
            joblib.dump(rf_grid_search, models_path)

            register_model(target)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    main(config_path=args.config)
