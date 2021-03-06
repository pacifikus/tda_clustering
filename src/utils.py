import logging
import sys
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pandas.plotting import table
from itertools import cycle


def read_config(cfg_path: str) -> dict:
    """Read config from .yaml file.

    Args:
        cfg_path: path to .yaml file
    Returns:
        Loaded configuration
    """
    with open(cfg_path, 'r') as f_open:
        return yaml.load(f_open, Loader=yaml.SafeLoader)


def save_plot_as_file(ax: plt.Axes, title: str, filepath: str) -> None:
    """Plot pandas to .png.

    Args:
        ax: dfigure to plot
        title: title of plot
        filepath: path to save .png file
    """
    fig = ax.get_figure()
    plt.title(title)
    fig.savefig(filepath)


def save_df_as_img(df: pd.DataFrame, filepath: str) -> None:
    """Plot dataset information from `describe` method to .png.

    Args:
        df: dataframe to plot data
        filepath: path to save .png files
    """
    plot = plt.subplot(111, frame_on=False)
    plot.xaxis.set_visible(False)
    plot.yaxis.set_visible(False)

    table(plot, df, loc='upper right')

    plt.savefig(filepath)


def plot_grid_search_results(
    gs_results: dict,
    scoring: dict,
    best_params: dict,
    filepath: str,
    refit_scorer: str = 'F1_macro',
) -> None:
    """Plot gridsearch results to .png.

    Args:
        gs_results: results from GridSearchCV by each iteration
        scoring: dict with GridSearchCV scoring functions
        best_params: best_params best estimator params
        filepath: path to save .png file
        refit_scorer: main metric of final refitted estimator
    """
    plt.figure(figsize=(13, 13))

    fig, ax = plt.subplots()
    ax.set_ylim(bottom=0.2, top=0.8)
    ax.set_title(
        f"GridSearchCV results (max_depth = {best_params['max_depth']})",
        fontsize=12,
    )
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Score')

    n_estimators = len(np.unique(gs_results['param_n_estimators']))
    start = np.nonzero(
        gs_results['param_max_depth'] == best_params['max_depth'],
    )[0][0]
    x_axis = np.array(
        gs_results['param_n_estimators'].data[start:start + n_estimators],
        dtype=float,
    )

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        sample_styles = [('test', '-')]
        for sample, style in sample_styles:
            sample_score_mean = gs_results[
                f'mean_{sample}_{scorer}'
            ][start:start + n_estimators]
            sample_score_std = gs_results[
                f'std_{sample}_{scorer}'
            ][start:start + n_estimators]
            ax.fill_between(
                x_axis,
                sample_score_mean - sample_score_std,
                sample_score_mean + sample_score_std,
                alpha=0.1 if sample == 'test' else 0,
                color=color,
            )
            ax.plot(
                x_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=1 if sample == 'test' else 0.7,
                label='%s (%s)' % (scorer, sample),
            )

    best_index = np.nonzero(gs_results[f'rank_test_{refit_scorer}'] == 1)[0][0]
    best_score = gs_results[f'mean_test_{refit_scorer}'][best_index]
    ax.plot(
        [x_axis[abs(best_index - start)]] * 2,
        [0, best_score],
        linestyle='-.',
        color='k',
        marker='x',
        markeredgewidth=3,
        ms=8,
        markevery=[-1],
    )
    ax.annotate(
        '%0.2f' % best_score,
        (x_axis[abs(best_index - start)], best_score + 0.005),
    )
    plt.legend(loc='best')
    plt.grid(True)
    fig.savefig(filepath)


def plot_feature_importances(
    feature_importances: np.ndarray,
    std: np.ndarray,
    filepath: str,
    fi_type: str = 'MDI',
) -> None:
    """
    Plot feature importances plot to .png file.

    Args:
        feature_importances: computed values of feature importances
        std: standard deviation of `feature_importances`
        filepath: path to store .png plot
        fi_type: type of feature importances computation
    """
    fig, ax = plt.subplots(figsize=(28, 12))
    feature_importances.plot.bar(yerr=std, ax=ax)
    title = f'Feature importances using {fi_type}'
    ax.set_ylabel('Mean decrease in value')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    save_plot_as_file(ax, title=title, filepath=filepath)


def plot_roc_curves(
    fpr: dict,
    tpr: dict,
    roc_auc: dict,
    n_classes: int,
    clf_type: str,
    filepath: str,
) -> None:
    """
    Plot computed roc curves with micro and macro avg to .png file.

    Args:
        fpr: false positive rates
        tpr: true positive rates
        roc_auc: roc_auc values
        n_classes: number of classes
        clf_type: type of classifier
        filepath: path to save .png file
    """
    lw = 2

    plt.figure()
    plt.plot(
        fpr['micro'],
        tpr['micro'],
        label='micro-average ROC curve (area = {0:0.2f})'.format(
            roc_auc['micro'],
        ),
        color='deeppink',
        linestyle=':',
        linewidth=4,
    )

    plt.plot(
        fpr['macro'],
        tpr['macro'],
        label='macro-average ROC curve (area = {0:0.2f})'.format(
            roc_auc['macro'],
        ),
        color='navy',
        linestyle=':',
        linewidth=4,
    )

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for idx, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[idx],
            tpr[idx],
            color=color,
            lw=lw,
            label='ROC curve of class {0} (area = {1:0.2f})'.format(
                idx,
                roc_auc[idx],
            ),
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-curve for {clf_type} classifier')
    plt.legend(loc='lower right')
    plt.savefig(filepath)


def get_console_handler() -> logging.StreamHandler:
    """Get console handler.

    Returns:
        logging.StreamHandler which logs into stdout
    """
    console_handler = logging.StreamHandler(sys.stdout)
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_fmt)

    console_handler.setFormatter(formatter)

    return console_handler


def get_logger(
    name: str = __name__,
    log_level: Union[str, int] = logging.DEBUG,
) -> logging.Logger:
    """Get logger object.

    Args:
        name: logger name
        log_level: logging level, can be string name or integer value
    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(get_console_handler())
    logger.propagate = False

    return logger
