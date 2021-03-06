# tda_clustering

[![Actions Status](https://github.com/pacifikus/tda_clustering/workflows/Tests/badge.svg)](https://github.com/pacifikus/tda_clustering/actions)
[![Actions Status](https://github.com/pacifikus/tda_clustering/workflows/StyleGuide/badge.svg)](https://github.com/pacifikus/tda_clustering/actions)
[![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide)

## Overview

This is a project to predict optimal clustering algorithm and quality metric for a given dataset using topological data analysis features. 

Main idea is to train meta-algorithm on classification datasets to create prediction for unknown data.

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── get_openml_data.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── calcers    <- Calcers realization for features
    │   │   │   ├── base_calcer.py
    │   │   │   ├── mapper_features.py
    │   │   │   ├── stats_features.py
    │   │   │   ├── target_features.py
    │   │   │   └── tda_features.py
    │   │   ├── tda_feature_engineering.py
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models
    │   │   ├── feature_importances.py
    │   │   └── train_model.py
    │   │
    │   ├── mlflow_utils.py
    │   └── utils.py
    ├── tests              <- unit tests
    │   ├── conftest.py
    │   └── tests.py
    └── test_environment.py

## How to run

First of all, run MLflow. You can specify backend-store-uri and default-artifact-root as you wish:

```
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root artifacts
```

To collect OpenML meta-data run

```
src/data/get_openml_data.py --config config/params.yaml
```

To collect OpenML datasets and build processed data run:

```
src/features/build_features.py --config config/params.yaml
```

To train model run:

```
src/models/train_model.py --config config/params.yaml
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
