import os
import sys

import path
import pytest

folder = path.Path(__file__).abspath()
sys.path.append(folder.parent.parent)

from src.utils import read_config
import pandas as pd

path_current_directory = os.path.dirname(__file__)


@pytest.fixture(scope='module')
def config():
    config_path = os.path.join(path_current_directory, '../config/params.yaml')
    cfg = read_config(config_path)
    return cfg


@pytest.fixture(scope='module')
def tmp_folder(tmpdir_factory):
    fn = tmpdir_factory.mktemp('figures')
    return fn


@pytest.fixture(scope='module')
def openml_data():
    data = {
        'did': [1, 2, 3, 4, 5, 6],
        'name': ['labor', 'colic', 'credit', 'diabetes', 'haberman', 'breast'],
        'NumberOfInstances': [57, 286, 699, 23, 690, 768],
        'NumberOfFeatures': [17, 10, 10, 23, 16, 9],
        'NumberOfClasses': [2, 2, 2, 2, 2, 2],
    }
    return pd.DataFrame.from_dict(data)
