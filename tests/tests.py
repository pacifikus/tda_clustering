from src.data.get_openml_data import get_openml_df, plot_hist_by_cols
import pandas as pd
import os


def test_get_openml_data(config, tmp_folder):
    config['data']['filepaths']['figure_folder'] = tmp_folder
    result = get_openml_df(config)

    assert type(result) is pd.DataFrame
    assert result.columns.tolist() == config['data']['openml_settings']['cols_to_load']


def test_plot_hist_by_cols(tmp_folder, openml_data):
    cols = ['NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses']
    plot_hist_by_cols(cols=cols, figure_folder=tmp_folder, openml_df=openml_data)
    assert len(
        [
            entry for entry in os.listdir(tmp_folder)
            if os.path.isfile(os.path.join(tmp_folder, entry))
        ]
    ) == len(cols)
    for col in cols:
        assert os.path.exists(os.path.join(tmp_folder, f'{col}_hist.png'))
