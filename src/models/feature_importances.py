import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def get_feature_importances_mdi(model, X_train):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=X_train.columns)
    return forest_importances, std


def get_feature_importances_permutation(model, X_test, y_test, config):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=config['train']['f_importances']['permutation']['n_repeats'],
        random_state=config['base']['seed'],
        n_jobs=-1
    )
    forest_importances = pd.Series(result.importances_mean, index=X_test.columns)
    return forest_importances, result.importances_std
