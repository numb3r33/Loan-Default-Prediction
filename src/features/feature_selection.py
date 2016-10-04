import numpy as np

__all__ = [
            "TreeBasedSelection"
            ]

class TreeBasedSelection(object):
    """
    Trains a extra trees classifier and based on the number of features
    to be selected chooses the top features based on the feature importance.
    """
    def __init__(self, estimator, target, n_features_to_select=None):
        self.estimator            =  estimator
        self.n_features_to_select =  n_features_to_select
        self.target               =  target

    def fit(self, X, y=None):
        self.estimator.fit(X, self.target)

        self.importances = self.estimator.feature_importances_
        self.indices     = np.argsort(self.importances)[::-1]

        return self

    def transform(self, X):
        return X[:, self.indices[:self.n_features_to_select]]
