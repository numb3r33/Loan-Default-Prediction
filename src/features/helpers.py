import numpy as np

__all__ = ["feature_importance",
			"create_golden_feature"
			]

def create_golden_feature(df):
	df['f528-f527'] = df['f528'] - df['f527']
	df['f528-f274'] = df['f528'] - df['f274']

	return df


def feature_importance(estimator, X, y):
	estimator.fit(X, y)

	importances = estimator.feature_importances_
	indices     = np.argsort(importances)[::-1]

	return indices
