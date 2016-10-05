import numpy as np

__all__ = [
			"TreeBasedSelection",
			"forward_step_selection"
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


def forward_step_selection(X_train, y_train, X_test, y_test, selected_features, potential_features):
	"""
	Takes in a list of feature indices in order of their importance
	and chooses feature until it no longer improves the overall
	AUC score and returns list of selected features.
	"""
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import roc_auc_score

	is_improving = True

	auc_old = .5

	while is_improving == True:
		auc_scores = []

		for feature in potential_features:
			Xtr = X_train.loc[:, selected_features + [feature]]
			Xte = X_test.loc[:, selected_features + [feature]]

			model = LogisticRegression(n_jobs=2)
			model.fit(Xtr, y_train)

			preds = model.predict_proba(Xte)[:, 1]

			auc_score = roc_auc_score(y_test, preds)
			auc_scores.append(auc_score)

		auc_score_new = max(auc_scores)

		if auc_score_new > auc_old:
			auc_old = auc_score_new
			feature = potential_features.pop(auc_scores.index(auc_score_new))
			selected_features.append(feature)
		else:
			print('No longer able to improve AUC score')
			is_improving = False

	return selected_features





