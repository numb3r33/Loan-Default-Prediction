import numpy as np
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin

from .helpers import create_golden_feature


__all__ = [
			"GoldenFeatures",
			"FeatureInteraction"
			]

class GoldenFeatures(BaseEstimator, TransformerMixin):
	"""
	Transformer that adds two golden features found in the Kaggle Forum
	"""

	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		X = create_golden_feature(X)
		return X

class FeatureInteraction(BaseEstimator, TransformerMixin):
	"""
	Transformer that creates new features based on interaction among
	features ( subtraction ) only for now.
	"""
	def __init__(self):
		pass

	@staticmethod
	def _combinations(features):
		return combinations(features, 2)

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		features = map(str, list(range(X.shape[1])))
		interactions = []

		for comb in self._combinations(features):
			feat_1, feat_2 =  comb
			# subtraction
			interactions.append(X[:, int(feat_2)] - X[:, int(feat_1)])
			# addition
			interactions.append(X[:, int(feat_2)] + X[:, int(feat_1)])
			# multiplication
			interactions.append(X[:, int(feat_2)] * X[:, int(feat_1)])

		return np.vstack(interactions).T
