"""
This :mod: `features` module includes methods to generate, select new features for the dataset.
"""

from .feature_generation import GoldenFeatures
from .feature_generation import FeatureInteraction
from .feature_selection import TreeBasedSelection

__all__ = [
			"GoldenFeatures",
			"FeatureInteraction",
			"TreeBasedSelection"
			]

