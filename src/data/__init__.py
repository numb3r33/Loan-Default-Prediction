"""
This :mod: `data` module includes methods to modify data.
"""

from .helpers import get_stratified_sample
from .helpers import fill_missing_values
from .preprocess import drop_features
from .preprocess import transform

__all__ = [
			"get_stratified_sample",
			"drop_features",
			"transform",
			"fill_missing_values"
			]
