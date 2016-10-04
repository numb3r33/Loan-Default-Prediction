import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

def bestF1(true, pred_probs):
	"""
	Find the best optimal F1 score across grid of cutoffs
	"""

	best        = 0
	best_cutoff = 0

	for cutoff in np.arange(0.01, 0.99, 0.01):
		preds = pd.Series(pred_probs > cutoff).apply(lambda x: 1 if x else 0)
		tmp   = f1_score(true, preds)

		if tmp > best:
			best        = tmp
			best_cutoff = cutoff

	print('Best F1 score: %f at cutoff: %f'%(best, best_cutoff))
	return best
