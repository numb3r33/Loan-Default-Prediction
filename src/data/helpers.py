import pandas as pd

from sklearn.cross_validation import train_test_split

def get_stratified_sample(X, y, train_size, random_state=10):
    """
    Takes in a feature set and target with percentage of training size and a seed for reproducability.
    Returns indices for the training and test sets.
    """

    itrain, itest = train_test_split(range(len(X)), \
    								 stratify=y, \
    								 train_size=train_size, \
    								 random_state=random_state)
    return itrain, itest


def fill_missing_values(df):
	columns = df.columns

	for col in columns:
		if pd.isnull(df[col]).any():
			df[col] = df[col].fillna(df[col].median())

	return df
