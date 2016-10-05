
__all__ = ['transform',
            'drop_features'
            ]

def transform(df, features, transform_func):
	# only do transformation on features with positive values
	mask = (df[features].values >= 0).all(axis=0)
	df[features].loc[:, mask] = df[features].loc[:, mask].apply(transform_func)
	return df

def drop_features(df, features):
    return df.drop(features, axis=1)
