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