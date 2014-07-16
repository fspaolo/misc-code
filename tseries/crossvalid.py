def kfold_cv(X, K, randomise=False):
	"""Generates K (training, validation) pairs from the items in X.

	Each pair is a partition of X, where validation is an iterable
	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

	If randomise is true, a copy of X is shuffled before partitioning,
	otherwise its order is preserved in training and validation.
	"""
	if randomise: 
            from random import shuffle
            X = list(X)
            shuffle(X)
	for k in xrange(K):
		training = [x for i, x in enumerate(X) if i % K != k]
		validation = [x for i, x in enumerate(X) if i % K == k]
		yield training, validation


def kfold(X, K, randomise=False):
	"""Generates K (training, validation) pairs from the items in X.

	Each pair is a partition of X, where validation is an iterable
	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

	If randomise is true, a copy of X is shuffled before partitioning,
	otherwise its order is preserved in training and validation.
	"""
	if randomise: 
            from random import shuffle
            X = list(X)
            shuffle(X)
        n = len(X)
	for k in xrange(K):
		training = [i for i in xrange(n) if i % K != k]
		validation = [i for i in xrange(n) if i % K == k]
		yield training, validation


X = [i for i in xrange(10)]
for training, validation in kfold_cv2(X, K=2):
    print training
    print validation

