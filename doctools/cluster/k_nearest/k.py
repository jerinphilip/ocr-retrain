from scipy.spatial import distance


def k_neighbours(u, X, k):
	euc=[]
	for i in range(len(X)):
		euc.append((distance.euclidean(u, X[i]), X[i]))
	euc = sorted(euc, key=lambda x:x[0])
	return [euc[i][1] for i in range(k)]