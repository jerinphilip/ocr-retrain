from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
from sklearn.manifold import TSNE


def tsne(aff_matrix):
	X_tsne = TSNE(learning_rate=100).fit_transform(aff_matrix)
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c='r')
	plt.show()
