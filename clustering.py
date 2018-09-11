import numpy as np
from sklearn.cluster import MiniBatchKMeans


class ClusteringKMeans:

    def __init__(self, num_clusters, batch_size, X=None):
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.kmeans = MiniBatchKMeans(num_clusters, batch_size==batch_size, verbose=True)
        if X is not None:
            self.kmeans.fit(X)

    def test(self):
        X = np.random.rand(self.batch_size * 5, 8)
        self.kmeans.fit(X)
        print('Test is done!')
    
    def predict(self, x):
        return self.kmeans.predict(x)

    def fit(self, X):
        self.kmeans.fit(X)

    def score(self, X):
        return self.kmeans.score(X)
