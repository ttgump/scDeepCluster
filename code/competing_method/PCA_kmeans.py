#### PCA + kmeans on the normalized count data ####

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py
from sklearn import metrics

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

for i in range(1, 21):
    data_mat = h5py.File('./normalized_raw_data/splatter_simulate_data_normalized_'+str(i)+'.h5')
    x = np.array(data_mat['X'])
    y_true = np.array(data_mat['Y'])
    pca = PCA(n_components=2)
    x_pca = pca.fit(x).transform(x)

    kmeans = KMeans(n_clusters=3, n_init=20)
    y_pred = kmeans.fit(x_pca).labels_

    acc = np.round(cluster_acc(y_true, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
    print('PCA + k-means result '+str(i)+': ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

