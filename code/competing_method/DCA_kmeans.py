#### deep count autoencoder (DCA) + k-means ####

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy.api as sc
from dca.api import dca
from preprocess import read_dataset, normalize
from sklearn.cluster import KMeans
from sklearn import metrics

import h5py

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


if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--data_file', default=None)
    args = parser.parse_args()

    data_mat = h5py.File('./normalized_raw_data/'+str(args.data_file))
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])

    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

    sc.pp.filter_genes(adata, min_counts=1)
    dca(adata, threads=1)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    print(adata)

    dca_pca = adata.obsm.X_pca[:, :2]

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(dca_pca)


    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print('data: '+str(args.data_file)+' DCA+PCA+kmeans: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))