# scDeepCluster
scDeepCluster, a novel model-based deep embedded clustering for Single Cell RNA-seq data. See details in our paper: "Clustering single-cell RNA-seq data with a model-based deep learning approach" published in Nature Machine Intelligence https://www.nature.com/articles/s42256-019-0037-0.

Requirement:

Python --- 3.6.3

Keras --- 2.1.4

Tensorflow --- 1.1.0

Scanpy --- 1.0.4

Nvidia Tesla K80 (12G)

Usage:

python scDeepCluster.py --data_file data.h5 --n_clusters 10

set data_file to the destination to the data (stored in h5 format, with two components X and Y, where X is the cell by gene count matrix and Y is the true labels), n_clusters to the number of clusters.

The final output reports the clustering performance, here is an example on 10X PBMC scRNA-seq data:

Final: ACC= 0.8100, NMI= 0.7736, ARI= 0.7841
