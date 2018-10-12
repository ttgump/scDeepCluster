# DDCAC
Denoising Deep Count Autoencoder Clustering (scDeepCluster) for Single Cell RNA-seq data

Requirement:

Python --- 3.6.3

Keras --- 2.1.4

Tensorflow --- 1.1.0

Scanpy --- 1.0.4

Nvidia Tesla K80 (12G)

To run:

python DDCAC.py --data_file data.h5 --n_clusters 10

set data_file to the destination to the data (stored in h5 format, with two components X and Y, where X is the cell by gene count matrix and Y is the true labels), n_clusters to the number of clusters.

The output report the clustering performance, here is an example on 10X PBMC:

Final: ACC= 0.8100, NMI= 0.7736, ARI= 0.7841