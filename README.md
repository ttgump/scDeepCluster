# scDeepCluster
scDeepCluster, a model-based deep embedding clustering for Single Cell RNA-seq data. See details in our paper: "Clustering single-cell RNA-seq data with a model-based deep learning approach" published in Nature Machine Intelligence https://www.nature.com/articles/s42256-019-0037-0.

![alt text](https://github.com/ttgump/scDeepCluster/blob/master/network.png?raw=True)

Requirement:

Python --- 3.6.3

Keras --- 2.1.4

Tensorflow --- 1.1.0

Scanpy --- 1.0.4

Nvidia Tesla K80 (12G)

Please note that if using different versions, the results reported in our paper might not be able to repeat.

Usage:

python scDeepCluster.py --data_file data.h5 --n_clusters 10

set data_file to the destination to the data (stored in h5 format, with two components X and Y, where X is the cell by gene count matrix and Y is the true labels), n_clusters to the number of clusters.

The final output reports the clustering performance, here is an example on 10X PBMC scRNA-seq data:

Final: ACC= 0.8100, NMI= 0.7736, ARI= 0.7841

**Recommend the pytorch version**, I add some new features, see https://github.com/ttgump/scDeepCluster_pytorch

The raw data used in this paper can be found: https://figshare.com/articles/dataset/scDeepCluster_supporting_data/17158025

**Online app**

Online app website: https://app.superbio.ai/apps/107?id=62712ec148139943a4273ae1
