# scDeepCluster
scDeepCluster, a model-based deep embedding clustering for Single Cell RNA-seq data. See details in our paper: "Clustering single-cell RNA-seq data with a model-based deep learning approach" published in Nature Machine Intelligence https://www.nature.com/articles/s42256-019-0037-0.

## Table of contents
- [Network diagram](#diagram)
- [Requirements](#requirements)
- [Usage](#usage)
- [Pytorch version](#pytorch_version)
- [Raw data](#data)
- [Online app](#app)
- [Contact](#contact)

## <a name="diagram"></a>Network diagram

![alt text](https://github.com/ttgump/scDeepCluster/blob/master/network.png?raw=True)

## <a name="requirements"></a>Requirements

Python --- 3.6.3

Keras --- 2.1.4

Tensorflow --- 1.1.0

Scanpy --- 1.0.4

Nvidia Tesla K80 (12G)

Please note that if using different versions, the results reported in our paper might not be able to repeat.

## <a name="usage"></a>Usage

```sh
python scDeepCluster.py --data_file data.h5 --n_clusters 10
```

set data_file to the destination to the data (stored in h5 format, with two components X and Y, where X is the cell by gene count matrix and Y is the true labels), n_clusters to the number of clusters.

The final output reports the clustering performance, here is an example on 10X PBMC scRNA-seq data:

Final: ACC= 0.8100, NMI= 0.7736, ARI= 0.7841

## <a name="pytorch_version"></a>Pytorch version

**Recommend the pytorch version, I have added some new features:** 1. automatically estimating number of clusters after pretraining; 2. clustering on datasets from different batches. 

See detail at https://github.com/ttgump/scDeepCluster_pytorch

## <a name="data"></a>Raw data

The raw data used in this paper can be found: https://figshare.com/articles/dataset/scDeepCluster_supporting_data/17158025

## <a name="app"></a>Online app

Online app website: https://app.superbio.ai/apps/107

## <a name="contact"></a>Contact

Tian Tian tiantianwhu@163.com
