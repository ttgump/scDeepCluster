#### normalize the raw count data ####

import scanpy.api as sc
from preprocess import read_dataset, normalize
import h5py
import numpy as np

for i in range(1, 21):
    data_mat = h5py.File('splatter_simulate_data_'+str(i)+'.h5')
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])

    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                        transpose=False,
                        test_split=False,
                        copy=True)

    adata = normalize(adata,
                        size_factors=True,
                        normalize_input=False,
                        logtrans_input=True)

    h5f = h5py.File("../normalized_raw_data/splatter_simulate_data_normalized_"+str(i)+".h5", "w")
    h5f.create_dataset('X', data=adata.X, compression="gzip", compression_opts=9)
    h5f.create_dataset('Y', data=y, compression="gzip", compression_opts=9)
    h5f.close()
