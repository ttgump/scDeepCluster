from time import time
import numpy as np
from keras.models import Model
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, BatchNormalization, GaussianNoise, Layer, Activation, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.losses import kullback_leibler_divergence

from sklearn.cluster import KMeans
from sklearn import metrics
import scipy

import h5py
import scanpy.api as sc
from layers import ConstantDispersionLayer, SliceLayer, ColWiseMultLayer
from loss import poisson_loss, NB, ZINB
from preprocess import read_dataset, normalize
import tensorflow as tf

from numpy.random import seed
seed(2211)
from tensorflow import set_random_seed
set_random_seed(2211)

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

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


def autoencoder(dims, noise_sd=2.5, init='glorot_uniform', act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    sf_layer = Input(shape=(1,), name='size_factors')
    x = Input(shape=(dims[0],), name='counts')
    h = x
    h = GaussianNoise(noise_sd, name='input_noise')(h)

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], kernel_initializer=init, name='encoder_%d' % i)(h)
        h = GaussianNoise(noise_sd, name='noise_%d' % i)(h)
        h = Activation(act)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_hidden')(h)  # hidden layer, features are extracted from here

    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        h = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(h)

    # output

    pi = Dense(dims[0], activation='sigmoid', kernel_initializer=init, name='pi')(h)

    disp = Dense(dims[0], activation=DispAct, kernel_initializer=init, name='dispersion')(h)

    mean = Dense(dims[0], activation=MeanAct, kernel_initializer=init, name='mean')(h)

    output = ColWiseMultLayer(name='output')([mean, sf_layer])
    output = SliceLayer(0, name='slice')([output, disp, pi])

    return Model(inputs=[x, sf_layer], outputs=output)


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class DCAC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 noise_sd=3,
                 alpha=1.0,
                 ridge=0,
                 debug=False):

        super(DCAC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.noise_sd = noise_sd
        self.alpha = alpha
        self.act = 'relu'
        self.ridge = ridge
        self.debug = debug
        self.autoencoder = autoencoder(self.dims, noise_sd=self.noise_sd, act = self.act)

        # prepare clean encode model
        ae_layers = [l for l in self.autoencoder.layers]
#        print(ae_layers)
        hidden = self.autoencoder.input[0]
        for i in range(1, len(ae_layers)):
            if "noise" in ae_layers[i].name:
                next
            elif "dropout" in ae_layers[i].name:
                next
            else:
                hidden = ae_layers[i](hidden)
            if "encoder_hidden" in ae_layers[i].name:  # only get encoder layers
                break
#        hidden = self.autoencoder.get_layer(name='encoder_hidden').output
        self.encoder = Model(inputs=self.autoencoder.input[0], outputs=hidden)

        pi = self.autoencoder.get_layer(name='pi').output
        disp = self.autoencoder.get_layer(name='dispersion').output
        mean = self.autoencoder.get_layer(name='mean').output
        zinb = ZINB(pi, theta=disp, ridge_lambda=self.ridge, debug=self.debug)
        self.loss = zinb.loss

        clustering_layer = ClusteringLayer(self.n_clusters, alpha=self.alpha, name='clustering')(hidden)
        self.model = Model(inputs=self.autoencoder.input[0],
                           outputs=clustering_layer)

        self.pretrained = False
        self.centers = []
        self.y_pred = []

    def pretrain(self, x, y, batch_size=256, epochs=200, optimizer='adam', ae_file='ae_weights.h5'):
        print('...Pretraining autoencoder...')
        self.autoencoder.compile(loss=self.loss, optimizer=optimizer)
        es = EarlyStopping(monitor="loss", patience=50, verbose=1)
        self.autoencoder.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, callbacks=[es])
        self.autoencoder.save_weights(ae_file)
        print('Pretrained weights are saved to ./' + str(ae_file))
        self.pretrained = True

    def load_weights(self, weights_path):  # load weights of DCAC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    def get_loss(self, x_counts): # evaluate loss
        q = self.model.predict(x_counts)
        p = self.target_distribution(q)
        return self.model.evaluate(x=x_counts, y=p, batch_size=256)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def fit(self, x_counts, y, batch_size=256, maxiter=2e4, tol=1e-3, update_interval=140,
            ae_weights=None, save_dir='./results/dcac', optimizer='adam'):
        print('optimizer: ', optimizer)
        self.model.compile(loss='kld', optimizer=optimizer)

        print('Update interval', update_interval)
        save_interval = int(x_counts.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        # Step 1: pretrain
        if not self.pretrained and ae_weights is None:
            print('...pretraining autoencoders using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x, batch_size)
            self.pretrained = True
        elif ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('ae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x_counts))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/dcac_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L'])
        logwriter.writeheader()

        loss = 0
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x_counts, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(cluster_acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logwriter.writerow(dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss))
                    print('Iter-%d: ACC= %.4f, NMI= %.4f, ARI= %.4f;  L= %.5f'
                          % (ite, acc, nmi, ari, loss))

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x_counts.shape[0]:
                loss = self.model.train_on_batch(x=x_counts[index * batch_size::],
                                                 y=p[index * batch_size::])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x_counts[index * batch_size:(index + 1) * batch_size],
                                                 y=p[index * batch_size:(index + 1) * batch_size])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save DCAC model checkpoints
                print('saving model to: ' + save_dir + '/DCAC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DCAC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to: ' + save_dir + '/DCAC_model_final.h5')
        self.model.save_weights(save_dir + '/DCAC_model_final.h5')

        return self.y_pred


if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--update_interval', default=0, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/dcac')
    parser.add_argument('--ae_weight_file', default='ae_weights.h5')
    parser.add_argument('--data_file', default='data.h5')

    args = parser.parse_args()

    # load dataset
    optimizer1 = Adam(amsgrad=True)
    optimizer2 = 'adadelta'

    data_mat = h5py.File(args.data_file)
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
                      normalize_input=True,
                      logtrans_input=True)

    input_size = adata.n_vars

    print(adata.X.shape)
    print(y.shape)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)

#    args = parser.parse_args()
    if args.update_interval == 0:  # one epoch
        args.update_interval = int(adata.X.shape[0]/args.batch_size)
    print(args)


    # Define DCAC model
    dcac = DCAC(dims=[input_size, 256, 64, 32], n_clusters=args.n_clusters, noise_sd=2.5)
    plot_model(dcac.model, to_file='dcac_model.png', show_shapes=True)
    print("autocoder summary")
    dcac.autoencoder.summary()
    print("model summary")
    dcac.model.summary()

    t0 = time()
    # split training and validating data
    indices = np.random.permutation(adata.X.shape[0])
    training_idx, validate_idx = indices[:int(0.9*len(indices))], indices[int(0.1*len(indices)):]
    X_train = adata.X[training_idx,:]
    size_factors_train = adata.obs.size_factors[training_idx]
    Y_train = y[training_idx]
    raw_X_train = adata.raw.X[training_idx,:]

    X_validate = adata.X[validate_idx,:]
    size_factors_validate = adata.obs.size_factors[validate_idx]
    Y_validate = y[validate_idx]
    raw_X_validate = adata.raw.X[validate_idx,:]

    # Pretrain autoencoders before clustering
    if args.ae_weights is None:
        dcac.pretrain(x=[X_train, size_factors_train], y=raw_X_train, batch_size=args.batch_size, epochs=args.pretrain_epochs, optimizer=optimizer1, ae_file=args.ae_weight_file)

    # begin clustering, time not include pretraining part.

    dcac.fit(x_counts=X_train, y=Y_train, batch_size=args.batch_size, tol=args.tol, maxiter=args.maxiter,
             update_interval=args.update_interval, ae_weights=args.ae_weights, save_dir=args.save_dir, optimizer=optimizer2)

    # Show the final results
    y_pred = dcac.y_pred
    acc = np.round(cluster_acc(y[training_idx], dcac.y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y[training_idx], dcac.y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y[training_idx], dcac.y_pred), 5)
    print('training data Final: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
    print('training data clustering time: %d seconds.' % int(time() - t0))

    # Caluate loss for training and validation data
    train_loss = dcac.get_loss(x_counts=X_train)
    validate_loss = dcac.get_loss(x_counts=X_validate)

    print('training cluster loss: %.4f, validating cluster loss: %.4f' % (train_loss, validate_loss))
    G = train_loss/validate_loss
    print('Generalizability: %.4f' % G)

    y_validate_pred = dcac.predict_clusters(x=X_validate)
    validate_acc = np.round(cluster_acc(y[validate_idx], y_validate_pred), 5)
    validate_nmi = np.round(metrics.normalized_mutual_info_score(y[validate_idx], y_validate_pred), 5)
    validate_ari = np.round(metrics.adjusted_rand_score(y[validate_idx], y_validate_pred), 5)
    print('validate data: Final: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (validate_acc, validate_nmi, validate_ari))
