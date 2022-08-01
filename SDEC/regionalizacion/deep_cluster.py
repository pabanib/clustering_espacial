# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:58:48 2022

@author: Pablo
"""

from time import time
import numpy as np
import keras.backend as K
from keras.layers import Layer, InputSpec
from keras.layers import Dense, Input, concatenate
from tensorflow.keras import regularizers
from keras.models import Model
from tensorflow.keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans,AgglomerativeClustering
import matplotlib.pyplot as plt


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
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
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
        self.clusters = self.add_weight(shape = (self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

np.random.seed(5465)
n_enc = 6

class sdec():
    def __init__(self, n_enc, inp_shape = [100,100]):
        
        self.n_enc = n_enc
        self.n_clusters = n_enc
        self.inp1_shape = inp_shape[0]
        self.inp2_shape = inp_shape[1]
    
    def gen_modelo(self):
        
        inp1 = Input(shape = [self.inp1_shape,])
        inp2 = Input(shape = [self.inp2_shape,])
        dense1 = Dense(80, activation = "relu")(inp1)
        dense1 = Dense(40, activation = "relu")(dense1)
        dense1 = Dense(10, activation = "relu")(dense1)
        dense2 = Dense(80, activation = "relu")(inp2)
        dense2 = Dense(40, activation = "relu")(dense2)
        dense2 = Dense(10, activation = "relu")(dense2)
        concat = concatenate([dense1,dense2])
        concat = Dense(50, activation = "relu")(concat)
        enco = Dense(n_enc, activation = "relu")(concat)
        dense = Dense(40, activation = "relu",  kernel_regularizer = regularizers.l2(0.01))(enco)
        dense = Dense(80, activation = "relu",  kernel_regularizer = regularizers.l2(0.01))(dense)
        decoder = Dense(self.inp1_shape+self.inp1_shape, activation = "relu",  kernel_regularizer = regularizers.l2(0.01))(dense)
        
        autoencoder = Model(inputs = [inp1,inp2], outputs = decoder)
        encoder = Model(inputs = [inp1,inp2], outputs = enco)
        autoencoder.compile(optimizer = 'adam', loss = 'cosine_similarity')

        self.autoencoder = autoencoder
        self.encoder = encoder
   
        n_clusters = self.n_enc
        clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
        model = Model(inputs=encoder.input, outputs=clustering_layer)
        model.compile(optimizer='adam', loss='kld')
        self.model = model

    def train_autoencoder(self, inp, out):
        
        self.autoencoder.fit(inp,out, epochs = 100, validation_split = 0.2 , verbose = False) 

    def ajustar_modelo(self, X, W, cc = []):
        
        km = KMeans(self.n_clusters)
        km.fit(self.encoder.predict(X))
        if cc == []:
            cc = km.cluster_centers_
        else:
            cc = cc
        y_pred = km.labels_
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name="clustering").set_weights([cc])

        def target_distribution(q):
            weight = W.todense()@(q ** 2 / q.sum(0))
            return (weight.T / weight.sum(1).reshape(len(q),)).T
        
        loss = 0
        index = 0
        maxiter = 1000
        update_interval = 20
        
        tol = 0.0001 # tolerance threshold to stop training
        
        aglo = AgglomerativeClustering(self.n_clusters, connectivity=W)
        
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(X, verbose=0)
                p = target_distribution(q)  # update the auxiliary target distribution p
        
                # evaluate the clustering performance
                y_pred = aglo.fit_predict(q)
                #y_pred = q.argmax(1)
                
                """
                if y is not None:
                    acc = 0 #np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)     """
                # check stop criterion - model convergence
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            loss = self.model.fit(x=[X[0],X[1]], y=p, verbose = False)

        self.y_pred = aglo.fit_predict(q)

version = 5



