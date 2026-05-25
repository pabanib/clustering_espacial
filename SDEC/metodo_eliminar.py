"""
Ejecuta la metodología de clustering elegida de acuerdo a distintos parámetros que se eligen para configurarla

"""

from copy import copy
import numpy as np
import pandas as pd
import time
import sys

class metodo():
    def __init__(self, metodo, param,metric):
        """
        Parameters: 
        metodo: It is the clustering metodology of scikit learn
        param: Dictionary of  metodos's parameters 
        metrci: Dictionary of {name: callable()}
        """
        self.metodo = metodo
        self.param = param
        self.metric = metric
    def grid(self):
        from sklearn.model_selection import ParameterGrid
        parametros = list(ParameterGrid(self.param))
        return parametros
    
    def modelo(self, diccionario_parametros):
        dic = diccionario_parametros
        kmean = {'n_clusters': 8, 'init': 'k-means++', 'n_init': 10, 'max_iter':300, 'tol': 0.0001}
        aglo = {'n_clusters': 8, 'affinity': 'euclidean', 'connectivity':None, 'linkage': 'ward'}
        tskmean = {'n_clusters': 8, 'init': 'k-means++', 'n_init': 1, 'max_iter':300, 'tol': 0.0001, 'metric': "dtw"}
        kshape = {'n_clusters': 8,  'max_iter':100, 'tol': 0.0001, 'init': 'random'}
        if str(type(self.metodo)) == "<class 'sklearn.cluster._kmeans.KMeans'>":
            for k in kmean.keys():
                try:
                    kmean[k] = dic[k] 
                except KeyError:
                    pass                  
            metod = copy(self.metodo)    
            metod.__init__(n_clusters = kmean['n_clusters'], init = kmean['init'], n_init = kmean['n_init'], tol = kmean['tol'])
            return metod
        
        elif str(type(self.metodo)) == "<class 'sklearn.cluster._agglomerative.AgglomerativeClustering'>":
            for k in aglo.keys():
                try:
                    aglo[k] = dic[k] 
                except KeyError:
                    pass #print('la clave {} no se encuentra'.format(k))
            metod = copy(self.metodo)    
            metod.__init__(n_clusters = aglo['n_clusters'], affinity = aglo['affinity'], connectivity = aglo['connectivity'],linkage = aglo['linkage'] )
            return metod
        
        elif str(type(self.metodo)) == "<class 'tslearn.clustering.kmeans.TimeSeriesKMeans'>":
            for k in tskmean.keys():
                try:
                    tskmean[k] = dic[k] 
                except KeyError:
                    pass                  
            metod = copy(self.metodo)    
            metod.__init__(n_clusters = tskmean['n_clusters'], init = tskmean['init'], n_init = tskmean['n_init'], tol = tskmean['tol'], metric = "dtw")
            return metod
        
        elif str(type(self.metodo)) == "<class 'tslearn.clustering.kshape.KShape'>":
            for k in kshape.keys():
                try:
                    kshape[k] = dic[k] 
                except KeyError:
                    pass                  
            metod = copy(self.metodo)    
            metod.__init__(n_clusters = kshape['n_clusters'], init = kshape['init'], tol = tskmean['tol'] )
            return metod
        
        else:
            print('no se reconoce el método')
            
            
    def fit(self, data):
        p = self.grid()
        self.parametros = p
        modelos = []
        metrics = []
        for dic in p:
            try:
                model = self.modelo(dic)
                inicio = time.time()
                model.fit(data)
                rdo = self.calc_metric(data, model)
                tiempo = time.time()-inicio
                modelos.append([model,tiempo, sys.getsizeof(model)])
                metrics.append(rdo.values[0])
            except:
                print('fallo el sig dic:')
                print(dic)
        modelos = pd.DataFrame(modelos, columns = ('modelo', 'tiempo','Tamaño'))
        self.metrics = pd.DataFrame(metrics, columns = rdo.columns)
        self.modelos = modelos
        self.best_model_ = self.modelos.iloc[self.best_model(self.metrics).index]
        self.best_metrics_ = self.metrics.iloc[self.best_model(self.metrics).index]
        self.best_time_ = self.modelos[self.modelos.tiempo == self.modelos.tiempo.min()]
        
    def calc_metric(self,data,model):
        #every metric maid have parameters: (data, model)
        metric_result = []
        for k in self.metric.keys():
            metric_result.append(self.metric[k](data,model.labels_))
        
        self.metric_result=pd.DataFrame(metric_result, index = self.metric.keys())
        #self.metric_result = pd.DataFrame(metric_result)
        return self.metric_result.T
    
    def best_model(self, metricas):
        # the best model have the most larger sum of normalizer metrics
        from sklearn.preprocessing import StandardScaler
        st = StandardScaler()
        m= st.fit_transform(metricas)
        return metricas[m.sum(axis = 1) == m.sum(axis = 1).max()]
