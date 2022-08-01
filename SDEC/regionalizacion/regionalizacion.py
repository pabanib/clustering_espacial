# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:11:10 2021
@author: pabanib


"""
#from sklearn.cluster import KMeans, AgglomerativeClustering
from copy import copy
import numpy as np
import pandas as pd
import geopandas as gpd
#import metodo 
#import lq
import time
import sys
import sklearn
import SDEC.regionalizacion.procesos as procesos
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, PolynomialFeatures
import matplotlib.pyplot as plt
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran_Local

class Datos():
    
    def __init__(self, df, variables, poblacion):
        
        assert isinstance(df,gpd.geodataframe.GeoDataFrame), "El DataFrame no es un GeoDataFrame"
        
        if isinstance(df.index,pd.core.indexes.multi.MultiIndex):
            self.panel_df = df
            p = df.index.get_level_values(1).unique()[0]
            self.geo = df.loc[pd.IndexSlice[:,p],'geometry']
            self.df = [False]
            self.centroides = self.geo.centroid.reset_index(level = 1, drop = True)
            
        elif  isinstance(df.index,pd.core.indexes.base.Index):
            self.df = df
            self.geo = df['geometry']
            self.panel_df = [False]
            self.centroides = self.geo.centroid
                
        
        self.variables = variables
        self.poblacion = poblacion
        
        self.add_centroides = procesos.agrega_centroides(self.centroides)
        self.coord_centroides = self.add_centroides.coordenadas(self.centroides)
        
        
    def add_pipeline(self, pipeline):
        
        assert isinstance( pipeline,sklearn.pipeline.Pipeline ), 'Debe ingresar un pipeline de sklearn'
        self.pipeline = pipeline
        
    def ajustar_datos(self):

        pass
       
    def agregar_metrica(self,nombre, metrica):
        
        self.metricas[nombre] = metrica

    def convertir_a_panel(self, peris,variables = 1):
        
        df = self.df.drop('geometry', axis = 1)
        if variables > 1:
            df_ = df[df.columns[:-1]].stack()
            if isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
                columnas = list(df_.columns.get_level_values(0).unique())
            else:
                columnas = []
                for i in range(variables):
                    vari = self.variables[i]
                    columnas.append(vari)
                              
        else:
            df_ = df[df.columns[:list(df.columns).index(self.poblacion)]].stack()
            columnas = list(self.variables)
        columnas.append(self.poblacion)
        p = np.array([[df[self.poblacion].values,]]*peris)
        
        X = np.c_[df_.values, p.T.reshape(-1,1)]
        return pd.DataFrame(X, index = df_.index, columns = columnas)        
    
    def convertir_a_df(self, panel_df,variables):
        if all(self.panel_df) == False:
            df = self.df
        else:
            df = panel_df[variables].unstack()
        
            p = panel_df.index.get_level_values(1).unique()[0]
            df[self.poblacion] = panel_df.loc[pd.IndexSlice[:,p],:].set_index(df.index)[self.poblacion]
        
        return df
       
    def agregar_geometria(self):
        df = self.convertir_a_df(self.panel_df, self.variables)
        geo = self.geo.copy()
        geo.index = df.index
        df = gpd.GeoDataFrame(df, geometry = geo.values)
        
        return df
       
    def matriz_W(self, k = 6):
                
        df = self.agregar_geometria()
        self.W_queen = Queen.from_dataframe(df)
        self.W_rook = Rook.from_dataframe(df)
        self.W_knn = KNN.from_dataframe(df, k = k)
        
    def calc_Imoran(self, W, matriz = np.array([])):
        
        if len(matriz) > 0:
            df = matriz
        else:
            if all(self.df) == False:
                df = self.convertir_a_df(self.panel_df, self.variables)
                df = df.values
            else:
                df = self.df.drop('geometry', axis = 1)
                df = df.values
            
        I_locales = []
        for c in range(df.shape[1]):
            I = Moran_Local(df[:,c], W).Is
            I_locales.append(I)
        
        return np.array(I_locales).T
    def calc_prom_vec(self, W, matriz = np.array([])):
        if len(matriz) > 0:
            df = matriz
        else:
            if all(self.df) == False:
                df = self.convertir_a_df(self.panel_df, self.variables)
                df = df.values
            else:
                df = self.df.drop('geometry', axis = 1)
                df = df.values
                
        
        prom_vec =  W.sparse.toarray()@ df
                
        return prom_vec
    
    def separar_variables(self):
        v = copy(self.variables)
        v.append(self.poblacion)
        
        if all(self.df) == False:
            df = self.convertir_a_df(self.panel_df, self.variables)
            #df = df[v].values
            n = len(df)
            dfs = {}
            for i in v:
                d = df[[i]].values.reshape(n,-1)
                dfs[i] = d
            
        else:
            df = self.df.drop('geometry', axis = 1)
            n = len(df)
            df = df[v].values
            
            dfs = {}
            for i in range(df.shape[1]):
               d = df[:,i] 
               dfs[v[i]] = d.reshape(n,-1)
               
        #dfs[self.poblacion] =  df.loc[:,pd.IndexSlice[self.poblacion,:]] 
        return dfs

class dic_datos():
    def __init__(self, pipeline):
        assert isinstance( pipeline,sklearn.pipeline.Pipeline ), 'Debe ingresar un pipeline de sklearn'
        self.pipeline = pipeline
        self.dic = {}
            
    def agregar_data(self, key, df, ajustar = True):
        
        if ajustar:
            
            v = self.pipeline.fit_transform(df) if df.shape[0] > 1 else self.pipeline.fit_transform(df).T
            assert np.all(v == 0) ==  False, "Todos los valores son iguales"    
            self.dic[key] = v
        else:
            self.dic[key] = df if df.shape[0] > 1 else df.T
        
        
        
    def retornar_dfs(self, todo = False, separado = []):
        
        dfs = []
        if todo:
            l = list(self.dic.keys())
            df = self.dic[l[0]]
            for k in l[1:]:
                df = np.c_[df, self.dic[k]]
            dfs.append(df)    
        else:
            for i in separado:
                if isinstance(i, list):
                    df = self.dic[i[0]]
                    for j in i[1:]:
                        df = np.c_[df,self.dic[j]]
                    dfs.append(df)
                else:
                    df = self.dic[i]
                    dfs.append(df)
                    
        return dfs
    
class autoencoders():
    
    def __init__(self, n_encoders):
        self.n_encoders = n_encoders
    
    def crear_autoencoders(self, lista_df,**kwargs):
        n = len(lista_df)
        shape_inp = []
        self.inputs = []
        self.encoders = []
        for i in range(n):
            s = lista_df[i].shape[1]
            shape_inp.append(s)
            inp = layers.Input(shape = [s,])
            enc = layers.Dense(self.n_encoders, activation = "relu", kernel_regularizer = regularizers.l2(0.01))(inp)        
            self.inputs.append(inp)
            self.encoders.append(enc)
            
        concat = layers.concatenate(self.encoders)
        encoder = layers.Dense(self.n_encoders, activation = "relu" )(concat)
        
        decoder = layers.Dense(sum(shape_inp), activation = "sigmoid")(encoder)
        
        self.autoencoder = Model(inputs = self.inputs, outputs = decoder)
        if kwargs == {}:
            self.autoencoder.compile(optimizer = "sgd", loss = "mse")
        else:
            self.autoencoder.compile(**kwargs)
        
        self.enco = Model(inputs = self.inputs, outputs = encoder)
        
        return self
    
    def fit_autoencoders(self, inputs, outputs, epochs,**kwargs):
        
        self.crear_autoencoders(inputs,**kwargs)
        history = self.autoencoder.fit(inputs,outputs,epochs = epochs, verbose = 0)
        return self.enco.predict(inputs)
from ..coef_loc import regiones as reg
class calcular_metricas():
  
    def __init__(self, datos, regiones = []):
        
        #assert isinstance(datos, Datos), "Datos debe ser un objeto tipo regionalizacion.datos"
        self.datos = datos
        if all(datos.panel_df) == False:
            df = datos.df
            print("convertir a panel los datos")
        
        else:
            df = datos.convertir_a_df(datos.panel_df, datos.variables)
            self.lqperis = reg.paneldf_to_region(self.datos.panel_df, self.datos.poblacion, variables = self.datos.variables)
            #for c in datos.variables:
            #    V = datos.panel_df[[c]].unstack().values
            #    P = datos.panel_df[[datos.poblacion]].unstack().values
            #    lq_ = reg.region(V,P)
                #lq_ = lq.lq_peri(datos.panel_df[[c,datos.poblacion]])
            #    self.lqperis.append(copy(lq_))
                   
        
        self.df = df
        self.pobl = datos.poblacion
        self.regiones = regiones
    
    def obtener_panel(self, periodos, variables):
        panel_df = self.datos.convertir_a_panel(periodos, variables)
        self.lqperis = reg.paneldf_to_region(self.datos.panel_df, self.datos.poblacion, variables = self.datos.variables)
        #for c in self.datos.variables:
        #    V = self.datos.panel_df[[c]].unstack().values
        #    P = self.datos.panel_df[[self.datos.poblacion]].unstack().values
        #    lq_ = reg.region(V,P)
        #    #lq_ = lq.lq_peri(panel_df[[c,self.datos.poblacion]])
        #    self.lqperis.append(copy(lq_))
                                
    
    def indice_lq(self, X, grupos):
        
        ind = 0
        for i in self.lqperis:
            t = i.convertir_a_territ(grupos)
            ind += t.calc_ind_lq()
            #ind += i.calcular_indice_debil(grupos)
       
        return 1-((ind)/len(self.lqperis))
   
    
    def Hg_relat(self,X,grupos):
        try:
            df = self.df.drop('geometry', axis = 1)
        except:
            df = self.df
        return 1-lq.homog_relat2(df,grupos, list(df.columns)[-1])

    def MI(self,X,grupos):
        real = self.regiones
        if len(real) == 0:
            m = np.nan
        else:
            m = sklearn.metrics.adjusted_mutual_info_score(real,grupos) 
        return m

    def compara_metricas(self,modelo, datos):
        data = datos
        result = []
        for m in range(len(modelo.modelos)):
            model = modelo.modelos.iloc[m]['modelo']
            mi = self.MI(data,model.labels_)
            ilq = self.indice_lq(data, model.labels_)
            hgr = self.Hg_relat(data, model.labels_)
            sil = siluetas(data, model.labels_)
            cal = calinski(data, model.labels_)
            dav = inv_davies_bouldin(data,model.labels_)
            try:
                iner = model.inertia_
            except:
                iner = np.nan
            result.append(np.array([mi,ilq,hgr,sil,cal,dav,iner]))
        result = pd.DataFrame(result, columns = ['mi','ilq','hgr','sil','cal','dav'])
        return result
    def calc_metricas(self,datos, grupos):
        data = datos
        result = []
        
        mi = self.MI(data,grupos)
        ilq = self.indice_lq(data, grupos)
        hgr = self.Hg_relat(data, grupos)
        sil = siluetas(data, grupos)
        cal = calinski(data, grupos)
        dav = inv_davies_bouldin(data,grupos)
        result.append(np.array([mi,ilq,hgr,sil,cal,dav]))
        result = pd.DataFrame(result, columns = ['mi','ilq','hgr','sil','cal','dav'])
        return result


def siluetas(X, grupos):
    return sklearn.metrics.silhouette_score(X, grupos)
def calinski(x, grupos):
    return sklearn.metrics.calinski_harabasz_score(x, grupos)
def inv_davies_bouldin(X,grupos):
    return (sklearn.metrics.davies_bouldin_score(X,grupos))**-1

def mapa_grupos(geodf_,model):
    geodf = geodf_.copy()
    geodf['grup'] = model.best_model_['modelo'].iloc[0].labels_
    gpd.GeoDataFrame(geodf).plot(geodf['grup']+1, figsize = (15,12),
                                 categorical = True, legend = True, 
                                 legend_kwds = {'loc':'lower right'})
    plt.axis('off')
    #plt.legend(loc = 'lower right')
    plt.show()
    
def metric_grup(model):
    std = sklearn.preprocessing.StandardScaler()
    for i in model.metrics.columns:
        met = model.metrics[i]
        met = std.fit_transform(met.values.reshape(-1,1))
        plt.scatter(model.modelos['modelo'].apply(lambda x: x.n_clusters) , met, label = i)
    plt.legend()
    plt.show()
            
class entorno(Datos, dic_datos, autoencoders):
    
    def __init__(self, df, variables, poblacion, pipeline):
        # df es un dataframes
        # variables son las variables a intervenir, sin tener en cuento los periodos temporales
        # población es una variable poblacional
        # pipeline es una objeto tipo sklearn.preprocessing.Pipeline 

        Datos.__init__(self,df, variables,poblacion)
        dic_datos.__init__(self,pipeline)
        
        self.matriz_W(6)
        self.W = self.W_knn
        
        dat = Datos(df, variables, poblacion)
        self.metric = calcular_metricas(dat)
        self.metodos_rdos = {}
        self.met_desc = {}
        self.Metodos = {}
        self.Metodos_variacion = {}
        
    def procesar_datos(self):
        
        d = self.separar_variables()
        self.columnas = list(d.keys())
        for c in self.columnas:
            self.agregar_data(c, d[c])
        self.agregar_data('coord',self.coord_centroides)
        self.agregar_data('I_Moran', self.calc_Imoran(self.W))
        self.agregar_data('prom_vecinos', self.calc_prom_vec(self.W))
        
    def agregar_metodo(self,nombre ,metodo_, param, metricas):
      
        self.Metodos[nombre] = metodo(metodo_,param,metricas)
        self.Metodos_variacion[nombre] = 0
        self.metodos_rdos[nombre] = {}
        self.met_desc[nombre] = {}
        
    def calcular_metodo(self,nombre_metodo, columnas_input, ae = False, n_encoders = 2, centroides = True,**kwargs):
        
        num_var = self.Metodos_variacion[nombre_metodo] +1
        self.Metodos_variacion[nombre_metodo] = num_var
       
        self.met_desc[nombre_metodo][num_var] = {'columnas': columnas_input, 'ae':ae}
        
        
        if ae:
            autoencoders.__init__(self, n_encoders)    
            col_outputs = []
            for c in columnas_input:
                if isinstance(c,list):
                    for cc in c:
                        col_outputs.append(cc)
                else:
                    col_outputs.append(c)                        
            inp = self.retornar_dfs(separado = columnas_input)
            out = self.retornar_dfs(separado = [col_outputs])[0]
            
            self.fit_autoencoders(inp, out, epochs = 500,**kwargs)
            X = self.enco.predict(inp)
            
        else:
            X = self.retornar_dfs(separado = columnas_input)[0]
        
        if centroides:
            X = np.c_[X, self.dic['coord']]
        else:
            X = X
        
        self.Metodos[nombre_metodo].fit(X)
         
        self.metodos_rdos[nombre_metodo][num_var] =  copy(self.Metodos[nombre_metodo]) 
        
        return self.Metodos[nombre_metodo].best_metrics_
    
    def mapa(self, nombre_metodo, vers = 0):
        if all(self.df) == False:
            df = self.convertir_a_df(self.panel_df, self.variables)
            self.df = gpd.GeoDataFrame(df, geometry = self.geo.values)
        if vers ==0:
            mapa_grupos(self.df, self.Metodos[nombre_metodo] )
        else:
            mapa_grupos(self.df, self.metodos_rdos[nombre_metodo][vers])
        
    def interacciones(self):
        self.poly = sklearn.preprocessing.PolynomialFeatures(2)
        datos = self.retornar_dfs(separado = [self.columnas])[0]
        datos = self.poly.fit_transform(datos)
        self.agregar_data('interac', datos, ajustar = False)
        i_moran = self.calc_Imoran(self.W, matriz = datos)
        prom_vec = self.calc_prom_vec(self.W, matriz = datos)
        
        self.agregar_data('interac_Imoran', i_moran, ajustar = False)
        self.agregar_data('interac_prom_vec', prom_vec, ajustar = False)
        

version = 3   

#%%

# =============================================================================
# ent1 = entorno(var, ['var1'], 'personas', pipe)
# ent1.procesar_datos()
# 
# param = {'n_clusters' : [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
# metricas = {'hg': metrics.Hg_relat, 'lq': metrics.indice_lq }
# 
# from sklearn.cluster import KMeans, AgglomerativeClustering
# 
# ent1.agregar_metodo('km', KMeans(), param, metricas)
# 
# ent1.calcular_metodo('km', [['var1','personas']])
# ent1.mapa('km')
# ent1.calcular_metodo('km', [['var1','personas']], ae = True)
# ent1.mapa('km')
# ent1.mapa('km', vers = 2)
# 
# ent1.metodos_rdos
# ent1.met_desc
# 
# ent1.interacciones()
# ent1.calcular_metodo('km', ['interac'])
# ent1.mapa('km')
# 
# 
# 
# ent1.retornar_dfs(separado = ['interac'])[0].shape
# d = ent1.retornar_dfs(separado = [ent1.columnas])[0]#.shape
# 
# dd = pipe.fit_transform(ent1.poly.fit_transform(d))
# ent1.dic['interac_Imoran']
# d.std(axis = 0)
# =============================================================================


