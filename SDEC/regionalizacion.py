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
import sklearn
import SDEC.procesos as procesos
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

