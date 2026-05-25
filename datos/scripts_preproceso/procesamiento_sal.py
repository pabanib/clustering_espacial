#%%
from .lectura_datos_salarios import *
import pandas as pd
import geopandas as gpd
import SDEC.regionalizacion as reg
#import matplotlib.pyplot as plt
#import SDEC.regionalizacion.deep_cluster as dc
#from sklearn.cluster import KMeans, AgglomerativeClustering

#%%



class Salarios:
    def __init__(self):
        self.sal, self.geo, self.centroides, self.codiprov, self.personas = df_salarios()
        self.claes = pd.read_csv(DIR_DATOS_ORIG+"/clae_agg_corr.csv", sep=';')

    def detectar_outliers_iqr(self, data, miq=3):
        if len(data) < 4:
            return np.array([])
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - miq * iqr
        upper_bound = q3 + miq * iqr
        outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
        return outlier_indices

    def preparar_datos(self, variables, poblacion, log_transform=False):
        sal = self.sal.copy()
        atributos = sal.columns.tolist()[1:-10]
        atributos.append('personas')
        sal_ = sal.groupby(level=0)[atributos].rolling(12).mean()
        sal_.index = sal_.index.droplevel(0)
        sal_ = gpd.GeoDataFrame(sal_, geometry=sal['geometry'].values)
        sal = sal_.dropna()
        sal.loc[:, 'resto'] = sal.loc[:, 'A':'Z'].sum(axis=1) - sal.loc[:, 'C']
        datos = reg.Datos(sal, variables, poblacion)
        self.datos = datos
        df = datos.convertir_a_df(datos.panel_df, variables)
        if log_transform:
            prop = (np.log(df.iloc[:, :-1].values + 1)) / (np.log((df[poblacion].values.reshape(-1, 1) + 1)))
        else:
            prop = (df.iloc[:, :-1].values + 1) / (df[poblacion].values.reshape(-1, 1) + 1)
        self.prop = prop
        prop_st = (prop - prop.mean(axis=0)) / prop.std(axis=0)
        self.prop_st = prop_st
        return prop_st

    def generar_matriz(self, knn= 4):
        self.datos.matriz_W(knn)
        self.W = self.datos.W_knn
        return self.W
