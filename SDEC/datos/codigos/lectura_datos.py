# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:33:45 2021

El archvo lee los datos del covid realiza la trasnformaciones de las variables
necesarias para trabajarlas. 
Agrega los datos geográficos y guarda un archivo shp listo para trabajar

@author: paban
"""

import pandas as pd 
import geopandas as gpd
import os
import numpy as np
from . import preproceso as pre 
import pyproj
from ..constants import FILE_DATADIR_PYPROJ, DIR_PRINCIPAL, EPSG
pyproj.datadir.set_data_dir(FILE_DATADIR_PYPROJ)

#dir_principal = os.getcwd()
dir_datos = DIR_PRINCIPAL+'\\datos'

def __genera_shp():
    covid = pd.read_csv(dir_datos+'/Covid19Casos.csv')
    
    # Tranforma los datos para poder trabajarlos
    
    gen_datos = pre.preparar_datos(covid)
    del(covid)
    covid_geo = gen_datos.fit()
    
    import pickle 
    with open("datos/covid_geo.pickle", "wb") as f:
        pickle.dump(covid_geo, f)
    
    # Agrega condiciones espaciales a los datos
    Arg = gpd.read_file(DIR_PRINCIPAL+'/shape_files/pxdptodatosok.shp')
    
    Arg = Arg.drop(Arg.query("link in('94028','94021')").index, axis = 0) 
    Arg['link'] = Arg['link'].astype(int)
    Arg['mujeres'] = Arg['mujeres'].astype(int) 
    Arg['varones'] = Arg['varones'].astype(int)
    Arg['personas'] = Arg['personas'].astype(int)
    Arg['hogares'] = Arg['hogares'].astype(int)
    Arg['viv_part'] = Arg['viv_part'].astype(int)
    Arg['viv_part_h'] = Arg['viv_part_h'].astype(int)
    
    claves = ['residencia_dpto','fecha_apertura']
    atributos = ['clasificacion_resumen_Confirmado','fallecido']
    
    periodos = pd.PeriodIndex(covid_geo.fecha_apertura, freq = 'm')
    agrupa = covid_geo.groupby(['residencia_dpto', periodos])
    covid_geo = agrupa[atributos].sum()
    
    index = covid_geo.index
    idx = pd.IndexSlice
    
    covid_periodos = {}
    
    for i in index.get_level_values(1):
        dic = {}
        df = pd.merge(covid_geo.loc[idx[:,i],:], Arg['link'],how= 'right' ,left_on= 'residencia_dpto', right_on = 'link')
        df = df.fillna(0)
        df['link'] = df['link'].astype(int)
        df = pd.merge(df, Arg,how= 'right' ,left_on= 'link', right_on = 'link')
        dic['df'] = gpd.GeoDataFrame(df)
        covid_periodos[i] = dic
        del(dic)
    
    # controla que no quede ningún valor nulo
    for k in covid_periodos.keys():
        print(covid_periodos[k]['df'].isnull().values.any())
    
    peri = pd.PeriodIndex(covid_periodos.keys())[pd.PeriodIndex(covid_periodos.keys()) > pd.Period('2019-12')]
    peri = peri[peri < pd.Period('2021-08')]
    
    covid = []
    for k in peri:
        covid_periodos[k]['df']['mes'] = k
        covid.append(covid_periodos[k]['df'])
    covid = pd.concat(covid)
    covid['mes'] = covid.mes.astype(str)
    covid = covid.set_index(['link','mes'])
    
    index = covid.index
    
    covid.to_file('datos/covid_periodos.shp', index = True)

def df_covid(dir_datos = dir_datos):
    if os.path.exists(dir_datos+'/covid_periodos.shp') == False:
        __genera_shp()
    
    covid = gpd.read_file(dir_datos+'/covid_periodos.shp', index = True)
    covid = covid.set_index(['link','mes']).sort_index(level = 0)
    covid = covid.loc[pd.IndexSlice[:,'2020-03':],:]
    covid = covid.to_crs(epsg = EPSG)
    
    # Separamos los campos geometricos del dataframe
    geo = covid.loc[pd.IndexSlice[:,'2021-01'],'geometry']
    geo = geo.reset_index(level = 'mes', drop = True)
    centroides = covid.loc[pd.IndexSlice[:,'2021-01'],'geometry'].centroid#to_crs('EPSG:4694').centroid
    centroides = centroides.reset_index(level = 'mes', drop = True)

    codiprov = covid.loc[pd.IndexSlice[:,'2021-01'],['codpcia','departamen','provincia']]
    
    columnas = ['clasificac', 'fallecido']

    # Variables acumuladas a partir del mes que todas tienen al menos 1 

    covid_acum = covid[columnas].groupby(covid.index.get_level_values(0)).cumsum()
    # buscamos el mes en que todos los dptos tienen al menos 1 contagio
    mes = 0
    valor = True
    while valor == True:
        Mes = covid.index.get_level_values(1).unique()[mes]
        valor = np.any(covid_acum.loc[pd.IndexSlice[:,Mes],'clasificac'] == 0)
        mes +=1
    print("El mes desde el cuál todos los dptos tienen al menos 1 contagiado es: "+str(Mes))
    covid_acum['personas'] = covid.personas
    personas = covid.loc[pd.IndexSlice[:,'2021-07'],'personas']
    
    return covid,geo, centroides, codiprov,covid_acum,personas


if __name__ == "__main__":
    covid,geo, centroides, codiprov,covid_acum,personas = df_covid()
    
    