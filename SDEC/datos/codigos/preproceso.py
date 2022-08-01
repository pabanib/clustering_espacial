# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 22:06:48 2021

@author: paban
"""

import pandas as pd
import datetime

#covid = pd.read_csv('D:/MisDocumentos/Documentos/Bases de datos/Covid19Casos/Covid19Casos.csv')

#df = covid.query("residencia_departamento_nombre == 'Lavalle' and residencia_provincia_nombre == 'Mendoza'")

campos_fecha = ['fecha_inicio_sintomas','fecha_apertura','fecha_internacion','fecha_cui_intensivo','fecha_fallecimiento','fecha_diagnostico']

def transf_fechas(df, campos_fecha):    
    df[campos_fecha] = df[campos_fecha].fillna('1990-01-01')
    
    for j in campos_fecha:
        try:
            df[j] = df[j].apply(lambda i: datetime.datetime.fromisoformat(i))
        except:
            pass
    return df

#transf_fechas(covid, campos_fecha)
#covid.dtypes

campos_bool = ['cuidado_intensivo','fallecido','asistencia_respiratoria_mecanica']
#df[campos_bool] == 'SI'
#(df[campos_bool] == 'SI').sum()

def transf_bool(df, campos_bol, palabra):
    #df es el dataframe a transformar
    #campos_bol es un listado de columnas con los campos booleanos
    #palabra es el indicador que quiero que sea verdadero
    df[campos_bool] = df[campos_bool] == palabra
    
    return df

#transf_bool(df, campos_bool, 'SI')
#df.dtypes

campos_categ = ['sexo','origen_financiamiento','clasificacion_resumen']

from numpy import inf

def frac_edad(df):
    df['edad'] =  df.edad*(df.edad_años_meses == 'Años') #convierte a los menores de 1 año en 0 años
    clas_edad = pd.cut(df.edad, [0,18,40,60,inf], labels = ['menores','adulto_joven', 'adulto_medio','adulto_mayor']) 
    
    return pd.get_dummies(clas_edad)

#ed = frac_edad(df)
#ed.columns

def indice_geog(df):
    def completar(n):
        return str(n).zfill(3)

    df['residencia_provincia_id'] = df['residencia_provincia_id'].astype(str)
    df['residencia_departamento_id'] = df['residencia_departamento_id'].apply(completar) 
    df['residencia_dpto'] = (df.residencia_provincia_id+df.residencia_departamento_id).astype(int)

    return df

class preparar_datos():
    
    def __init__(self, df):
        self.df = df
        self.atributos = {'campos_fecha': campos_fecha, 'campos_bool': campos_bool, 'campos_categ': campos_categ}
         
        
    def fit(self):
        d = self.df.copy()
        
        d = transf_fechas(d, self.atributos['campos_fecha'])
        d = transf_bool(df= d, campos_bol = self.atributos['campos_bool'] ,palabra = 'SI')
        self.dummies = pd.get_dummies(d[self.atributos['campos_categ']] )
        d[self.dummies.columns] = self.dummies
        self.edad = frac_edad(d)
        d[self.edad.columns] = self.edad
        d = indice_geog(d)
        
        return d 
    
#datos = preparar_datos(covid)
#covid_ = datos.fit()        
#covid_.dtypes        
#d = covid_.groupby(['residencia_dpto',covid_.fecha_apertura.dt.year]).sum()


