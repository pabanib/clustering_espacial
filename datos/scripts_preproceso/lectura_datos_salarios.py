import os
import sys
import pandas as pd
import geopandas as gpd

project_root = os.path.abspath(os.path.join(os.getcwd(), '...'))
if project_root not in sys.path:
    sys.path.append(project_root)
import pyproj
#from datos.constants import FILE_DATADIR_PYPROJ, DIR_PRINCIPAL, EPSG
from constantes import FILE_DATADIR_PYPROJ, DIR_PRINCIPAL, EPSG, DIR_DATOS
pyproj.datadir.set_data_dir(FILE_DATADIR_PYPROJ)


# %%
DIR_DATOS_ORIG = os.path.join(DIR_DATOS,"originales")
DIR_DATOS_PROC = os.path.join(DIR_DATOS,"procesados/salarios")
#dir_datos = DIR_PRINCIPAL+'\\datos\\salarios\\'

def __genera_sal_old():
    sal = pd.read_csv(DIR_DATOS_ORIG+'/sal.csv')
    sal.drop(sal.columns[0], axis = 1, inplace=True)
    sal = sal.set_index([sal.columns[0],sal.columns[1],sal.columns[2]])
    clae = sal.groupby(level = [1,2]).sum()
    clae['prom_clae'] = clae['masa_cer']/clae['puestos']
    clae = clae.groupby(level = 0).mean()

    clae_bajo = clae.sort_values('prom_clae').head(25)
    clae_alto = clae.sort_values('prom_clae').tail(25)
    l = []
    for i in sal.reset_index().clae2:
        if i in list(clae_bajo.index):
            l.append(1)
        else:
            l.append(0)


    sal['clae_bajo'] = np.array(l) * sal['puestos']
    
    l = []
    for i in sal.reset_index().clae2:
        if i in list(clae_alto.index):
            l.append(1)
        else:
            l.append(0)


    sal['clae_alto'] = np.array(l) * sal['puestos']

    sal = sal.groupby(level = [0,2]).sum()
    #sal = sal.groupby(level = 0).rolling(12).mean()
    #sal = sal.dropna()
    sal = sal.reset_index()
    return sal

def __genera_sal():
    sal = pd.read_csv(DIR_DATOS_ORIG+'/sal.csv')
    sal.drop(sal.columns[0], axis = 1, inplace=True)
    sal = sal.set_index([sal.columns[0],sal.columns[1],sal.columns[2]])
    claes = pd.read_csv(os.path.join(DIR_DATOS_ORIG,"clae_agg_corr.csv"), sep = ';')
    claes = claes.groupby('clae2').first().letra 
    df2 = pd.merge(sal, claes, left_on='clae2', right_index= True)
    activ = df2.groupby(['codigo_departamento_indec','fecha','letra']).sum().puestos.unstack()
    df2 = df2.groupby(['codigo_departamento_indec','fecha']).sum()[['puestos','masa','masa_cer']]
    sal = pd.merge(df2, activ, left_index = True, right_index=True)
    return sal.reset_index()


# %%

def __genera_shp():
    sal = __genera_sal()
    sal.fecha = pd.to_datetime(sal.fecha)
    sal.masa_cer = sal.masa_cer/1000
    # Tranforma los datos para poder trabajarlos
    
    # Agrega condiciones espaciales a los datos
    Arg = gpd.read_file(DIR_DATOS_ORIG+'/shape_files/pxdptodatosok.shp')
    
    Arg = Arg.drop(Arg.query("link in('94028','94021')").index, axis = 0) 
    Arg['link'] = Arg['link'].astype(int)
    Arg['mujeres'] = Arg['mujeres'].astype(int) 
    Arg['varones'] = Arg['varones'].astype(int)
    Arg['personas'] = Arg['personas'].astype(int)
    Arg['hogares'] = Arg['hogares'].astype(int)
    Arg['viv_part'] = Arg['viv_part'].astype(int)
    Arg['viv_part_h'] = Arg['viv_part_h'].astype(int)
    
    claves = ['codigo_departamento_indec','fecha']
    #atributos = ['puestos','clae_bajo','clae_alto', 'w_mean']
    atributos = ['puestos', 'masa_cer', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Z']
    
    #periodos = pd.PeriodIndex(covid_geo.fecha_apertura, freq = 'm')
    agrupa = sal.groupby(claves)
    sal_geo = agrupa[atributos].sum()
    
    index = sal_geo.index
    idx = pd.IndexSlice
    
    df = pd.merge(sal_geo, Arg,how= 'inner' ,left_on= 'codigo_departamento_indec', right_on = 'link')
    df = df.fillna(0)
    sal_geo = gpd.GeoDataFrame(df.reset_index())

    sal_geo['mes'] = sal.fecha.dt.strftime('%Y-%m')
    sal = sal_geo.set_index(['link','mes'])
           
    sal.to_file(DIR_DATOS_PROC+'/sal_periodos.shp', index = True)

# %%
def df_salarios(dir_datos = DIR_DATOS_PROC):
    if os.path.exists(dir_datos+'/sal_periodos.shp') == False:
        __genera_shp()
    
    sal = gpd.read_file(dir_datos+'/sal_periodos.shp', index = True)
    sal = sal.set_index(['link','mes']).sort_index(level = 0)
    #sal = sal.loc[pd.IndexSlice[:,'2020-03':],:]
    sal = sal.to_crs(epsg = EPSG)
    
    # Separamos los campos geometricos del dataframe
    geo = sal.loc[pd.IndexSlice[:,'2021-01'],'geometry']
    geo = geo.reset_index(level = 'mes', drop = True)
    centroides = sal.loc[pd.IndexSlice[:,'2021-01'],'geometry'].centroid#to_crs('EPSG:4694').centroid
    centroides = centroides.reset_index(level = 'mes', drop = True)

    codiprov = sal.loc[pd.IndexSlice[:,'2021-01'],['codpcia','departamen','provincia']]
    
    columnas = ['puestos','clae_bajo','clae_alto', 'w_mean']
    personas = sal.loc[pd.IndexSlice[:,'2021-07'],'personas']
   
    return sal,geo, centroides, codiprov,personas