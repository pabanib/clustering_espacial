#%%
from salarios.procesamiento_sal import Salarios
import SDEC.regionalizacion.deep_cluster as dc
import numpy as np
import pickle
import yaml 
import warnings
warnings.filterwarnings("ignore")

#%%

with open('salarios/casos.yaml', 'r') as file:
    casos = yaml.safe_load(file)

parametros = {
    'variables': ['masa_cer', 'puestos'],
    'poblacion': 'personas',
    'n_clusters': [30],
    'semilla': 42,
    'knn': 4,

}



def entrenar(parametros, nombre ):
    """
    Entrena el modelo SDEC con los parámetros especificados.
    
    :param parametros: Diccionario con los parámetros de entrenamiento.
    :return: Resultados del entrenamiento.
    """
    region = 'cuyo_A_C'
    # Aquí podrías usar region, variables, poblacion, n_clusters
    print(f"Entrenando para la región {region} con variables {parametros['variables']} y {parametros['n_clusters']} clusters.")

    np.random.seed(parametros['semilla'])

    proc = Salarios()
    prop_st = proc.preparar_datos(parametros['variables'], parametros['poblacion'])
    cant_peris = 108
    W = proc.generar_matriz(parametros['knn'])

    resultados = {}
    for n_clusters in parametros['n_clusters']:
        X1 = prop_st[:, :cant_peris]
        X2 = prop_st[:, cant_peris:]
        sc = dc.sdec(n_clusters, [cant_peris, cant_peris])
        sc.gen_modelo()
        sc.train_autoencoder([X1, X2], prop_st)
        sc.ajustar_modelo([X1, X2], W.sparse)
        resultados[n_clusters] = sc

    casos = {'parametros': parametros,
            'resultados': resultados,}

    with open(r'D:\Archivos\Codigos\clustering espacial\salarios\resultados/resultados_{}.pkl'.format(nombre), 'wb') as f:
        pickle.dump(casos, f)
        print("Resultados guardados en 'resultados/resultados_sdec.pkl'")
# %%
parametros = {}
for caso in casos['casos']:
    if caso['procesar']:
        nombre = caso['nombre']
        parametros['variables'] = caso['variables']
        parametros['poblacion'] = caso['poblacion']
        parametros['n_clusters'] = range(caso['n_clusters_d'], caso['n_clusters_h'])
        parametros['knn'] = caso['matriz']
        parametros['semilla'] = caso.get('semilla', 42) 
        entrenar(parametros, nombre) # Valor por defecto si no se especifica
        break