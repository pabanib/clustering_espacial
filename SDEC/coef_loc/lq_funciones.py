# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:17:23 2022

@author: Pablo
"""
#%%
import numpy as np
import scipy.stats as st

def cprop(var, pobl):
    # Calcula la proporción de una variable sobre otra poblacional por unidad 
    # var es un array o una lista
    # pobl es un array o una lista
    var, pobl = np.array(var), np.array(pobl)
    assert np.all(pobl > 0) , 'Valores 0 o negativos en la población'
    assert np.all(var >= 0) , 'Valores negativos en la variable'
    
    return var/pobl

def clq(var, pobl):
    # Calcula el coef. de localización de una variable sobre una poblacional por unidad
    # var es un array o una lista
    # pobl es un array o una lista
    var, pobl = np.array(var), np.array(pobl)
    c = cprop(var,pobl)
    S = np.sum(var)
    assert all(S > 0) if isinstance(S,np.ndarray) else S>0, 'La variable es siempre 0'
    
    return c/(S/sum(pobl))


def clambda(var, pobl):
    # Calcula la realización de la variable de acuerdo a los valores de la prop en la pobl
    # var es un array o una lista
    # pobl es un array o una lista
    var, pobl = np.array(var), np.array(pobl)
    assert np.all(pobl > 0) , 'Valores 0 o negativos en la población'
    theta = pobl/sum(pobl)
    S = sum(var)
    return S*theta      
    
def varianza_m(var, pobl):
    # Calcula la varianza teórica brindad por el paper de moinedin
    # var es un array o una lista
    # pobl es un array o una lista
    pi = cprop(var,pobl)
    p = sum(var)/sum(pobl)
    ni = pobl
    n = sum(pobl)
    assert np.all(pi <= 1), 'Hay prop mayores que 1'    
    
    var = (pi*(1-pi)/(ni*p**2))+((pi**2)*(1-p)/(n*p**3))+(2*(pi**2)*(1-pi)/(n*p**3))
    return var

def bootstrap_reg(var, pobl):
    """
    Remuestrea las variables por región compuesta de áreas con su variable y su población
    hay que pasarle una lista de listas
    """
    k = len(var)
    vari = []
    pobl_ = []
    for i in range(k):
        rv, rp = np.array(var[i]), np.array(pobl[i])
        n = len(rv)
        c = np.random.choice(np.arange(n), size = n)
        vari.append(np.sum(rv[c]))
        pobl_.append(np.sum(rp[c]))
    return np.array(vari),np.array(pobl_)

def varianza_b(var, pobl, n = 1000):
    # Calcula la varianza haciendo bootstrap
    lqs = []
    for i in range(n):
        v,p = bootstrap_reg(var, pobl)
        lqs.append(clq(v,p))
    lqs = np.array(lqs)
    return np.var(lqs, axis = 0)

def intervalos(var, pobl, a = .95, varianza = 'm',*args,**kwargs):
    """Calcula intervalos de confianza para cada área La varianza por defecto es la que está en el paper de Moinedin,
     pero se puede cambiar por la varianza bootstrap
        en este caso hay que pasarle listas de listas como variables
     """
    if varianza.startswith("m"):
        Vari = varianza_m
        VAR = var
        POBL = pobl
    elif varianza.startswith("b"):
        Vari = varianza_b
        VAR = []
        POBL = []
        for i in range(len(var)):
            v = sum(var[i])
            p = sum(pobl[i])
            VAR.append(v)
            POBL.append(p)

    else:
        raise "Varianza desconocida"
    N = len(VAR)

    vari = Vari(var,pobl, *args, **kwargs)
    sd = np.sqrt(vari)
    sd
    indices = clq(VAR,POBL)
    sdnorm = st.t.ppf((1+a)/2,N-1)*sd
    #sdnorm = st.norm.ppf((1+a)/2)*sd
    
    return np.array([indices-sdnorm,indices+sdnorm]).T


def intersec(a,b):
    # Ante dos intérvalos contínuos y cerrados devuelve si existe intersección entre ellos
    
    
    if min(a) < max(b) and max(a)< min(b):
        return False
    elif min(b) < max(a) and max(b) < min(a):
        return False
    else:
        return True

def matrix_inters(x):
    #x debe ser un array de 2 dimensiones indicando los intervalos   
    l = len(x)
    dic = {}    
    for i in range(len(x)):
        v = []
        for j in range(len(x)):
            v.append(intersec(x[i],x[j]))
        dic[i]=v
    matrix = np.array(list(dic.values()))-np.eye(l)
    return matrix

def indice(x):
    return matrix_inters(x).sum()/(len(x)*(len(x)-1))

def matrix_inters_k(lista):
    # lista debe ser una lista de array de 2 dimensiones en donde cada array representa una variable con su intervalo
    variables = len(lista)
    l = len(lista[0])
    dic = {}
    for g in range(l):
        dic[g] = {}
        for i in range(variables):
            dic[g][i] = lista[i][g]
    
    x = {}
    for i in dic.keys():
        v = []
        for j in dic.keys():
            r = 1
            for k in dic[i].keys(): 
                if i == j:
                    r = r*1
                else:
                     r = r* intersec(dic[i][k],dic[j][k])
            v.append(r)
        x[i]=v
    matrix = np.array(list(x.values()))-np.eye(l)
    return matrix


def matrix_inters_k_debil(lista):
    # lista debe ser una lista de array de 2 dimensiones en donde cada array representa una variable con su intervalo
    variables = len(lista)
    l = len(lista[0])
    dic = {}
    for g in range(l):
        dic[g] = {}
        for i in range(variables):
            dic[g][i] = lista[i][g]
    
    x = {}
    for i in dic.keys():
        v = []
        for j in dic.keys():
            r = 0
            for k in dic[i].keys(): 
                if i == j:
                    r += 1
                else:
                     r += intersec(dic[i][k],dic[j][k])
            v.append(r/variables)
        x[i]=v
    matrix = np.array(list(x.values()))-np.eye(l)
    return matrix



#%%

# a = np.array([3,300,4000])
# b = np.array([100,4000,5000])
# clq(a,b)
# clambda(a,b)
# varianza(a,b)
# #intervalos(a/1000,b/1000)

# def var2(a,b):
#     V = []
#     for i in range(len(a)):
#         v = np.var(a/b)
#         V.append(v)
#     return np.array(V)

# inte = intervalos(a,b, varianza = var2)
# inte[0]
# matrix_inters(inte)
# indice(inte)
# indice(intervalos(a,b))

if __name__ == "__main__":

    v = np.random.randint(0,20, size = (4,3))
    p = v+np.random.randint(1,20, size = (4,3))

    varianza_b(v,p)
    varianza_m(v,p)
    clq(v,p)
    clq(v[:,0],p[:,0])
    
# %%
