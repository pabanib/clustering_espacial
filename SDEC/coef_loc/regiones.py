# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:25:21 2022

@author: Pablo
"""
#%%

import sys
sys.path.append("./SDEC/coef_loc/")
from . import lq_funciones as lq

class area:
 
    def __init__(self, variable, poblacion):
        # variable puede ser una lista de más de una variable
        # Igual población
        try:
            variable = variable.tolist()
        except:
            pass

        self.variable = []
        if isinstance(variable, list):
            self.variable = variable
        else:
            self.variable.append(variable) 
        
        try:
            poblacion = poblacion.tolist()
        except:
            pass

        self.poblacion = []
        if isinstance(poblacion, list):
            self.poblacion = poblacion
        else:
            self.poblacion.append(poblacion) 
        
        if len(self.poblacion) == 1:
            self.poblacion = self.poblacion*len(self.variable)
        elif len(self.variable) == len(self.poblacion):
            pass
        else:
            raise 
    
    @property
    def dict_(self):
        dic = {}
        for i in range(len(self.variable)):
            dic[i] = {'var': self.variable[i], 'pob': self.poblacion[i]}
            
        return dic
    
    @property
    def shape(self):
        return len(self.variable)
        
 
    
class region(area):
    
       
    def __init__(self, variable, poblacion):
        areas = []
        var = lq.np.array(variable)
        pob = lq.np.array(poblacion)
        for i in range(len(variable)):
            a = area(variable[i],poblacion[i])
            areas.append(a)
        self.areas = areas
        
        area.__init__(self, var.sum(axis = 0).tolist(), pob.sum(axis = 0).tolist())
        self.add_varianza()

    @property
    def shape(self):
        return len(self.areas),len(self.variable)        
    
    @property
    def variables(self):
        v = {}
        for i in range(self.shape[1]):
            v[i] = []
            for j in self.areas:
                v[i].append(j.dict_[i]['var'])
        return v                
    
    @property
    def poblaciones(self):
        v = {}
        for i in range(self.shape[1]):
            v[i] = []
            for j in self.areas:
                v[i].append(j.dict_[i]['pob'])
        return v 
    
    def calc_lq(self):
        lqs = []
        for i in range(self.shape[1]):
            lqs.append(lq.clq(self.variables[i], self.poblaciones[i]))
        
        return lq.np.array(lqs).T
        
    def calc_lambda(self):
        lqs = []
        for i in range(self.shape[1]):
            lqs.append(lq.clambda(self.variables[i], self.poblaciones[i]))
        
        return lq.np.array(lqs).T        
        
    def add_varianza(self, vari = lq.varianza_m):
         self.vari = vari
    
    def calc_homogeneidad(self):
        lam = self.calc_lambda()
        v = []
        for i in range(self.shape[1]):
            x = (self.variables[i]-lam[:,i])**2
            v.append(x)
            
        return lq.np.array(v).T.mean(axis = 0)
        
    def calc_varianza(self):
        
        v = []
        for i in range(self.shape[1]):
            v.append(self.vari(lq.np.array(self.variables[i]), lq.np.array(self.poblaciones[i])))
        
        return lq.np.array(v).T   

    def intervalos(self):
        int = []
        for i in range(self.shape[1]):
            x, y = lq.np.array(self.variables[i]), lq.np.array(self.poblaciones[i])
            interv = lq.intervalos(x,y, a = .95)
            int.append(interv)
        return int

    def calc_ind_lq(self):
        v = []
        for i in range(self.shape[1]):
            x, y = lq.np.array(self.variables[i]), lq.np.array(self.poblaciones[i])
            interv = lq.intervalos(x,y, a = .95)
            ind = lq.indice(interv)
            v.append(ind)
        return lq.np.array(v).T

    def values(self):
        v = list(self.variables.values())
        p = list(self.poblaciones.values())
        return lq.np.array(v).T, lq.np.array(p).T
    def convertir_a_territ(self, agrup):
        v,p = self.values()
        t = territorio(v,p,agrup)
        return t
    
class territorio(region):
    
    def __init__(self, variable, poblacion, agrup):
        var = lq.np.array(variable)
        pob = lq.np.array(poblacion)
        agrup = lq.np.array(agrup)
        g = set(agrup)
        regiones = []
        V = []
        P = []
        for i in g:
            v = var[agrup == i]
            p = pob[agrup == i]
            r = region(v.tolist(),p.tolist())
            V.append(r.variable)
            P.append(r.poblacion)
            regiones.append(r)
        self.regiones = regiones
        
        region.__init__(self, V,P)
    
    def __name__(self):
        return 'Territorio'
    
    @property
    def variables(self):
        v = super().variables
        
        return v
    
    @property
    def poblaciones(self):
        v = super().poblaciones
        return v 
    def variables_areas(self):
        "Devuelve los datos de cada una de las áreas agrupados por región"
        variables_areas = {}
        poblaciones_areas = {}
        for k in self.variables.keys():
            V = []
            P = []
            for r in self.regiones:
                v = r.variables[k]
                p = r.poblaciones[k]
                V.append(v)
                P.append(p)
            variables_areas[k] = V
            poblaciones_areas[k] = P
        return variables_areas,poblaciones_areas
        
    def calc_lq(self):
        return super().calc_lq()
        
    def calc_lambda(self):

        return super().calc_lambda()
        
    def add_varianza(self, vari = lq.varianza_m):
         self.vari = vari
    
    def calc_homogeneidad(self):
        " Calcula la homogeneidad como si el territorio fuera una región"
        return super().calc_homogeneidad()
        
    def calc_varianza(self, varianza = 'm',*args,**kwargs):
        "Calcula la varianza para las regiones que pertenecen al territorio"
        if varianza.startswith('m'):
            vari = super().calc_varianza()
        else:
            V, P = self.variables_areas()
            vari = []
            for k in V.keys():
                vari.append(lq.varianza_b(V,P,*args,**kwargs))
        return vari

    def intervalos(self, varianza = 'm'):
        if varianza.startswith('b'):
            V, P = self.variables_areas()
            int = []
            for k in V.keys():
                interv = lq.intervalos(V[k], P[k], varianza = varianza)
                int.append(interv)
        else:
            int = super().intervalos()
        return int

    def calc_ind_lq(self,varianza = 'm', prom = True):

        if varianza.startswith('b'):
            V, P = self.variables_areas()
            ilq = []
            for k in V.keys():
                interv = lq.intervalos(V[k], P[k], varianza = varianza)
                ilq_ = lq.indice(interv)
                ilq.append(ilq_)
            ilq = lq.np.array(ilq)
        else:
            ilq = super().calc_ind_lq()
        if prom:
            ilq = ilq.mean()
        return ilq
        
    def calc_homog_regiones(self):
        hgr = []
        for ii in range(self.shape[0]):
            hgr.append(self.regiones[ii].calc_homogeneidad())
            
        return lq.np.array(hgr)
            
    def calc_homog_particion(self):
        return(lq.np.mean(self.calc_homog_regiones(), axis = 0) )

class multi_territorio(region):
    def __init__(self, variable, poblacion, agrup):
        self.reg = region(variable,poblacion)
        self.territorios = {}
        self.territorios[0] = [self.reg.convertir_a_territ(agrup[0])]
        for i in range(1,len(agrup)):
            l = []
            for j in self.territorios[i-1][0].regiones:
                l.append(j.convertir_a_territ(agrup[i]))
            self.territorios[i] = l


def df_to_region(df,poblacion,grupos =[], variables = []):
    if 'geometry' in df.columns:
        df = df.drop('geometry', axis = 1)
    if variables == []:
        variables = list(df.columns)
        variables.remove(poblacion)
    if isinstance(grupos, lq.np.ndarray) or isinstance(grupos, list): 
        rdo = territorio(df[variables].values, df[poblacion].values,grupos)
    else:
        if grupos == []:
            rdo = region(df[variables].values, df[poblacion].values)
        elif grupos in df.columns:
            try:
                variables.remove(grupos)
            except ValueError:
                pass
            rdo = territorio(df[variables].values, df[poblacion].values, df[grupos].values)
        else:
            raise "La forma de agrupar es incorrecta"
        
    return rdo

def paneldf_to_region(paneldf,poblacion, grupos = [], variables = []):
    if 'geometry' in paneldf.columns:
        df = paneldf.drop('geometry', axis = 1)
    else:
        df = paneldf 
    if variables == []:
        variables = list(df.columns)
        variables.remove(poblacion)    
    pob = paneldf[[poblacion]].unstack().values
    reg = []
    for c in variables:
        d = df[[c]]
        d = d.unstack().values
        if grupos == []:
            r = region(d,pob)
        else:
            r = territorio(d,pob,grupos)
        reg.append(r)
    return reg        

#%%

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({'var1': np.random.randint(0,10,20),
                        'var2': np.random.randint(0,10,20),
                        'pob': np.random.randint(10,20,20)})

    r = df_to_region(df, 'pob')

    t = df_to_region(df, 'pob', np.random.randint(0,2,20))
    t2 = df_to_region(df, 'pob', np.random.randint(0,2,20).tolist())

    df2 = df.copy()
    df2['agrup'] = np.random.randint(0,2,20)

    t3 = df_to_region(df2, 'pob', 'agrup')

    t4 = df_to_region(df2, 'pob','agrup',['var1'])
    print(t4.calc_ind_lq(varianza = 'b'))

#%%

""" a = area([5,6,8],[9])
print(a)

lq.np.array([a.variable, a.variable]).sum(axis = 0)
lq.np.array(a.poblacion)

a.dict_.values()
a.shape
a.variable[0]

r = region([[5,6],[3,4]],[[9,8],[7,6]])

r.areas

r.variable
r.poblacion

r.shape

r.dict_

lq.np.array([5,6])+lq.np.array([3,4])


r.variables
r.poblaciones

r.calc_lq()
r.add_varianza()
r.calc_varianza()
r.vari

lq.varianza(lq.np.array(r.variables[0]),lq.np.array(r.poblaciones[0]))

x = lq.np.array(r.variables[0])
y = lq.np.array(r.poblaciones[0])

lq.varianza(x,y)
r.calc_homogeneidad()
r.calc_ind_lq()

d = [[5,6],[3,4],[3,1]],[[9,8],[7,6],[6,5]]
t = territorio(d[0],d[1], [1,1,2])

t.regiones
t.variables
t.poblaciones
t.calc_homogeneidad()
t.calc_ind_lq()
t.shape
t.calc_homog_regiones() """


# %%
