#%%
import SDEC.coef_loc.regiones as reg
# %%

import pandas as pd
import numpy as np
df = pd.DataFrame({'var1': np.random.randint(0,10,20),
                        'var2': np.random.randint(0,10,20),
                        'pob': np.random.randint(10,20,20)})

r = reg.df_to_region(df, 'pob')

t = reg.df_to_region(df, 'pob', np.random.randint(0,2,20))
t2 = reg.df_to_region(df, 'pob', np.random.randint(0,2,20).tolist())

df2 = df.copy()
df2['agrup'] = np.random.randint(0,2,20)

t3 = reg.df_to_region(df2, 'pob', 'agrup')

t4 = reg.df_to_region(df2, 'pob','agrup',['var1'])
t4.calc_ind_lq(varianza = 'b')
# %%

import SDEC.coef_loc.lq_funciones as lqf

# %%
"""
cada lista interna, en el ejemplo [2,5] es una región, con su variable y su 
correspondiente población.

"""

var = [[2,5],[6,7]]
pobl = [[10,8],[10,8]]

v, p = lqf.bootstrap_reg(var, pobl)
"la función bootstrap_reg, intercambia los valores por cada región y devuelve la suma"

# %%
lqf.varianza_b(var,pobl,2)

# %%
lqf.intervalos(var,pobl,varianza= 'b')
# %%
t4.variables

# %%
