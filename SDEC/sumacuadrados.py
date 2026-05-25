#%%
import numpy as np
from scipy.spatial.distance import cosine,correlation

#%%
def dist(x,y):
    x = np.array(x)
    y = np.array(y)
    return np.sum((x-y)**2)

def wss(X):
    X = np.array(X)
    C = np.median(X,axis = 0)
    filas = X.shape[0]
    rdo = []
    for i in range(filas-1):
        x = X[i,:]
        #r = []
        r = dist(x,C)
        #for j in range(1,filas):
        #    y = X[j,:]
        #    r.append(dist(x,y))
        rdo.append(r)
    return np.sum(rdo)

def SSD(X, grupos):
    X = np.array(X)
    grupos = np.array(grupos)
    g = np.unique(grupos)
    WSS = []
    for i in g:
        x = X[grupos == i]
        WSS.append(wss(x))
    TSS = wss(X)
    BSS = TSS-np.sum(WSS)
    RBTSS = BSS/TSS
    return {'TSS': TSS, 'WSS': np.sum(WSS)/len(WSS), 'BSS': BSS, 'RBTSS':RBTSS}

#%%
if __name__ == "__main__":
    x = [2,5]
    y = [1,9]
    print(dist(x,y))

    X = [[2,5],[3,1],[2,2]]
    print(wss(X))

    print(SSD(X, [1,1,2]))

    g1 = np.random.randn(10,2)+2
    g2 = np.random.randn(10,2)+4
    g3 = np.random.randn(10,2)+6

    G = np.array([g1,g2,g3]).reshape(-1,2)

    SSD(G, np.random.randint(0,2,30))

    r = np.array([np.ones(10),np.ones(10)+1,np.ones(10)+2]).reshape(-1,)
    SSD(G,r)

# %%

x = np.arange(100)

y1 = 2*x+np.random.randn()
y2 = 100 + 2*x+np.random.randn()
y3 = -(x**2)+np.random.randn()

# %%
print(np.corrcoef(y1,y2),
dist(y1,y2),
dist(y1-np.mean(y1),y2-np.mean(y2)),
cosine(y1,y2))

# %%
dist(y2,y3)

# %%
dist(y1-np.mean(y1),y3-np.mean(y3))

# %%

cosine(y1,y3)

# %%
cosine(y2,y3)
# %%
