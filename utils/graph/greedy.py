import numpy as np

# Find the path by greedy search
def greedyPath(kij,i,iwt=0):
    norb = kij.shape[0]
    ista = i
    wt = 0.0
    order = [ista]
    for i in range(norb-2):
        wvec = []
        ivec = []
        for k in range(norb):
            if k == ista: continue # no self edge
            if k in order: continue # edge go back is not allowed
            ivec.append(k)
            # Local weight
            if iwt == 0:
                wk = kij[ista,k]
            elif iwt == 1:
                wk = np.sum(kij[order,k])
            wvec.append(wk)
        ord = np.argsort(wvec)[-1::-1]
        ista = ivec[ord[0]]
        order.append(ista)
    # add the rest
    for i in range(norb):
        if i not in order:
            order.append(i)
            break        
    return order

def scoreFunction(kij,order):
    norb = kij.shape[0]
    wt = 0.0
    for i in range(norb-1):
        wt += kij[order[i],order[i+1]]
    return wt
    
def greedyOrdering(kij,iwt=0):
    assert iwt == 0 or iwt == 1
    norb = kij.shape[0]
    ordernew = None
    scoremax = -1
    idx = -1
    for i in range(norb):
        order = greedyPath(kij,i,iwt)
        score = scoreFunction(kij,order)
        #print(i,score,order)
        if score > scoremax:
            scoremax = score
            ordernew = order
            idx = i
    return idx,scoremax,ordernew