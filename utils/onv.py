import numpy as np

# FOCUS convention : {|0>,|up,dw>,|up>,|dw>}
def occ2idx(occA,occB):
    dic = {(False,False):0,
           (True,True):1,
           (True,False):2,
           (False,True):3}
    return dic[(occA,occB)]

def idx2occ(idx):
    dic = {0:(False,False),
           1:(True,True),
           2:(True,False),
           3:(False,True)}
    return dic[idx]


# Occupation number vector 
class ONV():
    def __init__(self,size=None,onv=None):
        if isinstance(onv,np.ndarray):
            self.vec = onv.copy()
        else:
            assert(size != None and size%2 == 0)
            self.vec = np.zeros(size, np.bool_)

    def __getitem__(self,i):
        return self.vec[i]
    def __setitem__(self,i,val):
        self.vec[i] = val

    # for spatial orbital    
    def setOcc(self,iorb,idx):
        (self.vec[2*iorb],self.vec[2*iorb+1]) = idx2occ(idx)
    def getIdx(self,iorb):
        return occ2idx(self.vec[2*iorb],self.vec[2*iorb+1])

    # phase to convert |ON>=f[onv]|IaIb>
    def phase(self):
        p = 0
        for i in range(2,self.vec.shape[0],2):
            if self.vec[i]: p += np.sum(self.vec[1:i:2])%2
            #print(i,[j for j in range(1,i,2)])
        return -2*(p%2)+1

if __name__ == '__main__':

    s = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    state = ONV(onv=s)
    state.phase()
    print(state.phase())
    # ababab
    # state = ONV(12)
    # state[0] = 1
    # state[1] = 1
    # state[2] = 1
    # state[3] = 1
    # state[4] = 1
    # state[5] = 1
    # state[6] = 0
    # state[7] = 0
    # state[8] = 0
    # state[9] = 0
    # state[10] = 0
    # state[11] = 0
    # print(state[3])
    # print(state.vec)
    # print(state.phase())

