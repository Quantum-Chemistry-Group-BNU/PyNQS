import numpy as np
import scipy
from opt_einsum import contract as einsum

def shapes(sites):
    shape = []
    for i in range(len(sites)):
        shape.append(sites[i].shape)
    return shape

def overlap(sites1,sites2):
    assert len(sites1) == len(sites2)
    env = np.ones((1,1),dtype=sites1[0].dtype)
    for i in range(len(sites1)-1,-1,-1):
        tmp = einsum('lnr,rk->lnk',sites2[i],env)
        env = einsum('lnr,mnr->lm',sites1[i].conj(),tmp)
    return env

def get_SvN(pop):
    s = 0.0
    for p in pop:
        if p < 1.e-16: continue
        s += -p*np.log(p)
    return s

def singleSiteEntropy(sites,iroot=0):
    nsite = len(sites)
    sp = np.zeros(nsite)
    nroots = sites[0].shape[0]
    env = np.zeros((nroots,nroots),dtype=sites[0].dtype)
    env[iroot,iroot] = 1.0
    for i in range(nsite):
        tmp = einsum('lr,rnk->lnk',env,sites[i])
        env = einsum('knl,knr->lr',sites[i],tmp)
        # compute rho
        rho = einsum('lmr,lnr->mn',sites[i],tmp)
        sp[i] = get_SvN(np.diag(rho))
    return sp

def twoSiteEntropy(sites,iroot=0):
    nsite = len(sites)
    spq = np.zeros((nsite,nsite))
    nroots = sites[0].shape[0]
    env = np.zeros((nroots,nroots),dtype=sites[0].dtype)
    env[iroot,iroot] = 1.0
    for i in range(nsite):
        tmp = einsum('lr,rnd->lnd',env,sites[i])
        env = einsum('lnu,lnd->ud',sites[i],tmp)
        # compute uncontracted rho
        rho1 = einsum('lmu,lnd->mnud',sites[i],tmp)
        for j in range(i+1,nsite):
            tmp = einsum('mnud,dor->mnour',rho1,sites[j])
            rho1 = einsum('uol,mnour->mnlr',sites[j],tmp)
            # compute rho2
            rho2 = einsum('upr,mnour->mpno',sites[j],tmp)
            rho2 = rho2.reshape(16,16)
            # Note that the physical indices in CTNS are [0,2,a,b]
            #print(i,j,np.linalg.norm(rho2-rho2.T))
            e,v = scipy.linalg.eigh(rho2)
            spq[i,j] = get_SvN(e)
            spq[j,i] = spq[i,j]
    return spq

def mutualInformation(spq,sp):
    norb = sp.shape[0]
    Ipq = np.zeros_like(spq)
    for i in range(norb):
        for j in range(i):
            Ipq[i,j] = sp[i] + sp[j] - spq[i,j]
            Ipq[j,i] = Ipq[i,j]
    return Ipq
