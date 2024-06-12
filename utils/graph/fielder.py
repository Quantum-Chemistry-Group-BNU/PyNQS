#
# Fiedler
#
import numpy
import scipy.linalg

#
# It seems necessary to exclude -1.0, otherwise dii=0.0
# and sometime the first eigenvector is not zero any more
# due to a large negative eigenvalue ->
#
# (POSITIVE definiteness of the laplacian matrix???)
#
# Or it does not matter due to the constant shift???
#

#
# dij = Jij/Sqrt[Jii*Jjj] - 1.0
#
def distanceMatrix(eri):
   nb = eri.shape[0]
   dij = numpy.zeros((nb,nb))
   for i in range(nb):
      for j in range(nb):
        dij[i,j] = eri[i,i,j,j]/numpy.sqrt(eri[i,i,i,i]*eri[j,j,j,j]) 
   return dij

# Kij = (ij|ij)
def exchangeMatrix(eri):
   nb = eri.shape[0]
   kij = numpy.zeros((nb,nb))
   for i in range(nb):
      for j in range(nb):
         kij[i,j] = eri[i,j,j,i] 
   return kij

# L = D - K
def laplacian(dij):
   nb  = dij.shape[0]
   lap = numpy.zeros((nb,nb))
   lap = -dij
   # See DMRG in practive 2015 Dii = sum_j Kij
   diag = numpy.einsum('ij->i',dij)
   lap += numpy.diag(diag)
   return lap

# Get the orbital ordering
def orbitalOrdering(eri,mode='kmat',debug=False):
   if debug: print('\n[fielder.orbitalOrdering] determing ordering based on',mode.lower())
   nb  = eri.shape[0]
   if mode.lower() == 'dij':
      dij = distanceMatrix(eri)
   elif mode.lower() == 'kij':
      dij = exchangeMatrix(eri)	  
   elif mode.lower() == 'kmat':
      dij = eri.copy() 
   lap = laplacian(dij)
   eig,v = scipy.linalg.eigh(lap)
   # From postive to negative
   order=numpy.argsort(v[:,1])[::-1]
   order2=numpy.argsort(v[:,2])[::-1]
   if debug: 
      print('dij:\n',dij)
      print('eig:\n',eig)
      print('v[1]=',v[:,1])
      print('new order1:',order)
      print('new order2:',order2)
   return order


if __name__ == '__main__':
   #
   # i/j 0   1   2   3
   #
   # 3  *---*---*---*
   #    |12 |13 |14 |15
   # 2  *---*---*---*
   #    |8  |9  |10 |11
   # 1  *---*---*---*
   #    |4  |5  |6  |7
   # 0  *---*---*---*
   #     0   1   2   3
   #
   # no PBC (degenerate):
   # new order: [12  8 13  9  4 14  0  5 10 15  1 11  6  2  7  3]
   # new order: [ 0  1  4  5  2  8  3  6  9 12  7 13 10 11 14 15]
   # PBC:
   # new order: [ 8  4 11  9 12  7  5 10  0 15 13  6  3  1 14  2]
   #
   def t2d(n,t,ifpbc=True):
      nsite = n*n
      tmatrix = numpy.zeros((nsite*2,nsite*2))
      tmat = numpy.zeros((nsite,nsite))
      if ifpbc:
         for i in range(n):
            tmat[i*n,(i+1)*n-1] = -t
            tmat[(i+1)*n-1,i*n] = -t
         for i in range(n):
            tmat[i,(n-1)*n+i] = -t
            tmat[(n-1)*n+i,i] = -t
      # Row:
      for i in range(n):
         for j in range(n-1):
            ijC = i*n+j
            ijR = i*n+j+1
            tmat[ijC,ijR] = -t
            tmat[ijR,ijC] = -t
      # Up:
      for i in range(n-1):
         for j in range(n):
            ijC = i*n+j
            ijU = (i+1)*n+j
            tmat[ijC,ijU] = -t
            tmat[ijU,ijC] = -t
      # Save 
      tmatrix[0::2,0::2] = tmat
      tmatrix[1::2,1::2] = tmat
      return nsite,tmatrix

   nsite,kmat = t2d(4,-1.0,ifpbc=False)
   print('nsite=',nsite)
   kmat = kmat[0::2,0::2]
   orbitalOrdering(kmat,mode='kmat',debug=True)
