import numpy as np
k_to_mev=8.6173e-2
thz_to_mev=4.1357 /2/np.pi
ha_to_ry=2
ry_to_mev=13605.662285137 /2/np.pi
ry_to_ev=ry_to_mev/1000
from scipy import interpolate
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

from mpi4py import MPI

# set up MPI world





mu_allendynes=0.1 #9 #0.2 #16

 
class eliashberg_solver():
 def __init__(self):
   self.allT=[]
   self.A2F=[]
   self.allT = list(np.arange(1,4, .5)) + list(np.arange(4,5, 0.2))
#   for i in range(150):
#     self.allT.append(1.+i/20.)
 

 def calc_bandwidth(self,data):
  bandwidth=np.mean([ (np.max(j)-np.min(j))/2 for j in data.ENE]) #*ha_to_ry
  print(bandwidth)
  return bandwidth

 def gap_parallel(self,Z1,D1,OM,K,beta,nshift,OMP):
  comm = MPI.COMM_WORLD
  size = comm.Get_size() # new: gives number of ranks in comm
  rank = comm.Get_rank()
  partials=[]
  for i in Z1,D1,OM,K:
   partials.append( np.empty(int(len(i)/size), dtype='d'))
   comm.Scatter(i, partials[-1], root=0)



 def gap(self,Z1,D1,OM,K,beta,nshift,OMP):
   allZ=np.zeros((len(OM)))
   allD=np.zeros((len(OM)))
   for nom1,om1 in enumerate(OM):  
    Z=0 
    D=0
    for nom2,om2 in enumerate(OM): #[:nom1]):
     mian=np.sqrt( om2**2  + D1[nom2]**2)
     Z+=om2/om1*K[nshift+nom1-nom2]/mian
    # print(OMP[nshift+nom1-nom2],om1-om2)
     D+= ( (K[nshift+nom1-nom2]-self.mu))*D1[nom2]/mian #if abs(om2)<omega_cut:
    Z=Z*np.pi/beta    + 1
    allZ[nom1]=Z #max(0.1,Z)
    D=D*np.pi/beta / Z1[nom1] #/len(OM)
    allD[nom1]=D #max(0.1,D)
   return allD,allZ
 

 def read_a2f(self):
  if len(self.A2F)==0:
   h='a2f.dat-vasp'
   tmp=np.loadtxt(h,skiprows=1)
   tmp=tmp[tmp[:,0]>0,:]
   self.OM_a2f=tmp[:,0]*thz_to_mev
   self.A2F=tmp[:,1]/2 #per spin
  f = interpolate.interp1d(self.OM_a2f, self.A2F)
  #self.OM_a2f=self.OM_a2f[::10]
  #self.A2F=f(self.OM_a2f)
  self.OM_a2f=self.OM_a2f*thz_to_mev
  self.a2fom=self.A2F*self.OM_a2f
  self.omom=self.OM_a2f*self.OM_a2f
  print(self.a2fom.shape)
 
 def calc_mu(self):
  self.omega_cut=20*self.OM_a2f[-1]
#bandwidth=calc_bandwidth()
#print('bandwidth',bandwidth)
#mu=mustar/(1-mustar*np.log(bandwidth/(OM_a2f[-1]/ry_to_mev)))
  self.mu=1./(1./mu_allendynes-np.log(self.omega_cut/self.OM_a2f[-1]))
  print('mu recalculated wit wc/wmax=',self.mu)



 def solve_eq(self):
  allgaps=[]
  for T in self.allT:
    beta=1./(T*k_to_mev)
    nmax=int(np.floor(0.5*(self.omega_cut *beta/np.pi-1)))
    nmin=int(np.ceil(0.5*(-self.omega_cut *beta/np.pi-1)))
    nshift=abs(nmin-nmax)
    OM=[(2*n+1)*np.pi/beta   for n in range(nmin,nmax+1)]
    OMP=[(2*p)*np.pi/beta   for p in range(nmin-nmax,2*nmax+2)]
    K=[np.trapz( 2*self.a2fom/( (omp)**2+self.omom),self.OM_a2f)  for omp in OMP]
    print('lambda for p=0: ',K[abs(nmin-nmax)+1])
    #print(OM)
    if len(OM)==0:
     allgaps.append(0)
     continue     

#    OM=np.array(OM[:-1])
    print(T,len(OM),self.omega_cut,OM[-1],OM[0])

    D=np.ones((len(OM)))*OM
    Z=np.ones((len(OM)))*OM
    for i in range(10000):
     D2,Z2=self.gap(Z,D,OM,K,beta,nshift,OMP ) 
     w1=np.mean(np.abs((D2-D)))
     w2=np.mean(np.abs((Z2-Z)))

     if (w1<1e-5  and w2<1e-5): 
         break
     D=D2
     Z=Z2
#         print(i)
     if i==100: print('warning: reached 100 iterations')
    print('end of T=',T,' with values iter=',i,' gap=',D[-nmin],'nw',len(OM))
#    plt.plot(OM,D)
    allgaps.append(D[-nmin])


  h=open('gap.dat','w')
  for i in range(len(self.allT)):
   h.write(str(self.allT[i])+' '+str(allgaps[i])+'\n')
  h.close()


  plt.plot(self.allT,allgaps)
  plt.show()

 

#eliashberg=eliashberg_solver()
