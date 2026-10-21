import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interpn
from multiprocessing import Process,Pool
import copy
from calc_elph import *
PRECIS=6
ELECTRONMASS_SI  = 9.10938215e-31   # Kg
AMU_SI           = 1.660538782e-27  #Kg
AMU_AU           = AMU_SI / ELECTRONMASS_SI
AMU_RY           = AMU_AU / 2. #=911.44
RY_TO_THZ=3289.8449
RY_TO_GHZ=RY_TO_THZ*1000
RY_TO_CM_1=9.1132804961430295837e-6
eV_to_THz=241.7991
k_to_mev=8.6173e-2
thz_to_mev=4.1357 #/2/np.pi
ha_to_ry=2
ry_to_mev=13605.662285137 #/2/np.pi
ry_to_ev=ry_to_mev/1000
ev_to_mev=1000


def w0gauss(x,degauss=0.6):
  mask=200/13.606*np.ones(x.shape)
  x2=x/degauss
  sqrtpm1= 1. / 1.77245385090551602729
  sqrt2=2.**0.5
  # cold smearing  (Marzari-Vanderbilt-DeVita-Payne)
#  try: arg=np.minimum(mask,(x2 - 1.0 /  sqrt2 ) **2.)
#  except: arg=min(200/13.606,x2**2)
#  return  sqrtpm1 *np.exp ( - arg) * (2.0 - sqrt2 * x2)/degauss  #exp(-arg)*sqrtpm1/degauss 
  #gauss
  try: arg=np.minimum(mask,x2 **2.)
  except: arg=min(200/13.606,x2**2)
  return  np.exp(-arg)*sqrtpm1/degauss 


  

class eliashberg_aniso_solver():
 def __init__(self,data):
   mu_allendynes=0.1
   self.allT=[3]
  # for i in range(35):
  #   self.allT.append(1.5+i/5.)
   self.no_of_modes=data.no_of_modes
   self.nbnd=len(data.ENE)
   #self.nkp_full=len(data.vkpt_k)
   self.nkp=len(data.vkpt_k)
   self.ENE_all=data. ENE #[[bnd[k[3]] for k in data.vkpt_fbz] for bnd in data.ENE]
   self.delta_all=w0gauss(data.ENE) /ev_to_mev
   #self.delta_all=np.array([[bnd[k[3]] for k in data.vkpt_fbz] for bnd in self.delta])
   self.deltasum=np.sum(self.delta_all)
   print ('deltasum',self.deltasum)
   nw0=200
   nw=nw0+16
   for numk, k in enumerate(data.vkpt_k): 
     for numkp, kp in enumerate(data.vkpt_kp):
      for nu in range(data.no_of_modes):
       if(data.freq[numkp,numk,nu]<0.1/(3289.8449/13.606)): data.elph[numkp,numk,nu,:,:]=0
       if(data.freq[numkp,numk,nu]==0): data.freq[numkp,numk,nu]=1e-7 
   self.OM_a2f=np.linspace(1e-7,np.max(data.freq),100)*ev_to_mev
   self.ph_freq=data.freq*ev_to_mev
   self.degauss_ph=np.max(self.ph_freq)/nw0*3 #for dirac delta with phonon frequencies
   self.omega_cut=4*self.OM_a2f[-1]
#bandwidth=calc_bandwidth()
#print('bandwidth',bandwidth)
#mu=mustar/(1-mustar*np.log(bandwidth/(OM_a2f[-1]/ry_to_mev)))
   self.mu=1./(1./mu_allendynes-np.log(self.omega_cut/self.OM_a2f[-1]))
   self.no_of_pool = 1
   #self.mu/=np.sum(self.delta_all)

   self.tetra=construct_tetra(*data.nkpx)
 # self.tetra_noneq=[[data.vkpt_fbz[m][3] for m in t ] for t in self.tetra] 
   self.tetra_WEIGHTS,self.tetra_GS,self.tetra_sorted=dos_weights(self.tetra, self.ENE_all, ef=0)
   self.tetra_WEIGHTS_kp,self.tetra_GS_kp,self.tetra_sorted_kp=dos_weights(self.tetra, data.ENE_kp, ef=0)

     
 
 def solve_eq(self,data):
  allgaps=[]
  for T in self.allT:
    self.beta=1./(T*k_to_mev)
    nmax=int(np.floor(0.5*(self.omega_cut *self.beta/np.pi-1)))
    nmin=int(np.ceil(0.5*(-self.omega_cut *self.beta/np.pi-1)))
    print(nmin,nmax)
    self.nshift=abs(nmin-nmax)
    self.OM=np.array([(2*n+1)*np.pi/self.beta   for n in range(nmin,nmax+1)])
    self.OMP=np.array([(2*p)*np.pi/self.beta   for p in range(nmin-nmax,2*nmax+2)])
    self.parallel_lambda(data)

    if len(self.OM)==0:
     allgaps.append(0)
     continue    
 
    print(T,len(self.OM),self.omega_cut,self.OM[-1],self.OM[0])
    self.D=np.ones((self.nbnd,self.nkp, len(self.OM)))*self.OM
    self.Z=np.ones((self.nbnd,self.nkp, len(self.OM)))*self.OM 
    for i in range(10000):
     D2,Z2=self.parallel_gap()
     w1=np.mean(np.abs((D2-self.D))) #/np.sum(np.abs((self.D)))
     w2=np.mean(np.abs((Z2-self.Z)))# /np.sum(np.abs((self.Z)))
     print(w1,w2)
    # print(np.mean(np.abs((Z2-self.Z))),np.mean(np.abs((D2-self.D))))
     if (w1<1e-5  and w2<1e-5): 
         break
     self.D=D2
     self.Z=Z2 
     if i==100: print('warning: reached 100 iterations')
    print('end of T=',T,' with values iter=',i,' gap=',self.D[:,:,-nmin],'nw',len(self.OM))
    allgaps.append(self.D[:,:,-nmin])
    data.write_frmsf(data.nkpx,data.e,data.ENE,self.D[:,:,-nmin],'delta')

 def parallel_gap(self):
  D=np.zeros(shape=(self.nbnd,self.nkp,len(self.OM)),dtype=float)
  Z=np.zeros(shape=(self.nbnd,self.nkp,len(self.OM)),dtype=float)

  print(self.no_of_pool, "proccesses about to be launched")
  with Pool(self.no_of_pool) as pol:
   results=pol.map(self.calc_gap_k,\
              [q for q in range(self.no_of_pool)])
   for q in range(self.no_of_pool):
      D[:,self.nkmin_nkmax[q][0]:self.nkmin_nkmax[q][1],:],Z[:,self.nkmin_nkmax[q][0]:self.nkmin_nkmax[q][1],:]= results[q] #i collect it just in the case i'd like to visualize it later
  return D,Z

 def calc_gap_k(self,q):
  D=np.zeros(shape=(self.D[:,self.nkmin_nkmax[q][0]:self.nkmin_nkmax[q][1],:]).shape,dtype=float)
  Z=np.zeros(shape=(self.D[:,self.nkmin_nkmax[q][0]:self.nkmin_nkmax[q][1],:]).shape,dtype=float)
  for iband in range(self.nbnd):
   for k in range(self.nkmin_nkmax[q][0],self.nkmin_nkmax[q][1]):
     D[iband,k-self.nkmin_nkmax[q][0]],Z[iband,k-self.nkmin_nkmax[q][0]]=self.calc_gap_given_k(self.lambda_kkp[iband,k,:,:,:],self.Z[iband,k,:])
  return D,Z


 def calc_gap_given_k(self,K,oldZ):
   allZ=np.zeros(self.OM.shape)
   allD=np.zeros(self.OM.shape)
   #print(K.shape)
   for nom1,om1 in enumerate(self.OM):  
    Z=0 
    D=0
    for nom2,om2 in enumerate(self.OM): #[:nom1]):
     mian=np.sqrt(om2**2  + self.D[:,:,nom2]**2)
     Z+=om2*K[:,:,self.nshift+nom1-nom2]/mian  
     D+= ( (K[:,:,self.nshift+nom1-nom2]-self.mu))*self.D[:,:,nom2]/mian #if abs(om2)<omega_cut:
    Z=Z*np.pi/self.beta*self.delta_all /om1 
    allZ[nom1]=np.sum(Z,axis=(0,1)) /self.deltasum   + 1 #sum over jband kp
    D=D*np.pi/self.beta*self.delta_all   #/len(OM)
    allD[nom1]=np.sum(D,axis=(0,1))/self.deltasum #sum over jband kp
    allD[nom1]/= oldZ[nom1]
   return allD,allZ


 def parallel_lambda(self,data):
  self.lambda_kkp=np.zeros(shape=(self.nbnd,data.vkpt,  self.nbnd,len(data.vkpt_kp),  len(self.OMP)),dtype=float)
  self.lambdaa=np.zeros(shape=(self.nbnd,data.vkpt),dtype=float)
  print("parallel lambda")
  self.nk_per_pool=self.nkp/(self.no_of_pool-1) if self.no_of_pool >1 else self.nkp
  self.nkmin_nkmax=[[int(q*self.nk_per_pool),int(min(self.nkp,(q+1)*self.nk_per_pool))] for q in range(self.no_of_pool)]
  with Pool(self.no_of_pool) as pol:
   results=pol.map(self.calc_lambda_kkp,\
              [[data,*self.nkmin_nkmax[q]] for q in range(self.no_of_pool)])
   for q in range(self.no_of_pool):
      self.lambda_kkp[:,self.nkmin_nkmax[q][0]:self.nkmin_nkmax[q][1],:,:]=results[q] #i collect it just in the case i'd like to visualize it later
  for jband in range(self.nbnd):
   for numk in range(self.nkp): 
     #self.lambdaa[jband,numk]=np.sum(self.lambda_kkp[jband,numk,:,:,int(len(self.OMP)/2)]*self.delta_all)/self.deltasum
     lam_tetra=np.array([[[ self.lambda_kkp[nib,nk] for nk in t]  for nt,t in enumerate(ib)]  for nib,ib in enumerate(self.tetra_sorted_kp)])
     self.lambdaa[jband][numk] += np.sum(self.tetra_GS_kp*np.sum(lam_tetra*self.tetra_WEIGHTS_kp,axis=2) )
  self.lambdaa/=len(self.tetra)
  data.write_frmsf(data.nkpx,data.e,data.ENE,self.lambdaa,'lambda')
  print('max lambnda',np.max(self.lambdaa))
  lam_tetra=np.array([[[ self.LAMBDA_sum[nib,nk] for nk in t]  for nt,t in enumerate(ib)]  for nib,ib in enumerate(self.tetra_sorted)])
  lam=np.sum(self.tetra_GS*np.sum(lam_tetra*self.tetra_WEIGHTS,axis=2) )/len(self.tetra)

  print('lambda', lam)

  
 def calc_lambda_kkp(self,args):
  try:
   [data,nkmin,nkmax]=args 
  except:
   data=args
   nkmin,nkmax=0,-1
 # print(nkmin,nkmax)
  lambda_kkp=np.zeros(shape=(self.nbnd,nkmax-nkmin,  self.nbnd,self.nkp,len(self.OMP)),dtype=float)
  print('min max freq (THz)',np.min(data.freq)*eV_to_THz,np.max(data.freq)*eV_to_THz)

  for jband in range(self.nbnd):
   for numk in range(nkmin,nkmax): 
   # print(numk)
    if abs(data.ENE[jband][numk])>2: continue 
    for iband in range(self.nbnd):
     for numkp, kp in enumerate(data.vkpt_kp): 
      if abs(data.ENE_kp[iband][numkp])>2: continue
      oma2f_a2f=self.OM_a2f*self.calc_a2f_kkp(data,jband,numk,iband,numkp)
      for nw,w in enumerate(self.OMP): 
         lambda_kkp[jband][numk-nkmin][iband][numkp][nw]= 2*np.trapz(oma2f_a2f/((w)**2+self.OM_a2f**2),x=self.OM_a2f)
   print(nkmin,nkmax,'finished')
   return lambda_kkp

 def calc_a2f_kkp(self,data,jband,numk,iband,numkp): 
  a2f_kkp=np.zeros(shape=(len(self.OM_a2f)),dtype=float)
  for nw,w in enumerate(self.OM_a2f):
       for nu in range(self.no_of_modes):
          a2f_kkp[nw]+= np.absolute((data.elph[numkp][numk][nu][iband][jband]))**2*w0gauss(w-self.ph_freq[numkp][numk][nu],self.degauss_ph)
  return a2f_kkp*self.deltasum/self.nkp*ev_to_mev*ev_to_mev #because data.elph is in ev
