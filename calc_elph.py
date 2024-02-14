import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interpn
from multiprocessing import Process,Pool
import copy
PRECIS=6
ELECTRONMASS_SI  = 9.10938215e-31   # Kg
AMU_SI           = 1.660538782e-27  #Kg
AMU_AU           = AMU_SI / ELECTRONMASS_SI
AMU_RY           = AMU_AU / 2. #=911.44
RY_TO_THZ=3289.8449
RY_TO_GHZ=RY_TO_THZ*1000
RY_TO_CM_1=9.1132804961430295837e-6
eV_to_THz=241.7991

def construct_tetra(nk1,nk2,nk3):
	tetra=[[0 for k in range(6*nk1*nk2*nk3)] for i in range(4)]
	for i in range(nk1):
		for j in range(nk2):
			for k in range(nk3):
           #  n1-n8 are the indices of k-point 1-8 forming a cube
				ip1 = (i+1)%nk1 #np.mod(i+1,nk1)
				jp1 = (j+1)%nk2 #np.mod(j+1,nk2)
				k1 = (k+1)%nk3 #np.mod(k+1,nk3)
				n1 = k   + j  *nk3 + i   *nk2*nk3 
				n2 = k   + j  *nk3 + ip1 *nk2*nk3 
				n3 = k   + jp1*nk3 + i   *nk2*nk3 
				n4 = k   + jp1*nk3 + ip1 *nk2*nk3 
				n5 = k1 + j  *nk3 + i   *nk2*nk3 
				n6 = k1 + j  *nk3 + ip1 *nk2*nk3 
				n7 = k1 + jp1*nk3 + i   *nk2*nk3 
				n8 = k1 + jp1*nk3 + ip1 *nk2*nk3 
           #  there are 6 tetrahedra per cube (and nk1*nk2*nk3 cubes)
				n  = 6 * ( k + j*nk3 + i*nk3*nk2 )
				tetra [0][n] =n1
				tetra [1][n] = n2
				tetra [2][n] = n3
				tetra [3][n] = n6

				tetra [0][n+1] = n2
				tetra [1][n+1] = n3
				tetra [2][n+1] = n4
				tetra [3][n+1] = n6

				tetra [0][n+2] = n1
				tetra [1][n+2] = n3
				tetra [2][n+2] = n5
				tetra [3][n+2] = n6

				tetra [0][n+3] = n3
				tetra [1][n+3] = n4
				tetra [2][n+3] = n6
				tetra [3][n+3] = n8

				tetra [0][n+4] = n3
				tetra [1][n+4] = n6
				tetra [2][n+4] = n7
				tetra [3][n+4] = n8

				tetra [0][n+5] = n3
				tetra [1][n+5] = n5
				tetra [2][n+5] = n6
				tetra [3][n+5] = n7
	return tetra

def calc_dos(tetra, gamma,et,ibnd, ef=0):
  ntetra=len(tetra[0])
  Tint = 0.0 
  o13 = 1.0 /3.0 
  eps  = 1.0e-14
  voll  = 1.0 /ntetra
  P1 = 0.0 
  P2 = 0.0 
  P3 = 0.0 
  P4 = 0.0 
  for nt in range(ntetra):
      #
      # etetra are the energies at the vertexes of the nt-th tetrahedron
      #
     etetra=[]
     for i in range(4):
        etetra.append( et[ibnd][tetra[i][nt]])

     itetra=np.argsort(etetra) #hpsort (4,etetra,itetra)
     etetra=[ etetra[i] for i in itetra]
      #
      # ...sort in ascending order: e1 < e2 < e3 < e4
      #
     [e1,e2,e3,e4] = [etetra[0],etetra[1],etetra[2],etetra[3]]

      #
      # k1-k4 are the irreducible k-points corresponding to e1-e4
      #
     ik1,ik2,ik3,ik4 = tetra[itetra[0]][nt],tetra[itetra[1]][nt],tetra[itetra[2]][nt],tetra[itetra[3]][nt]
     Y1,Y2,Y3,Y4  = gamma[ibnd][ik1],gamma[ibnd][ik2],gamma[ibnd][ik3],gamma[ibnd][ik4]

     if( e3 < ef and ef < e4): # THEN
        f14 = (ef-e4)/(e1-e4)
        f24 = (ef-e4)/(e2-e4)
        f34 = (ef-e4)/(e3-e4)

        G  =  3.0  * f14 * f24 * f34 / (e4-ef)
        P1 =  f14 * o13
        P2 =  f24 * o13
        P3 =  f34 * o13
        P4 =  (3.0  - f14 - f24 - f34 ) * o13

     elif ( e2 < ef and ef < e3 ):

        f13 = (ef-e3)/(e1-e3)
        f31 = 1.0  - f13
        f14 = (ef-e4)/(e1-e4)
        f41 = 1.0 -f14
        f23 = (ef-e3)/(e2-e3)
        f32 = 1.0  - f23
        f24 = (ef-e4)/(e2-e4)
        f42 = 1.0  - f24

        G   =  3.0  * (f23*f31 + f32*f24)
        P1  =  f14 * o13 + f13*f31*f23 / G
        P2  =  f23 * o13 + f24*f24*f32 / G
        P3  =  f32 * o13 + f31*f31*f23 / G
        P4  =  f41 * o13 + f42*f24*f32 / G
        G   =  G / (e4-e1)

     elif ( e1 < ef and ef < e2 ) :

        f12 = (ef-e2)/(e1-e2)
        f21 = 1.0  - f12
        f13 = (ef-e3)/(e1-e3)
        f31 = 1.0  - f13
        f14 = (ef-e4)/(e1-e4)
        f41 = 1.0  - f14

        G  =  3.0  * f21 * f31 * f41 / (ef-e1)
        P1 =  o13 * (f12 + f13 + f14)
        P2 =  o13 * f21
        P3 =  o13 * f31
        P4 =  o13 * f41

     else:

        G = 0.0 

     Tint = Tint + G * (Y1*P1 + Y2*P2 + Y3*P3 + Y4*P4) 

  dos_gam = Tint* voll

  return dos_gam  # #2 because DOS_ee is per 1 spin


def w0gauss(x,degauss=0.6):
  mask=200/13.606*np.ones(x.shape)
  x2=x/degauss
  sqrtpm1= 1. / 1.77245385090551602729
  sqrt2=2.**0.5
  # cold smearing  (Marzari-Vanderbilt-DeVita-Payne)
  try: arg=np.minimum(mask,(x2 - 1.0 /  sqrt2 ) **2.)
  except: arg=min(200/13.606,x2**2)
  return  sqrtpm1 *np.exp ( - arg) * (2.0 - sqrt2 * x2)/degauss  #exp(-arg)*sqrtpm1/degauss 
  #gauss
  #try: arg=np.minimum(mask,x2 **2.)
  #except: arg=min(200/13.606,x2**2)
  #return  np.exp(-arg)*sqrtpm1/degauss 
  
class lambd():
 def __init__(self,data):
   ELPH_sum=[]  

 def init_tetra(self,data):
  self.ENE_all=[[bnd[k[3]] for k in data.vkpt_fbz] for bnd in data.ENE]
  self.tetra=construct_tetra(*data.nkpx)
  self.tetra_noneq=[[data.vkpt_fbz[m][3] for m in t ] for t in self.tetra] 

 def calc_lambda(self,data): 
  self.LAMBDA_sum=np.zeros(shape=(data.no_of_bands,\
           len(data.vkpt_ibz)),dtype=float)
  for numk, k in enumerate(data.vkpt_ibz): 
     for numkp, kp in enumerate(data.vkpt_fbz):
      for nu in range(data.no_of_modes):
       if(data.freq[numkp,numk,nu]<0.01/(3289.8449/13.606)): data.elph[numkp,numk,nu,:,:]=0

  '''
  for jband in range(data.no_of_bands):
   for numk, k in enumerate(data.vkpt_ibz): 
    if abs(data.ENE[jband][numk])>2: continue
    for iband in range(data.no_of_bands):
     for nu in range(data.no_of_modes):
        self.LAMBDA_sum[jband][numk] +=2*calc_dos(self.tetra,(np.absolute((data.elph[:,numk,nu,:,jband])**2 /data.freq[:,numk,nu, np.newaxis])).transpose(),self.ENE_all,iband)
  dos=np.sum([calc_dos(self.tetra_noneq,np.ones(data.ENE.shape),data.ENE,iband) for iband in range(data.no_of_bands)])
  print('dos',dos)
  print('lambda',np.sum([calc_dos(self.tetra_noneq,self.LAMBDA_sum,data.ENE,iband) for iband in range(data.no_of_bands)])/dos )
  '''
  for jband in range(data.no_of_bands):
   for numk, k in enumerate(data.vkpt_ibz): 
    if abs(data.ENE[jband][numk])>2: continue 
    for iband in range(data.no_of_bands):
     for numkp, kp in enumerate(data.vkpt_fbz): 
      if abs(data.ENE[iband][kp[3]])>2: continue
      weight=w0gauss(-data.ENE[iband][kp[3]]) #*weight0 #*self.WEIGHTS_OF_K[self.kQ[numkp][1]]
      for nu in range(data.no_of_modes):
          self.LAMBDA_sum[jband][numk] += 2*np.absolute((data.elph[numkp][numk][nu][iband][jband]))**2/data.freq[numkp][numk][nu] * weight
  self.LAMBDA_sum/=len(data.vkpt_fbz)

  dos=np.sum([calc_dos(self.tetra_noneq,np.ones(data.ENE.shape),data.ENE,iband) for iband in range(data.no_of_bands)])
  print('dos',dos)
  print('lambda',np.sum([calc_dos(self.tetra_noneq,self.LAMBDA_sum,data.ENE,iband) for iband in range(data.no_of_bands)])/dos )

  data.write_frmsf(data.nkpx,data.e,data.ENE,data.vkpt_fbz,self.LAMBDA_sum,'lambda')


 def calc_a2f(self,data): 

  self.A2F=np.zeros(shape=(100,data.no_of_bands,\
           len(data.vkpt_ibz)),dtype=float) 
  OM=np.linspace(0,np.max(data.freq),100)
  freq2=[[data.no_of_bands*[list(data.freq[:,numk,nu])]  for numk in range(len(data.vkpt_ibz))] for nu in range(data.no_of_modes)]
  for jband in range(data.no_of_bands):
   for numk, k in enumerate(data.vkpt_ibz): 
    if abs(data.ENE[jband][numk])>2: continue
    for iband in range(data.no_of_bands):
      for nw,w in enumerate(OM):
       print(nw,w)
       for nu in range(data.no_of_modes):
         self.A2F[nw][jband][numk] +=calc_dos(self.tetra,(np.absolute((data.elph[:,numk,nu,:,jband])**2 )).transpose(),self.ENE_all+w-freq2[nu][numk],iband)

  #dos=np.sum([calc_dos(self.tetra_noneq,np.ones(data.ENE.shape),data.ENE,iband) for iband in range(data.no_of_bands)])
  #print('dos',dos)
  #data.write_frmsf(data.nkpx,data.e,data.ENE,data.vkpt_fbz,self.ELPH_sum,'lambda')

  #print('lambda',np.sum([calc_dos(self.tetra_noneq,self.ELPH_sum,data.ENE,iband) for iband in range(data.no_of_bands)])/dos )



 def calc_a2f_smearing(self,data): 
  nw=200
  degauss=np.max(data.freq)/nw*3 #for dirac delta with phonon frequencies
  self.ELPH_sum=np.zeros(shape=(len(data.vkpt_fbz),data.no_of_modes),dtype=float)
  self.A2F=np.zeros(shape=(nw),dtype=float) 
  self.F=np.zeros(shape=(nw),dtype=float)
  for numk, k in enumerate(data.vkpt_ibz): 
     for numkp, kp in enumerate(data.vkpt_fbz):
      for nu in range(data.no_of_modes):
       if(data.freq[numkp,numk,nu]<0.1/(3289.8449/13.606)): data.elph[numkp,numk,nu,:,:]=0

  OM=np.linspace(0,np.max(data.freq),nw)
  for jband in range(data.no_of_bands):
   print(jband)
   for numk, k in enumerate(data.vkpt_ibz): 
    if abs(data.ENE[jband][numk])>2: continue
    weight0=w0gauss(-data.ENE[jband][numk]) *data.k_weights[numk]
    for iband in range(data.no_of_bands):
     for numkp, kp in enumerate(data.vkpt_fbz): 
      if abs(data.ENE[iband][kp[3]])>2: continue
      weight=w0gauss(-data.ENE[iband][kp[3]])*weight0 #*weight0 #*self.WEIGHTS_OF_K[self.kQ[numkp][1]]
      for nu in range(data.no_of_modes):
       self.ELPH_sum[data.kkp_to_q[numkp][numk]][nu] += np.absolute((data.elph[numkp][numk][nu][iband][jband]))**2 * weight
  self.ELPH_sum/=len(data.vkpt_fbz)

  for nw,w in enumerate(OM):
   for nq in range(len(data.vkpt_fbz)):
    for nu in range(data.no_of_modes):
      weight=w0gauss(w-data.freq_q[nq,nu],degauss)
      self.A2F[nw]+=weight*self.ELPH_sum[nq][nu]
      self.F[nw]+=weight
  self.A2F/=len(data.vkpt_fbz)
  self.F/=len(data.vkpt_fbz)
  h=open('a2f.dat','w')
  h.write('#w(THz), a2F, F\n')
  for nw,w in enumerate(OM):
    h.write(str(w*eV_to_THz)+' '+str(self.A2F[nw]/eV_to_THz)+' '+str(self.F[nw]/eV_to_THz)+'\n')
  h.close()
  #dos=np.sum([calc_dos(self.tetra_noneq,np.ones(data.ENE.shape),data.ENE,iband) for iband in range(data.no_of_bands)])
  #print('dos',dos)
  #data.write_frmsf(data.nkpx,data.e,data.ENE,data.vkpt_fbz,self.ELPH_sum,'lambda')

  #print('lambda',np.sum([calc_dos(self.tetra_noneq,self.ELPH_sum,data.ENE,iband) for iband in range(data.no_of_bands)])/dos )

 def calc_a2f_smearing(self,data): 
  nw=200
  degauss=np.max(data.freq)/nw*3 #for dirac delta with phonon frequencies
  self.ELPH_sum=np.zeros(shape=(len(data.vkpt_fbz),data.no_of_modes),dtype=float)
  self.A2F=np.zeros(shape=(nw),dtype=float) 
  self.F=np.zeros(shape=(nw),dtype=float)
  for numk, k in enumerate(data.vkpt_ibz): 
     for numkp, kp in enumerate(data.vkpt_fbz):
      for nu in range(data.no_of_modes):
       if(data.freq[numkp,numk,nu]<0.1/(3289.8449/13.606)): data.elph[numkp,numk,nu,:,:]=0

  OM=np.linspace(0,np.max(data.freq),nw)
  for jband in range(data.no_of_bands):
   print(jband)
   for numk, k in enumerate(data.vkpt_ibz): 
    if abs(data.ENE[jband][numk])>2: continue
    weight0=w0gauss(-data.ENE[jband][numk]) *data.k_weights[numk]
    for iband in range(data.no_of_bands):
     for numkp, kp in enumerate(data.vkpt_fbz): 
      if abs(data.ENE[iband][kp[3]])>2: continue
      weight=w0gauss(-data.ENE[iband][kp[3]])*weight0 #*weight0 #*self.WEIGHTS_OF_K[self.kQ[numkp][1]]
      for nu in range(data.no_of_modes):
       self.ELPH_sum[data.kkp_to_q[numkp][numk]][nu] += np.absolute((data.elph[numkp][numk][nu][iband][jband]))**2 * weight
  self.ELPH_sum/=len(data.vkpt_fbz)

  for nw,w in enumerate(OM):
   for nq in range(len(data.vkpt_fbz)):
    for nu in range(data.no_of_modes):
      weight=w0gauss(w-data.freq_q[nq,nu],degauss)
      self.A2F[nw]+=weight*self.ELPH_sum[nq][nu]
      self.F[nw]+=weight
  self.A2F/=len(data.vkpt_fbz)
  self.F/=len(data.vkpt_fbz)
  h=open('a2f.dat','w')
  h.write('#w(THz), a2F, F\n')
  for nw,w in enumerate(OM):
    h.write(str(w*eV_to_THz)+' '+str(self.A2F[nw]/eV_to_THz)+' '+str(self.F[nw]/eV_to_THz)+'\n')
  h.close()
  #dos=np.sum([calc_dos(self.tetra_noneq,np.ones(data.ENE.shape),data.ENE,iband) for iband in range(data.no_of_bands)])
  #print('dos',dos)
  #data.write_frmsf(data.nkpx,data.e,data.ENE,data.vkpt_fbz,self.ELPH_sum,'lambda')

  #print('lambda',np.sum([calc_dos(self.tetra_noneq,self.ELPH_sum,data.ENE,iband) for iband in range(data.no_of_bands)])/dos )

   
 def calc_dos(self,data):
  tetra=construct_tetra(*data.nkpx)
  tetra_noneq=[[data.vkpt_fbz[m][3] for m in t ] for t in tetra]
  data.ENE_v=data.ENE
  data.weights_v=data.k_weights
  ne=50
  de=(np.max(data.ENE_v)-np.min(data.ENE_v))/ne
  e=np.linspace(np.min(data.ENE_v),np.max(data.ENE_v),ne)
  h=open('dos.dat','w')
  for i in e:
    print(i)
    h.write(str(i)+' ')
    h.write(str(np.sum([calc_dos(tetra_noneq, np.ones(data.ENE.shape),data.ENE-i,iband ) for iband in range(data.no_of_bands)]) ))
    h.write('\n')
  h.close()
  
  #self.ENE=self.ENE_v
