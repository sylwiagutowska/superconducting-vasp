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
	tetra=[[0 for k in range(4)] for i in range(6*nk1*nk2*nk3)]
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
				tetra [n][0] =n1
				tetra [n][1] = n2
				tetra [n][2] = n3
				tetra [n][3] = n6

				tetra [n+1][0] = n2
				tetra [n+1][1] = n3
				tetra [n+1][2] = n4
				tetra [n+1][3] = n6

				tetra [n+2][0] = n1
				tetra [n+2][1] = n3
				tetra [n+2][2] = n5
				tetra [n+2][3] = n6

				tetra [n+3][0] = n3
				tetra [n+3][1] = n4
				tetra [n+3][2] = n6
				tetra [n+3][3] = n8

				tetra [n+4][0] = n3
				tetra [n+4][1] = n6
				tetra [n+4][2] = n7
				tetra [n+4][3] = n8

				tetra [n+5][0] = n3
				tetra [n+5][1] = n5
				tetra [n+5][2] = n6
				tetra [n+5][3] = n7
	return tetra

def dos_weights(tetra, et, ef=0):
  ntetra=len(tetra)
  Tint = 0.0 
  o13 = 1.0 /3.0 
  eps  = 1.0e-14
  voll  = 1.0 /ntetra
  P1 = 0.0 
  P2 = 0.0 
  P3 = 0.0 
  P4 = 0.0 
  WEIGHTS=[]
  GS=[]
  TETRA_sorted=[]
  for ibnd in range(len(et)):
   WEIGHTS.append([])
   GS.append([])
   TETRA_sorted.append([])
   for nt in range(ntetra):
      #
      # etetra are the energies at the vertexes of the nt-th tetrahedron
      #
     
     etetra=[]
     for i in range(4):
        etetra.append( et[ibnd][tetra[nt][i]])

     itetra=np.argsort(etetra) #hpsort (4,etetra,itetra)
     TETRA_sorted[-1].append([tetra[nt][i] for i in itetra])
     etetra=[ etetra[i] for i in itetra]
      #
      # ...sort in ascending order: e1 < e2 < e3 < e4
      #
     [e1,e2,e3,e4] = [etetra[0],etetra[1],etetra[2],etetra[3]]

      #
      # k1-k4 are the irreducible k-points corresponding to e1-e4
      #
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
     WEIGHTS[-1].append([P1,P2,P3,P4])
     GS[-1].append(G)

  return np.array(WEIGHTS),np.array(GS)  , np.array(TETRA_sorted,dtype=int)


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
        etetra.append( et[ibnd][tetra[nt][i]])

     itetra=np.argsort(etetra) #hpsort (4,etetra,itetra)
     etetra=[ etetra[i] for i in itetra]
      #
      # ...sort in ascending order: e1 < e2 < e3 < e4
      #
     [e1,e2,e3,e4] = [etetra[0],etetra[1],etetra[2],etetra[3]]

      #
      # k1-k4 are the irreducible k-points corresponding to e1-e4
      #
     ik1,ik2,ik3,ik4 = tetra[nt][itetra[0]],tetra[nt][itetra[1]],tetra[nt][itetra[2]],tetra[nt][itetra[3]]
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
#  try: arg=np.minimum(mask,(x2 - 1.0 /  sqrt2 ) **2.)
#  except: arg=min(200/13.606,x2**2)
#  return  sqrtpm1 *np.exp ( - arg) * (2.0 - sqrt2 * x2)/degauss  #exp(-arg)*sqrtpm1/degauss 
  #gauss
  try: arg=np.minimum(mask,x2 **2.)
  except: arg=min(200/13.606,x2**2)
  return  np.exp(-arg)*sqrtpm1/degauss 


  

class lambd():
 def __init__(self,data):
   ELPH_sum=[]  

 def init_tetra(self,data):
  self.ENE=data.ENE #[[bnd[k[3]] for k in data.vkpt_fbz] for bnd in data.ENE]
  self.ENE_all=self.ENE 
  self.tetra=construct_tetra(*data.nkpx)
 # self.tetra_noneq=[[data.vkpt_fbz[m][3] for m in t ] for t in self.tetra] 
  self.tetra_WEIGHTS,self.tetra_GS,self.tetra_sorted=dos_weights(self.tetra, self.ENE_all, ef=0)
  self.tetra_WEIGHTS_kp,self.tetra_GS_kp,self.tetra_sorted_kp=dos_weights(self.tetra, data.ENE_kp, ef=0)

     
  self.dos=np.sum(self.tetra_GS*np.sum(self.tetra_WEIGHTS,axis=2) )/len(self.tetra)
  print('dos tetra',self.dos)

 def calc_lambda(self,data): 
  self.LAMBDA_sum=np.zeros(shape=(data.no_of_bands,\
           len(data.vkpt_k)),dtype=float)
  print('min max freq (THz)',np.min(data.freq)*eV_to_THz,np.max(data.freq)*eV_to_THz)
  print(data.elph.shape)
  data.freq[np.arange(len(data.vkpt_k)), np.arange(len(data.vkpt_k)), :3] = 0.0

  for numk, k in enumerate(data.vkpt_k): 
     for numkp, kp in enumerate(data.vkpt_kp):
      for nu in range(data.no_of_modes):
       if(data.freq[numkp,numk,nu]<0.1/(3289.8449/13.606)): 
         data.elph[numkp,numk,nu,:,:]=0
       if(data.freq[numkp,numk,nu]==0): 
         data.freq[numkp,numk,nu]=1e-7 

  print("tetra", self.tetra_sorted.shape)
  #sumweight=np.sum([[w0gauss(-data.ENE[iband][k[3]]) for k in data.vkpt_fbz] for iband in range(data.no_of_bands)])
  for jband in range(data.no_of_bands):
   for numk, k in enumerate(data.vkpt_k): 
    print(numk)
    if abs(data.ENE[jband][numk])>2: continue 
    for nu in range(data.no_of_modes):
     l=np.array([abs(data.elph[:,numk,nu,iband,jband])**2/data.freq[:,numk,nu] for iband in range(data.no_of_bands)])
     lam_tetra=np.array([[[ l[nib,nk] for nk in t]  for nt,t in enumerate(ib)]  for nib,ib in enumerate(self.tetra_sorted_kp)])
     self.LAMBDA_sum[jband][numk] += np.sum(self.tetra_GS_kp*np.sum(lam_tetra*self.tetra_WEIGHTS_kp,axis=2) )
  self.LAMBDA_sum/=len(self.tetra)
#     for iband in range(data.no_of_bands):
#      for numkp, kp in enumerate(data.vkpt_fbz): 
#       if abs(data.ENE[iband][kp[3]])>2: continue
#          weight=w0gauss(-data.ENE[iband][kp[3]]) #*weight0 #*self.WEIGHTS_OF_K[self.kQ[numkp][1]]
#          self.LAMBDA_sum[jband][numk] += 2*np.absolute((data.elph[numkp][numk][nu][iband][jband]))**2/data.freq[numkp][numk][nu] * weight
#  self.LAMBDA_sum/=sumweight
 
  lam_tetra=np.array([[[ self.LAMBDA_sum[nib,nk] for nk in t]  for nt,t in enumerate(ib)]  for nib,ib in enumerate(self.tetra_sorted)])
  lam=np.sum(self.tetra_GS*np.sum(lam_tetra*self.tetra_WEIGHTS,axis=2) )/len(self.tetra)

  print('lambda', lam)
  #dos=np.sum([calc_dos(self.tetra,np.ones(data.ENE.shape),data.ENE,iband) for iband in range(data.no_of_bands)])
  #print(np.min(data.ENE),np.max(data.ENE))
  #print('dos',dos)
  #print('lambda',np.sum([calc_dos(self.tetra_noneq,self.LAMBDA_sum,data.ENE,iband) for iband in range(data.no_of_bands)])) #/dos )

  data.write_frmsf(data.nkpx,data.e,data.ENE,self.LAMBDA_sum,'lambda')



 def calc_a2f(self, data):
    nfreq = 500
    nbands = data.no_of_bands
    nk = len(data.vkpt_k)
    nkp = len(data.vkpt_kp)
    nmode = data.no_of_modes
    #for numk, k in enumerate(data.vkpt_k): 
    #  for numkp, kp in enumerate(data.vkpt_kp):
    #    for nu in range(data.no_of_modes):
    #     if(data.freq[numkp,numk,nu]<0.1/(3289.8449/13.606)): data.elph[numkp,numk,nu,:,:]=0
    #     data.freq[numkp,numk,nu] = 0
    data.freq[np.arange(len(data.vkpt_k)), np.arange(len(data.vkpt_k)), :3] = 0.0
    for numk, k in enumerate(data.vkpt_k): 
     for numkp, kp in enumerate(data.vkpt_kp):
      for nu in range(data.no_of_modes):
       if(data.freq[numkp,numk,nu]<0.1/(3289.8449/13.606)): 
         data.elph[numkp,numk,nu,:,:]=0
       if(data.freq[numkp,numk,nu]==0): 
         data.freq[numkp,numk,nu]=1e-7 
    self.A2F = np.zeros((nfreq, nbands, nk), dtype=float)

    OM = np.linspace(0, np.max(data.freq), nfreq)
    degauss = 0.0001

    # Shape: (nfreq, nkp, nk, nmode)
    ph_weights = w0gauss(
        OM[:, None, None, None] - data.freq[None, :, :, :],
        degauss
    ) 



         
    F = ph_weights.sum(axis=(1, 2, 3))/ (nk * nkp)
    # Shapes:
    # tetra_sorted_kp  = (nbands, ntetra, 4)
    # tetra_WEIGHTS_kp = (nbands, ntetra, 4)
    # tetra_GS_kp      = (nbands, ntetra)
    kp_idx = self.tetra_sorted_kp

    for jband in range(nbands):
        for numk in range(nk):
            if abs(data.ENE[jband, numk]) > 2:
                continue

            print(numk)

            for nu in range(nmode):
                # Original:
                # a[iband, kp] =
                #     abs(data.elph[kp,numk,nu,iband,jband])**2
                #
                # Shape: (nbands, nkp)
                elph2 = np.abs(
                    data.elph[:, numk, nu, :, jband]
                ).T**2

                # Gather elph2 onto tetrahedron vertices.
                #
                # elph2[:, None, :]       -> (nbands, 1, nkp)
                # kp_idx                  -> (nbands, ntetra, 4)
                #
                # Result: (nbands, ntetra, 4)
                elph_tetra = np.take_along_axis(
                    elph2[:, None, :],
                    kp_idx,
                    axis=2
                )

                # Phonon weights for all frequencies.
                #
                # (nfreq, nkp)
                weights = ph_weights[:, :, numk, nu]

                # Gather phonon weights onto tetrahedron vertices.
                #
                # Result: (nfreq, nbands, ntetra, 4)
                weights_tetra = np.take_along_axis(
                    weights[:, None, None, :],
                    kp_idx[None, :, :, :],
                    axis=3
                )

                # Multiply everything and sum over:
                #   band
                #   tetrahedron
                #   tetrahedron vertex
                #
                # Result: (nfreq,)
                contribution = np.sum(
                    weights_tetra
                    * elph_tetra[None, :, :, :]
                    * self.tetra_WEIGHTS_kp[None, :, :, :]
                    * self.tetra_GS_kp[None, :, :, None],
                    axis=(1, 2, 3)
                )

                self.A2F[:, jband, numk] += (
                    contribution / len(self.tetra)
                )

    # ------------------------------------------------------------
    # Total A2F
    # ------------------------------------------------------------

    # Gather all k-point values onto tetrahedron vertices.
    #
    # A2F:             (nfreq, nbands, nk)
    # result:          (nfreq, nbands, ntetra, 4)
    a_tetra = np.take_along_axis(
        self.A2F[:, :, None, :],
        kp_idx[None, :, :, :],
        axis=3
    )

    A2F_tot = np.sum(
        a_tetra
        * self.tetra_WEIGHTS[None, :, :, :]
        * self.tetra_GS[None, :, :, None],
        axis=(1, 2, 3)
    ) / len(self.tetra)
 
    h=open("a2f_iso.dat", 'w')
    for nw,w in enumerate(OM):
      h.write(f"{w} {A2F_tot[nw]} {F[nw]}\n")
    h.close()
  #print('dos',dos)
    data.write_frmsf(data.nkpx,data.e,data.ENE,np.sum(self.A2F,axis=0),'a2f')

  #print('lambda',np.sum([calc_dos(self.tetra_noneq,self.ELPH_sum,data.ENE,iband) for iband in range(data.no_of_bands)])/dos )



 def calc_a2f_smearing(self,data): 
  nw0=200
  nw=nw0+16
  degauss=np.max(data.freq)/nw0*2 #for dirac delta with phonon frequencies
  self.ELPH_sum=np.zeros(shape=(len(data.vkpt_kp),data.no_of_modes),dtype=float)
  self.LINEWIDTH=np.zeros(shape=(len(data.vkpt_kp),data.no_of_modes),dtype=float)

  data.freq[np.arange(len(data.vkpt_k)), np.arange(len(data.vkpt_k)), :3] = 0.0
  data.freq_q[0, :3] = 0.0
  data.freq_q[data.freq_q==0] = 1e-7
  for numk, k in enumerate(data.vkpt_k): 
     for numkp, kp in enumerate(data.vkpt_kp):
      for nu in range(data.no_of_modes):
       if(data.freq[numkp,numk,nu]<0.1/(3289.8449/13.606)): 
         data.elph[numkp,numk,nu,:,:]=0
       if(data.freq[numkp,numk,nu]==0): 
         data.freq[numkp,numk,nu]=1e-7 

  self.A2F=np.zeros(shape=(nw),dtype=float) 
  self.F=np.zeros(shape=(nw),dtype=float) 

  dos=0
  OM=np.linspace(0,np.max(data.freq)*(1+16./nw0),nw)
  for jband in range(data.no_of_bands):
   print(jband)
   for numk, k in enumerate(data.vkpt_k): 
    if abs(data.ENE[jband][numk])>2: continue
    weight0=w0gauss(-data.ENE[jband][numk]) *data.k_weights[numk]
    dos+=weight0
    for iband in range(data.no_of_bands):
     for numkp, kp in enumerate(data.vkpt_kp): 
      if abs(data.ENE_kp[iband][numkp])>2: continue
      weight=w0gauss(-data.ENE_kp[iband][numkp])*weight0 #*weight0 #*self.WEIGHTS_OF_K[self.kQ[numkp][1]]
      for nu in range(data.no_of_modes):
       self.ELPH_sum[data.kkp_to_q[numkp][numk]][nu] += np.absolute((data.elph[numkp][numk][nu][iband][jband]))**2 * weight
       self.LINEWIDTH[data.kkp_to_q[numkp][numk]][nu] += np.absolute((data.elph[numkp][numk][nu][iband][jband]))**2 * weight/data.freq[numkp,numk,nu]
  self.LINEWIDTH/=len(data.vkpt_k)
  print(dos)
  self.ELPH_sum/=dos 
  h=open('linewidth.dat','w')
  for nq, q in enumerate(data.vkpt_q):
   for i in range(3):
    h.write(str(q[i])+" ")
   for m, w in enumerate(data.freq_q[nq]):  
     h.write(str(w)+' '+str(self.LINEWIDTH[nq][m]))
  h.close()
               
  for nw,w in enumerate(OM):
   for nq in range(len(data.vkpt_q)):
    for nu in range(data.no_of_modes):
      weight=w0gauss(w-data.freq_q[nq,nu],degauss)
      self.A2F[nw]+=weight*self.ELPH_sum[nq][nu]
      self.F[nw]+=weight
  self.A2F/=len(data.vkpt_k)
  self.F/=len(data.vkpt_k)
  h=open('a2f-smear.dat','w')
  h.write('#w(THz), a2F, F\n')
  for nw,w in enumerate(OM):
    h.write(str(w*eV_to_THz)+' '+str(self.A2F[nw])+' '+str(self.F[nw]/eV_to_THz)+'\n')
  h.close()
 
 
 
 def calc_a2f_smearing2(self,data): 
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
  
 def calc_a2f_smearing1(self,data): 
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
      weight1=w0gauss(-data.ENE[iband][kp[3]])*weight0 #*weight0 #*self.WEIGHTS_OF_K[self.kQ[numkp][1]]
      for nw,w in enumerate(OM):
       for nu in range(data.no_of_modes):
        weight2=w0gauss(w-data.freq[numkp][numk][nu])
        weight=weight1*weight2
        self.A2F[nw]+= np.absolute((data.elph[numkp][numk][nu][iband][jband]))**2 * weight
        self.F[nw]+=weight2
  self.A2F/=(len(data.vkpt_fbz)*len(data.vkpt_ibz))
  self.F/=(len(data.vkpt_fbz)*len(data.vkpt_ibz))

 
  h=open('a2f.dat','w')
  h.write('#w(THz), a2F, F\n')
  for nw,w in enumerate(OM):
    h.write(str(w*eV_to_THz)+' '+str(self.A2F[nw])+' '+str(self.F[nw]/eV_to_THz)+'\n')
  h.close()


 def calc_a2f_smearing3(self,data): 
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
    weight0=w0gauss(-data.ENE[jband][numk]) #*data.k_weights[k[3]]
    for iband in range(data.no_of_bands):
     for numkp, kp in enumerate(data.vkpt_fbz): 
      if abs(data.ENE[iband][kp[3]])>2: continue
      weight=w0gauss(-data.ENE[iband][kp[3]])*weight0 #*weight0 #*self.WEIGHTS_OF_K[self.kQ[numkp][1]]
      for nq in data.kkp_to_q[numkp][numk]:
       for nu in range(data.no_of_modes):
        self.ELPH_sum[nq][nu] += np.absolute((data.elph[numkp][numk][nu][iband][jband]))**2 * weight
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
  
     
 def calc_dos(self,data):
  tetra=construct_tetra(*data.nkpx)
  #tetra_noneq=[[data.vkpt_fbz[m][3] for m in t ] for t in tetra]
  data.ENE_v=data.ENE
  data.weights_v=data.k_weights
  ne=50
  de=(np.max(data.ENE_v)-np.min(data.ENE_v))/ne
  e=np.linspace(np.min(data.ENE_v),np.max(data.ENE_v),ne)
  h=open('dos.dat','w')
  for i in e:
    print(i)
    h.write(str(i)+' ')
    h.write(str(np.sum([calc_dos(tetra, np.ones(data.ENE.shape),data.ENE-i,iband ) for iband in range(data.no_of_bands)]) ))
    h.write('\n')
  h.close()
  
  #self.ENE=self.ENE_v
