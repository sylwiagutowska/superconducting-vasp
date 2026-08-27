import h5py
import numpy as np
from operator import itemgetter
from multiprocessing import Process,Pool
import xml.etree.ElementTree as ET

eV_to_THz=241.7991

class data():
 def __init__(self):
  self.ENE=[]
  self.elph=[]
  self.nkpx=[]

 def print_attrs(self,name, obj):
    print (name)
    for key, val in obj.attrs.items():
        print( "    %s: %s" % (key, val))

 def write_frmsf_noneq(self,nkpx,e,ENE,KVEC,DATA=[],name='data'):
  h=open(name+'.frmsf','w')
  h.write(str(nkpx[0])+' '+str(nkpx[1])+' ' +str(nkpx[2])+'\n')
  h.write('1\n'+str(len(ENE))+'\n')
  for i in e:
   for j in i:
    h.write(str(j)+' ')
   h.write('\n')
  for bnd in ENE:
   for k in KVEC:
    h.write(str(bnd[k[3]])+'\n')
  if len(DATA)!=0:
   if len(DATA[0])==nkpx[0]*nkpx[1]*nkpx[2]:
    for bnd in DATA:
     for k in bnd:
      h.write(str(k)+'\n')
   else:
    for bnd in DATA:
     for nk,k in enumerate(KVEC):
      h.write(str(bnd[k[3]])+'\n')
   h.close()

 def write_frmsf(self,nkpx,e,ENE,DATA=[],name='data'):
  h=open(name+'.frmsf','w')
  h.write(str(nkpx[0])+' '+str(nkpx[1])+' ' +str(nkpx[2])+'\n')
  h.write('1\n'+str(len(ENE))+'\n')
  for i in e:
   for j in i:
    h.write(str(j)+' ')
   h.write('\n')
  for bnd in ENE:
   for k in bnd:
    h.write(str(k)+'\n')
  if len(DATA)!=0:
   if len(DATA[0])==nkpx[0]*nkpx[1]*nkpx[2]:
    for bnd in DATA:
     for k in bnd:
      h.write(str(k)+'\n')
   else:
    for bnd in DATA:
     for nk,k in enumerate(KVEC):
      h.write(str(bnd[k[3]])+'\n')
   h.close()


 def sorting(self,allk2):
  Xall=[]
  allk2=sorted(allk2, key=itemgetter(0))
  i=0
  while i<len(allk2): 
   X=[]
   x=allk2[i]
   while i<len(allk2) and x[0]==allk2[i][0]:
    X.append(allk2[i])
    i=i+1
   if len(X)>1: X=sorted(X, key=itemgetter(1))
   Xall.append(X)
  Yall=[]
  for X in Xall:
   x=X[0]
   i=0
   while i<len(X): 
    Y=[]
    x=X[i]
    while i<len(X) and x[1]==X[i][1]:
     Y.append(X[i])
     i=i+1
    Y=sorted(Y, key=itemgetter(2))
    Yall.append(Y)
  allk=[]
  for i in Yall:
   for j in i:
    allk.append(j)
  #  print ' Sorting - Done!'
  return allk

 def single_job(self,q,step,data,fermi_beg,fermi_stop):
  print(q)
  return data['/matrix_elements/elph'][0,q*step:(q+1)*step,:,:,fermi_beg:fermi_stop,fermi_beg:fermi_stop]

 def parallel_job(self,npool,step,data,fermi_beg,fermi_stop):
  with Pool(npool) as pol:
   results=pol.map(self.single_job,
              [[q,step,data,fermi_beg,fermi_stop] for q in range(npool)])
   
 def read_vaspout(self):
  vh5 = h5py.File('vaspout.h5')  #read file
  vh5.visititems(self.print_attrs) #list all attributes
  self.ef=(vh5['/results/electron_dos/efermi'][()] )
  self.k_weights=vh5['results/electron_phonon/electrons/eigenvalues/kpoints_symmetry_weight'][:]
  self.nkpx=[vh5['input/kpoints_dense/nkp'+m][()] for m in ['x','y','z']]  #no of fbz points in each direction
  try: 
   self.A2F=vh5['results/electron_phonon/phonons/self_energy_1/a2F'][:] #/4
   self.OM_a2f=vh5['results/electron_phonon/phonons/self_energy_1/frequency_grid'][:]/2/np.pi
   self.A2F=self.A2F[0,:]
   print(self.OM_a2f.shape,self.A2F.shape)
   self.A2F=self.A2F[self.OM_a2f>0]
   self.OM_a2f=self.OM_a2f[self.OM_a2f>0]
   h=open('a2f-vasp.dat','w')
   for ni,i in enumerate(self.OM_a2f):
    h.write(str(i)+' '+str(self.A2F[ni])+'\n')
   h.close()

  except: 
   self.A2F,self.OM_a2f=[],[]
  print(self.nkpx)
  print(self.ef,'\n')
  #print('fan',(vh5['results/electron_phonon/electrons/self_energy_1/selfen_dfan'])[()].shape)
  vh5.close()

 def read_phelel_params(self):
  vh5 = h5py.File('phelel_params.hdf5')  #read file
  vh5.visititems(self.print_attrs) #list all attributes
  print(vh5['/Dij'][()].shape)
  print('size of supercell',vh5['/phonon_supercell_matrix'][:])

 '''
 def generate_fullbz(self):
  self.vkpt_fbz = []
  for ikfbz in range(self.nkfbz):
    ikibz = self.indx_fbz2ibz[ikfbz]
    irot  = self.irot_fbz2ibz[ikfbz]
    k=list(np.round(np.matmul(self.igrpop[irot,:,:],self.vkpt_ibz[ikibz,:]),6))
    for i in range(3):
      while k[i]<0: k[i]+=1
      while k[i]>=1: k[i]-=1
    self.vkpt_fbz.append(list(np.round(k,6))+[self.indx_fbz2ibz[ikfbz]]+[ikfbz])
  self.vkpt_fbz = (self.sorting(self.vkpt_fbz )) #fbz points + its equivalent's index in ibz
  ordered_k_to_unordered=[k[4] for k in self.vkpt_fbz]
  self.unordered_k_to_ordered=[99999 for k in self.vkpt_fbz]
  for ni,i in enumerate(ordered_k_to_unordered): self.unordered_k_to_ordered[i]=ni
  print(self.nkpx)

 def k_to_q_conversion_fullbz(self):
#  ibz_to_fbz=[[] for i in self.vkpt_ibz]
#  for ni,i in enumerate(self.vkpt_fbz):
#   ibz_to_fbz[i[3]].append(ni)
  only_vec=[np.array([int(i*self.nkpx[ni]) for ni,i in enumerate(k[:3])]) for k in self.vkpt_fbz]
  self.kkp_to_q=[[ [] for j in self.vkpt_ibz] for i in (self.vkpt_fbz)] #np.zeros((len(self.vkpt_fbz),len(self.vkpt_fbz)),dtype=int)+100000
  self.freq_q=np.zeros((len(self.vkpt_fbz),(self.no_of_modes)))
#  for nk,k in enumerate(self.vkpt_fbz): self.vkpt_fbz[nk]=[np.array([int(i*self.nkpx[ni]) for ni,i in enumerate(k[:3])])]+k[3:]
  #for k in self.vkpt_ibz: k[:3]=np.array([int(i*self.nkpx[ni]) for ni,i in enumerate(k[:3])])
  for nkp,kp in enumerate(only_vec):
   for nk,k in enumerate(only_vec):
    q=kp-k
    for i in range(3):
      while q[i]<0: q[i]+=self.nkpx[i]
      while q[i]>=self.nkpx[i]: q[i]-=self.nkpx[i]
    nq=int(q[2]+self.nkpx[2]*q[1]+self.nkpx[2]*self.nkpx[1]*q[0])
    self.kkp_to_q[nkp][self.vkpt_fbz[nk][3]].append(nq)
    self.freq_q[nq,:]=self.freq[nkp,self.vkpt_fbz[nk][3],:]
  print(self.freq_q)
  print(self.kkp_to_q)
  
 def k_to_q_conversion(self):
#  ibz_to_fbz=[[] for i in self.vkpt_ibz]
#  for ni,i in enumerate(self.vkpt_fbz):
#   ibz_to_fbz[i[3]].append(ni)

  self.kkp_to_q=np.zeros((len(self.vkpt_fbz),len(self.vkpt_ibz)),dtype=int)+100000
  self.freq_q=np.zeros((len(self.vkpt_fbz),(self.no_of_modes)))
  for k in self.vkpt_fbz: k[:3]=np.array([int(i*self.nkpx[ni]) for ni,i in enumerate(k[:3])])
  for k in self.vkpt_ibz: k[:3]=np.array([int(i*self.nkpx[ni]) for ni,i in enumerate(k[:3])])
  for nk,k in enumerate(self.vkpt_fbz):
   for nkp,kp in enumerate(self.vkpt_ibz):
    q=kp[:3]-k[:3]
    for i in range(3):
      while q[i]<0: q[i]+=self.nkpx[i]
      while q[i]>=self.nkpx[i]: q[i]-=self.nkpx[i]
    nq=int(q[2]+self.nkpx[2]*q[1]+self.nkpx[2]*self.nkpx[1]*q[0])
    self.kkp_to_q[nk][nkp]=nq
    self.freq_q[nq,:]=self.freq[nk,nkp,:]
  print(self.freq_q)
  print(self.kkp_to_q)
 '''

 def calc_kkp_to_q(self):
    kp_int = np.rint(np.asarray(self.vkpt_kp) * self.nkpx).astype(int)
    k_int = np.rint(np.asarray(self.vkpt_k) * self.nkpx).astype(int)

    # Shape: (nkp, nk, 3)
    all_kkp = (kp_int[:, None, :] - k_int[None, :, :]) % self.nkpx

    # Unique kkp vectors
    all_kkp2 = all_kkp.reshape(-1, 3)
    self.vkpt_q, inverse = np.unique(all_kkp2, axis=0, return_inverse=True)

    # inverse gives the q-index for each kkp
    self.kkp_to_q = inverse.reshape(all_kkp.shape[:2])
    self.vkpt_q = np.round(self.vkpt_q / self.nkpx, 6)

    freq = np.asarray(self.freq)
    nq = len(self.vkpt_q)
    nmodes = freq.shape[-1]
 

    freq_flat = np.asarray(self.freq).reshape(-1, nmodes)
    first = np.zeros(nq, dtype=int)
    _, first = np.unique(inverse, return_index=True)
    self.freq_q = freq_flat[first]

    print("qpoints")
    print(self.vkpt_q.shape)
    print(self.vkpt_q)

 

 def read_vaspelph(self):
  vh5 = h5py.File('vaspelph.h5')  #read file
  vh5.visititems(self.print_attrs) #list all attributes
  self.igrpop = vh5['/kpoints/igrpop'][:] #symmetry operations (integers)
  a=vh5['results/positions/lattice_vectors'][:] #direct lattice vectors
  self.e=np.linalg.inv(a).transpose() #reciprocal lattice vectors
  #subtract one for the conversion from fortran to c indexes
  self.indx_fbz2ibz = vh5['/kpoints/indx_fbz2ibz'][:]-1 #index of fbz point's equivalent in ibz
  self.irot_fbz2ibz = vh5['/kpoints/irot_fbz2ibz'][:]-1 #symmetry operations leading from ibz to fbz point
  #self.vkpt_ibz = vh5['/kpoints/vkpt_ibz'][:] #ibz points
  self.vkpt_k =  np.round(vh5['kpoints/vkpt_k'][:],9)
  self.vkpt_kp =  np.round(vh5['kpoints/vkpt_kp'][:],9)
  kbands = {"start": vh5['matrix_elements/band_start_k'][()] - 1, "number": vh5['matrix_elements/nbands_k'][()]}
  kpbands = {"start": vh5['matrix_elements/band_start_kp'][()] - 1, "number": vh5['matrix_elements/nbands_kp'][()]}
  
  nkpt = self.nkpx[0]*self.nkpx[1]*self.nkpx[2]

 
#     print(i) #,  np.linalg.solve(self.e, i))
  print(f"bands in k-space: start: {kbands["start"]} number: {kbands["number"]}")
  print(f"bands in kp-space: start: {kpbands["start"]} number: {kpbands["number"]}")
  print(self.vkpt_kp.shape)
  self.nkfbz = len(self.irot_fbz2ibz) #number of fbz points
  #self.generate_fullbz() 

  self.ENE=vh5['matrix_elements/eigenvalues_k'][0,:,:].transpose() -self.ef #electron energies
  print('initial shape of energies in k-space (nband, nkp)', self.ENE.shape)
  fermi_nbnd_el=([nm for nm,m in enumerate(self.ENE) if any(m>-0.2) and any(m<0.2)]) #choose only bands which cross/are near fermi level
  if (len(fermi_nbnd_el) > kbands["number"]):
    fermi_nbnd_el = range(kbands["start"], kbands["start"] + kbands["number"])
  self.no_of_bands=len(fermi_nbnd_el)


  self.ENE_kp = vh5['matrix_elements/eigenvalues_kp'][0,:,:].transpose() -self.ef #electron energies
  print('initial shape of energies in kp-space (nband, nkp)', self.ENE_kp.shape)

  fermi_nbnd_el_kp=([nm for nm,m in enumerate(self.ENE_kp) if any(m>-0.2) and any(m<0.2)]) #choose only bands which cross/are near fermi level
  if (len(fermi_nbnd_el_kp) > kpbands["number"]):
    fermi_nbnd_el_kp = range(kpbands["start"], kpbands["start"] + kpbands["number"])
  self.no_of_bands_kp=len(fermi_nbnd_el_kp)

  self.ENE=self.ENE[fermi_nbnd_el,:] #i removed  - kbands["start"], i dont undestand the logic, CHECK IT
  self.ENE_kp=self.ENE_kp[fermi_nbnd_el_kp,:]
  
  print('shape of energies in k-space (nband, nkp)', self.ENE.shape)
  print('shape of energies in kp-space (nband, nkp)', self.ENE_kp.shape)
  print("bands",fermi_nbnd_el,fermi_nbnd_el)

  self.freq=vh5['matrix_elements/phonon_eigenvalues'][()]/(2*np.pi)/eV_to_THz
  print("freqs are in shape", self.freq.shape, "max", np.max(self.freq))
  print(np.max(self.freq)*eV_to_THz)
  self.no_of_modes=self.freq.shape[-1]
  #self.write_frmsf(self.nkpx,self.e,self.ENE,self.vkpt_fbz)
  print('electron-phonon read...')
  #read electron-phonon matrix elements
  elph_reim=[]
  fermi_beg_k =    fermi_nbnd_el[0] - vh5['matrix_elements/band_start_k'][()]
  fermi_stop_k =  fermi_nbnd_el[-1]+1 - vh5['matrix_elements/band_start_k'][()] #fermi_beg_k + vh5['matrix_elements/nbands_k'][()]+1
  fermi_beg_kp =  fermi_nbnd_el_kp[0] - vh5['matrix_elements/band_start_kp'][()]
  fermi_stop_kp =  fermi_nbnd_el_kp[-1]+1 - vh5['matrix_elements/band_start_kp'][()] #fermi_beg_kp + vh5['matrix_elements/nbands_kp'][()]+1
  print(fermi_beg_k, fermi_stop_k)
  #elph_reim = vh5['/matrix_elements/elph'][0,:,:,:,min(fermi_nbnd_el_kp):max(fermi_nbnd_el_kp),:]
  elph_reim = vh5['/matrix_elements/elph'][0,:,:,:,fermi_beg_kp:fermi_stop_kp,fermi_beg_k:fermi_stop_k] #[k' fullbzkp][k fullbzk][mode][jband][iband]
  #for i in range(int(len(self.vkpt_fbz)/250)):
  # print(i)
  # elph_reim.append(vh5['/matrix_elements/elph'][0,i*250:(i+1)*(250),:,:,fermi_beg:fermi_stop,fermi_beg:fermi_stop])
  vh5.close() 
#  elph_reim=elph_reim[:,:,:,:,fermi_nbnd_el]
  self.elph = elph_reim[:,:,:,:,:,0] +1j*elph_reim[:,:,:,:,:,1]
  print ('shape of elph [kp, k, mode, jband, iband]',self.elph.shape)

  #self.elph=self.elph.transpose((0,1,2,4,3))
  print('max elph element',np.max(np.absolute(self.elph)))

  #rearrange kpoints to ordered
  #self.elph= self.elph[self.unordered_k_to_ordered,:,:,:,:]
  #self.freq=self.freq[self.unordered_k_to_ordered,:,:]
  
  self.vkpt_k = np.where(self.vkpt_k >= 0, self.vkpt_k, self.vkpt_k + 1) 
  #self.vkpt_k = self.vkpt_k + np.array([0.5-1/self.nkpx[0],0.5-1/self.nkpx[1],0.5-1/self.nkpx[2]])
  self.vkpt_kp = np.where(self.vkpt_kp >= 0, self.vkpt_kp, self.vkpt_kp + 1) 
  #self.vkpt_kp = self.vkpt_kp + np.array([0.5-1/self.nkpx[0],0.5-1/self.nkpx[1],0.5-1/self.nkpx[2]])
  to_ordered_k_idx=[]
  for x in range(self.nkpx[0]):
     for y in range(self.nkpx[1]):
      for z in range(self.nkpx[2]):
        k_new = np.round([x/self.nkpx[0], y/self.nkpx[1], z/self.nkpx[2]], 9)
        for nk, k in enumerate(self.vkpt_k):
        # k2 = [ m if m>=0 else m+1 for m in k]
         if np.allclose(k_new, k, atol=1e-6):
            to_ordered_k_idx.append(nk)
            continue
  to_ordered_kp_idx=[]
  for x in range(self.nkpx[0]):
     for y in range(self.nkpx[1]):
      for z in range(self.nkpx[2]):
        k_new = np.round([x/self.nkpx[0], y/self.nkpx[1], z/self.nkpx[2]], 9)
        for nk, k in enumerate(self.vkpt_kp):
         if np.allclose(k_new, k, atol=1e-6):
            to_ordered_kp_idx.append(nk)
            continue
  print("ordered k ", len(to_ordered_k_idx))
  print("ordered kp", len(to_ordered_kp_idx))

  self.ENE = self.ENE[:, to_ordered_k_idx]
  self.elph = self.elph[:,to_ordered_k_idx,:,:,:]
  self.freq = self.freq[:, to_ordered_k_idx, :]
  self.vkpt_k = self.vkpt_k[to_ordered_k_idx]
  self.ENE_kp = self.ENE_kp[:, to_ordered_kp_idx]
  self.elph = self.elph[to_ordered_kp_idx,:,:,:,:]
  self.freq = self.freq[to_ordered_kp_idx, :, :]
  self.vkpt_kp = self.vkpt_kp[to_ordered_kp_idx]
  print('elph read finished')


  '''
 def read_vasprun(self):
  tree = ET.parse('dense_ene/vasprun.xml')
  root = tree.getroot()
  self.ENE_v=[]
  a=root.find('calculation/eigenvalues/array/set/set')
  for i in a: 
    if i.tag=='set' and 'kpoint' in i.attrib['comment']: 
      self.ENE_v.append([])
      for j in i.findall('r'):
        self.ENE_v[-1].append(j.text.split()[0])

  a=root.findall('kpoints/generation/v')
  for i in a: 
    if i.attrib['name']=='divisions': nk3=[int(m) for m in i.text.split()]
  print(nk3)
  nkp=nk3[0]*nk3[1]*nk3[2]
  a=root.findall('kpoints/varray')[1].findall('v') 
  self.weights_v=np.array([ m.text.split()[0] for m in a],dtype=float)
  self.nkpx_v=nk3
  ef=float(root.find('calculation/dos/i').text.split()[0])
  self.ENE_v=np.array(self.ENE_v,dtype=float).transpose()-ef
 '''

 def read_outcar(self):
  h=open('OUTCAR')
  for i in h.readlines():
   if 'Fermi energy on the dense k-point grid' in i: 
    ef=float(i.split()[7])
   elif 'BZINTS: Fermi energy:' in i:
    ef=float(i.split()[4][:-1])
    break
  h.close()
  try: 
   self.ENE=self.ENE+self.ef-ef
   self.ef=ef
  except: 0