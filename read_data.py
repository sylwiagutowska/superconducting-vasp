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

 def write_frmsf(self,nkpx,e,ENE,KVEC,DATA=[],name='data'):
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
  print(self.nkpx)
  print(self.ef,'\n')
  #print('fan',(vh5['results/electron_phonon/electrons/self_energy_1/selfen_dfan'])[()].shape)
  vh5.close()

 def read_phelel_params(self):
  vh5 = h5py.File('phelel_params.hdf5')  #read file
  vh5.visititems(self.print_attrs) #list all attributes
  print(vh5['/Dij'][()].shape)
  print('size of supercell',vh5['/phonon_supercell_matrix'][:])

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



 def read_vaspelph(self):
  vh5 = h5py.File('vaspelph.h5')  #read file
  vh5.visititems(self.print_attrs) #list all attributes
  self.igrpop = vh5['/kpoints/igrpop'][:] #symmetry operations (integers)
  a=vh5['results/positions/lattice_vectors'][:] #direct lattice vectors
  data.e=np.linalg.inv(a).transpose() #reciprocal lattice vectors
  #subtract one for the conversion from fortran to c indexes
  self.indx_fbz2ibz = vh5['/kpoints/indx_fbz2ibz'][:]-1 #index of fbz point's equivalent in ibz
  self.irot_fbz2ibz = vh5['/kpoints/irot_fbz2ibz'][:]-1 #symmetry operations leading from ibz to fbz point
  self.vkpt_ibz = vh5['/kpoints/vkpt_ibz'][:] #ibz points
  self.nkfbz = len(self.irot_fbz2ibz) #number of fbz points
  self.generate_fullbz() 

  self.ENE=vh5['/matrix_elements/eigenvalues'][0,:,:].transpose() -self.ef #electron energies
  fermi_nbnd_el=([nm for nm,m in enumerate(self.ENE) if any(m>-0.2) and any(m<0.2)]) #choose only bands which cross/are near fermi level
  self.no_of_bands=len(fermi_nbnd_el)
  self.ENE=self.ENE[fermi_nbnd_el,:]
  self.freq=vh5['matrix_elements/phonon_eigenvalues'][()]/(2*np.pi)/eV_to_THz
  print(np.max(self.freq)*eV_to_THz)
  self.no_of_modes=self.freq.shape[-1]
  #write_frmsf(nkpx,e,ENE,vkpt_fbz)
  print('electron-phonon read...')
  #read electron-phonon matrix elements
  elph_reim=[]
  fermi_beg=fermi_nbnd_el[0]
  fermi_stop=fermi_nbnd_el[-1]+1
  elph_reim = vh5['/matrix_elements/elph'][0,:,:,:,fermi_beg:fermi_stop,fermi_beg:fermi_stop] #[k' fullbzkp][k ibzkp][mode][jband][iband]
  #for i in range(int(len(self.vkpt_fbz)/250)):
  # print(i)
  # elph_reim.append(vh5['/matrix_elements/elph'][0,i*250:(i+1)*(250),:,:,fermi_beg:fermi_stop,fermi_beg:fermi_stop])
  vh5.close() 
#  elph_reim=elph_reim[:,:,:,:,fermi_nbnd_el]
  self.elph = elph_reim[:,:,:,:,:,0] +1j*elph_reim[:,:,:,:,:,1]
  #self.elph=self.elph.transpose((0,1,2,4,3))
  print(np.max(np.absolute(self.elph)))
  print (self.elph.shape)

  #rearrange kpoints to ordered
  self.elph= self.elph[self.unordered_k_to_ordered,:,:,:,:]
  self.freq=self.freq[self.unordered_k_to_ordered,:,:]
  print('elph read finished')



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


 def read_outcar(self):
  h=open('OUTCAR')
  for i in h.readlines():
   if 'Fermi energy on the dense k-point grid' in i: 
    ef=float(i.split()[7])
    break
  h.close()
  self.ENE=self.ENE+self.ef-ef
  self.ef=ef
