
import xml.etree.ElementTree as ET
import numpy as np
from operator import itemgetter
from matplotlib import pyplot as plt
import read_data
import calc_elph 
import copy
import eliashberg
import eliashberg_aniso
 
ELECTRONMASS_SI  = 9.10938215e-31   # Kg
AMU_SI           = 1.660538782e-27   #Kg
AMU_AU           = AMU_SI / ELECTRONMASS_SI
AMU_RY           = AMU_AU / 2. #=911.44 
RY_TO_THZ=3289.8449


def round_complex(y):
 prec=9
 try:
  for i in range(len(y)):
   for j in range(len(y[i])):
    y[i][j]=round(y[i][j].real,prec)+1j*round(y[i][j].imag,prec)
 except:
  try:
   for i in range(len(y)):
    y[i]=round(y[i].real,prec)+1j*round(y[i].imag,prec)
  except:
    y=round(y.real,prec)+1j*round(y.imag,prec)
 return y

data=read_data.data()
#print('phelel_params')
#data.read_phelel_params()
print('\nvaspout')
data.read_vaspout()
print('\nvaspelph')

data.read_vaspelph()
data.read_outcar()
data.calc_kkp_to_q()



#data.k_to_q_conversion() #dunno if needed check later
print(data)
#data.read_vasprun()
data.write_frmsf(data.nkpx,data.e,data.ENE,[],'ene')

#exit()
#plt.hist(np.abs(data.elph.flatten())**2)
#plt.yscale('log')
#plt.show()

elph=calc_elph.lambd(data)
elph.init_tetra(data)
#elph.calc_dos(data)
#elph.calc_lambda(data)
#elph.calc_a2f(data)
#elph.calc_a2f_smearing(data)
'''

eliashberg=eliashberg.eliashberg_solver()
#eliashberg.calc_bandwidth(data)
#eliashberg.A2F=data.A2F
#eliashberg.OM_a2f=data.OM_a2f
eliashberg.read_a2f()
eliashberg.calc_mu()
eliashberg.solve_eq()

'''
eliashberg_aniso=eliashberg_aniso.eliashberg_aniso_solver(data)

eliashberg_aniso.solve_eq(data)
