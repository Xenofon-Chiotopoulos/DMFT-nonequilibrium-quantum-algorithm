from scipy.optimize import minimize

"""Importing all the necessary packages"""

from qulacs.observable import *
from qulacs.quantum_operator import *
from qulacs import *
from qulacs.gate import *
from openfermion.ops import FermionOperator
from openfermion import *
import numpy as np
from qulacs import ParametricQuantumCircuit
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from qulacs.state import inner_product
import scipy as sci

"""## AIM code

Defining the hamiltonian Hed, which returns the

###Setup
"""

def Hed(U,eimp) :
    """
    H = \sum_i e_{i,j} c^+_i c_j + \sum_ij U_ij n_i n_j
    """

    norb = eimp.shape[0]
    N = norb
    nimp = np.count_nonzero(U) #maybe count number of imputrites, counts number of nonzeros aka impurities
    from itertools import product
    hi,h0=0,0
    #\sum_ij U_ij n_i n_j
    for i,j in product(range(nimp),range(nimp)) :
        if i != j :
            hi += FermionOperator(f'{i}^  {j}^  {j} {i} ', U[i,j]/2.)
            hi += FermionOperator(f'{i+N}^  {j+N}^  {j+N} {i+N} ', U[i,j]/2.)
            hi += FermionOperator(f'{i}^  {N+j}^  {N+j} {i} ', U[i,j])
        else :
            hi += FermionOperator(f'{i}^  {N+j}^  {N+j} {i} ', U[i,j])
    # \sum_i e_{i,j} c^+_i c_j
    for i,j in product(range(norb),range(norb)) :
         h0 += FermionOperator(f'{i}^ {j} ', eimp[i,j])
         h0 += FermionOperator(f'{i+N}^ {j+N} ', eimp[i,j])
    return h0,hi

norb = 3 # nb of orbital (impurity + bath)

nq = 2*norb # nb of qubits, 2* norb due to spin

U = np.zeros((norb,norb))

U[0,0] = 8 # We  put a U just on the first orbital which is the impurity
#I belive this is in Ev

e = np.zeros((norb,norb))

e[0,0] = -U[0,0]/2.  # Double counting, we can talk about that next time


eb = np.array([-i for i in range(norb-1)])
v = np.array([2*(i+1) for i  in range(norb-1)])

v = v/np.sum(v**2)**0.5


for i in range(1,norb) :
    e[i,i] = eb[i-1] #energy of bath i
    e[i,0] = v[i-1] #hopping amplitude between impurity and site i
    e[0,i] = v[i-1]

h0,hi =  Hed(U,e)
"""
By convention defined by myself,

impurity  up : 0th qubits
bath 1 up: 1st qubits
....
bath norb -1 up : norb-1 qubits
impurity  down : norb qubits
bath 1 dowb: 1 norb+1 qubits
....
bath norb  down : 2*norb-1  qubits

example if norb =5:
qubit 0 : imp up
qubit 1 : bath 1 up
qubit 2 : bath 1 up
qubit 3 : bath 2 up
qubit 4 : bath 3 up
qubit 5 : imp    down
qubit 6 : bath 1 down
qubit 7 : bath 1 down
qubit 8 : bath 2 down
qubit 9 : bath 3 down
"""
h = h0 + hi
#print(h)
hjw = jordan_wigner(h) # do the jordan wigner transformation


H =  create_quantum_operator_from_openfermion_text(f"{hjw}") # create a  qulacs hamiltonian

e,v = eigh(get_sparse_operator(hjw).todense())
egs = e[0]
print(egs)



def lanc(niter,u,h) :
    a=np.zeros((niter+1))
    b=np.zeros((niter+1))
    ut=np.zeros((niter+1,u.shape[0]),dtype=complex)
    ut[0]=u.T
    b[0]=0
    for i in range(niter) :
        a[i]=ut[i].T.dot(h).dot(ut[i])[0,0].real
        if i==0:
            b[i+1]=abs((ut[i].T.dot(h**2).dot(ut[i]))[0,0]-a[i]**2)**0.5
        else :
            b[i+1]=abs((ut[i].T.dot(h**2).dot(ut[i]))[0,0]-a[i]**2+b[i]**2-2*b[i]*ut[i].T.dot(h).dot(ut[i-1])[0,0])**0.5
        ut[i+1]=1./b[i+1]*(h.dot(ut[i])-a[i]*ut[i])
        if(i>=1) :
            ut[i+1]-=b[i]*ut[i-1]/b[i+1]
    return a,b[1:]
from scipy.linalg import eigh


def f(w,a,b) :
    if len(a) ==1 :
        return 1./(w-a[0])
    else :
        return 1./(w-a[0]-b[0]*f(w,a[1:],b[1:]))



iorb = 0
cp=get_sparse_operator(((FermionOperator(f'{iorb}^  ',1))),n_qubits=norb*2).todense()
cm=get_sparse_operator(((FermionOperator(f'{iorb}  ',1))),n_qubits=norb*2).todense()
eta=0.1
W = np.linspace(-10,10,10000)+1j*eta
h=get_sparse_operator(h).todense()
e,v=eigh(h)
gs=v[:,0]
egs=e[0]
print(f'EGS={egs}')
h=h-egs*np.identity(h.shape[0])

import scipy

n_lanczos=20

Ge1=0
for ic,c in enumerate([cp,cm]):
    u = (c.dot(gs)).T
    norm = u.T.dot(u)[0,0]**0.5
    print(norm)
    u = u/norm
    print((-1)**ic)
    a,b=lanc(n_lanczos,u,(-1)**ic*h)
    Ge1+=norm**2*f(W,a,b**2)
plt.figure(figsize=(16,8))
plt.subplot(211)
plt.plot(W.real,-Ge1.imag,'k--')
plt.subplot(212)
plt.plot(W.real,Ge1.real,'r--')

plt.show()
#test commit