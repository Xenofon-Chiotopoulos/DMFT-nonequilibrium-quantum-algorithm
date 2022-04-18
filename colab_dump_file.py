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
eta=0.05
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
from scipy.optimize import minimize
import math

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

e[0,0] = -U[0,0]/2. # Double counting, we can talk about that next time



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
#print(hjw)
#trotterization = trotter_operator_grouping(hjw)
H =  create_quantum_operator_from_openfermion_text(f"{hjw}") # create a  qulacs hamiltonian

e,v = eigh(get_sparse_operator(hjw).todense())
egs = e[0]
print(egs)

def circuit_ansatz(nq,norb, d_theta):
    "define circuit ansatz here"
    circuit =  QuantumCircuit(nq)

    count = 0
    for i in range(nq):
        circuit.add_RY_gate(i,d_theta[count])
        count += 1
    for i in range(norb):
        for j in range(nq):
            if j % 2 == 0:
                circuit.add_CNOT_gate(j,j + 1)
            circuit.add_RY_gate(j, d_theta[count])
            count += 1
        for j in range(1,nq-1):
            if j % 2 == 1:
                circuit.add_CNOT_gate(j,j+1)
            circuit.add_RY_gate(j, d_theta[count])
            count += 1
    return circuit

def cost(d_theta):
    state = QuantumState(nq) #Prepare |00000>
    circuit = circuit_ansatz(nq, norb, d_theta) #Construct quantum circuit
    circuit.update_quantum_state(state) #Operate quantum circuit on state
    return H.get_expectation_value(state).real #Calculate expectation value of Hamiltonian

def get_state(d_theta):
    state = QuantumState(nq) #Prepare |00000>
    circuit = circuit_ansatz(nq, norb, d_theta) #Construct quantum circuit
    circuit.update_quantum_state(state) #Operate quantum circuit on state
    return state #Output the current state 

dt = np.random.random(nq+ norb * (nq+nq-2))*1e-1
circ = cost(dt)

cost_history = []
d_theta = np.random.random(nq+ norb * (nq+nq-2))*1e-2
cost_history.append(cost(d_theta))
#print(cost_history)
method = "BFGS"
options = {"disp": True, "maxiter": 50, "gtol": 1e-6}
opt = minimize(cost, d_theta,    method=method,           callback=lambda x: cost_history.append(cost(x)))
update_theta = opt.x

plt.rcParams["font.size"] = 18
plt.figure(figsize=(14,8))
plt.plot((cost_history-egs), color="red", label="VQE")
#plt.plot(range(len(cost_history)), [molecule.fci_energy]*len(cost_history), linestyle="dashed", color="black", label="Exact Solution")
plt.xlabel("Iteration")
plt.ylabel("Error in measurment")
plt.yscale('log')
plt.title('Error in the ground state per iteration')
plt.legend()
plt.show()

def trotter_step(dt, nq, H, state, repeat=1):
    #qs = QuantumState(nq)
    qs =  state.copy()
    qc = QuantumCircuit(nq)
    for j in range(H.get_term_count()) :
        p = Observable(nq)
        p.add_operator(H.get_term(j))
        qc.add_observable_rotation_gate(p,-2*dt,repeat)
    qc.update_quantum_state(qs) # apply the quantum circuit qc to the state qs
    return qs

state_X = get_state(update_theta)
X(0).update_quantum_state(state_X) 
state_Y = get_state(update_theta)
Y(0).update_quantum_state(state_Y)

def time_evolutionf(sx,sy,dt,nt) :
    """
    sx = X|GS>
    sy = Y|GS>

    """

    qct = QuantumCircuit(nq)
    for j in range(H.get_term_count()):
        p = Observable(nq)
        p.add_operator(H.get_term(j))
        qct.add_observable_rotation_gate(p,-2*dt,1) # add_observable_rotation_gate = e^{ iHt/2.}
    # qct = e^
    sxxt = np.zeros(nt,dtype = complex)
    syyt = np.zeros(nt,dtype = complex)
    sxyt = np.zeros(nt,dtype = complex)
    syxt = np.zeros(nt,dtype = complex)

    #I think this for loop changes the quantum circuit every time depending on dt

    wsx = sx.copy() # e^(itH)X|GS>
    wsy = sy.copy() # e^(itH)X|GS>
    for i in range(nt):
        sxxt[i] = inner_product(sx,wsx)
        syxt[i] = inner_product(sy,wsx)
        sxyt[i] = inner_product(sx,wsy)
        syyt[i] = inner_product(sy,wsy)
        
        qct.update_quantum_state(wsx)
        qct.update_quantum_state(wsy)
    gh = -1j/4. * (sxxt + syyt - 1j*(sxyt - syxt))
    gp = -1j/4. * (sxxt + syyt + 1j*(sxyt - syxt))
    return gh,gp

def time_evolution_log(sx,sy,dt,nt) :
    """
    sx = X|GS>
    sy = Y|GS>

    """

    gh = 0
    gp = 0

   
    # qct = e^
    sxxt = np.zeros(nt,dtype = complex)
    syyt = np.zeros(nt,dtype = complex)
    sxyt = np.zeros(nt,dtype = complex)
    syxt = np.zeros(nt,dtype = complex)

    wsx = sx.copy() # e^(itH)X|GS>
    wsy = sy.copy() # e^(itH)X|GS>
    for i in range(nt):
        sxxt[i] = inner_product(sx,wsx)
        syxt[i] = inner_product(sy,wsx)
        sxyt[i] = inner_product(sx,wsy)
        syyt[i] = inner_product(sy,wsy)
        qct = QuantumCircuit(nq)
        for j in range(H.get_term_count()):
            p = Observable(nq)
            p.add_operator(H.get_term(j))
            qct.add_observable_rotation_gate(p,-2*dt[i],1)

        qct.update_quantum_state(wsx)
        qct.update_quantum_state(wsy)
    gh = -1j/4. * (sxxt + syyt - 1j*(sxyt - syxt))
    gp = -1j/4. * (sxxt + syyt + 1j*(sxyt - syxt))

    return gh,gp

def Gwp_Gwh_creation(dt = 0.5,nt = 20000):
    T = np.array([dt*i for i in range(nt)]) # Here, we may use a logaritmic grid
    t = T
    timestep = 1/nt
    for j in range(1, len(T)):
        t[j] = (dt)*np.exp(np.log(dt*nt/(dt))*(j/len(T)))
    T = t  # uncomment for log

    for i in range(1,len(t)-1):
        t[i] = t[i+1]-t[i]
    t[0] = t[1]
    t[-1] = t[-2]

    Gp, Gh = time_evolution_log(state_X,state_Y,t,nt) #uncommet for log
    #Gp, Gh = time_evolutionf(state_X,state_Y,dt,nt) #original time evo function

    T = np.array([dt*i for i in range(nt)])
    for j in range(1, len(T)):
        T[j] = (dt)*np.exp(np.log(dt*nt/(dt))*(j/len(T)))

    Gp = Gp*np.exp(1j*cost_history[-1]*T) # e^{i egs t}
    Gh = Gh*np.exp(1j*cost_history[-1]*T)
    eta=0.05
    W = np.linspace(-10,10,10000)
    Gwp = np.zeros(W.shape[0],dtype=complex)
    Gwh = np.zeros(W.shape[0],dtype=complex)
    for i in range(W.shape[0]) :
        Gwp[i] = np.sum(np.exp(1j*(W[i].real)*T[:-1] -eta*T[:-1])*(T[:-1]-T[1:])*Gp[:-1])
        Gwh[i] = np.sum(np.exp(-1j*(W[i].real)*T[:-1] -eta*T[:-1])*(T[:-1]-T[1:])*Gh[:-1])
    return Gwp, Gwh, T


#Good fidelity appears at 5000 iterations
#
dt_test = [5000] # Change this to change the number of total iterations
Gwp_list1 = []
Gwh_list1 = []
for i in dt_test:
  Gwp, Gwh, T = Gwp_Gwh_creation(1e-2,i)
  Gwp_list1.append(Gwp)
  Gwh_list1.append(Gwh)
list = [i for i in range(len(T))]
#plt.plot( list, T)

plt.figure(figsize=(18,12))
plt.subplot(211)
#plt.plot(W.real,-Ge.imag,'k--', label='U = 1')
plt.plot(W.real,-Ge1.imag,'b--')
#plt.legend(prop={'size': 14})
plt.xlabel('Frequency')
plt.ylabel('Imaginary value of G(w)')
plt.title('Comparing Classical and Quantum algorithms to find the DOS')
plt.subplot(212)
plt.xlabel('Frequency')
plt.ylabel('Imaginary value of G(w)')
plt.plot(W.real,((Gwp_list1[0]+Gwh_list1[0])).imag,'k--')
#plt.plot(W.real,((Gwp_list1[1]+Gwh_list1[1])).imag,'b--',label='U = 8')
#plt.legend(prop={'size': 14})
plt.show()