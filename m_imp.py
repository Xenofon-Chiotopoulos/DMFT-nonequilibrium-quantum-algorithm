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
import function_file as func

norb = 3
nq = 2*norb
U = np.zeros((norb,norb))
U[0,0] = 8
e = np.zeros((norb,norb))
e_imp = np.zeros((norb,norb))
e[0,0] = -U[0,0]/2
e_imp[0,0] = -U[0,0]/2
eb = np.array([-i for i in range(norb-1)])
v = np.array([2*(i+1) for i  in range(norb-1)])
v = v/np.sum(v**2)**0.5

for i in range(1,norb) :
    e[i,i] = eb[i-1] #energy of bath i
    e_imp[i,i] = eb[i-1]
    e[i,0] = v[i-1] #hopping amplitude between impurity and site i
    e[0,i] = v[i-1]


h0,hi =  func.Hed(U,e)
h = h0 + hi
hjw = jordan_wigner(h)
H_atomic =  create_quantum_operator_from_openfermion_text(f"{hjw}") 

h0_imp,hi_imp =  func.Hed(U,e_imp)
h_imp = h0_imp + hi_imp + h
hjw_imp = jordan_wigner(h_imp)
H_imp =  create_quantum_operator_from_openfermion_text(f"{hjw_imp}") 


def cost(d_theta):
    state = func.QuantumState(nq) #Prepare |00000>
    circuit = func.circuit_ansatz(nq, norb, d_theta) #Construct quantum circuit
    circuit.update_quantum_state(state) #Operate quantum circuit on state
    return H_atomic.get_expectation_value(state).real #Calculate expectation value of Hamiltonian

dt = np.random.random(nq+ norb * (nq+nq-2))*1e-1
circ = cost(dt)
cost_history = []
d_theta = np.random.random(nq+ norb * (nq+nq-2))*1e-2
cost_history.append(cost(d_theta))
#print(cost_history)
method = "BFGS"
options = {"disp": True, "maxiter": 50, "gtol": 1e-6}
opt = minimize(cost, d_theta, method=method, callback=lambda x: cost_history.append(cost(x)))
update_theta = opt.x

state_X = func.get_state(update_theta, norb)
X(0).update_quantum_state(state_X) 
state_Y = func.get_state(update_theta, norb)
Y(0).update_quantum_state(state_Y)

def time_evolution_imp(sx,sy,dt,nt) :

    qct = QuantumCircuit(nq)
    for j in range(H_imp.get_term_count()):
        p = Observable(nq)
        p.add_operator(H_imp.get_term(j))
        qct.add_observable_rotation_gate(p,-2*dt,1)

    sxxt = np.zeros(nt,dtype = complex)
    syyt = np.zeros(nt,dtype = complex)
    sxyt = np.zeros(nt,dtype = complex)
    syxt = np.zeros(nt,dtype = complex)

    wsx = sx.copy() # e^(itH_imp)X|GS>
    wsy = sy.copy() # e^(itH_imp)X|GS>
    for i in range(nt):
        sxxt[i] = inner_product(sx,wsx)
        syxt[i] = inner_product(sy,wsx)
        sxyt[i] = inner_product(sx,wsy)
        syyt[i] = inner_product(sy,wsy)
        
        qct.update_quantum_state(wsx)
        qct.update_quantum_state(wsy)
    phi_t = -1j/4. * (sxxt + syyt - 1j*(sxyt - syxt))
    t_phi = -1j/4. * (sxxt + syyt + 1j*(sxyt - syxt))
    return phi_t, t_phi

test_1, test_2 = time_evolution_imp(state_X, state_Y, 1e-2, 1000)
phi_t = np.array(test_1)
t_phi = np.array(test_2).T

print(abs(phi_t*t_phi*0.25))

'''

<m_imp> = <phi(t)|m_imp|phi(t)>
|phi(t)> = e^iH_2t|Gs> where Gs is of H_atomic

'''