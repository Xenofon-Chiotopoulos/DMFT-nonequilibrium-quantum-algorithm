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
U[0,0] = 0
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

h0,hi =  func.Hed(U,e_imp)
h = h0 + hi
hjw = jordan_wigner(h)
H_atomic =  create_quantum_operator_from_openfermion_text(f"{hjw}") 

h0_imp,hi_imp =  func.Hed(U,e)
h_imp = h0_imp + hi_imp 
hjw_imp = jordan_wigner(h_imp)
H_imp =  create_quantum_operator_from_openfermion_text(f"{hjw_imp}") 

e,v = eigh(get_sparse_operator(hjw).todense())
egs = e[0]

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
'''
plt.rcParams["font.size"] = 18
plt.figure(figsize=(14,8))
plt.plot((cost_history), color="red", label="VQE")
#plt.plot(range(len(cost_history)), [molecule.fci_energy]*len(cost_history), linestyle="dashed", color="black", label="Exact Solution")
plt.xlabel("Iteration")
plt.ylabel("Error in measurment")
#plt.yscale('log')
plt.title('Error in the ground state per iteration')
plt.legend()
'''

state_X = func.get_state(update_theta, norb)

def time_evolution_imp(sx,dt,nt) :

    qct = QuantumCircuit(nq)
    for j in range(H_imp.get_term_count()):
        p = Observable(nq)
        p.add_operator(H_imp.get_term(j))
        qct.add_observable_rotation_gate(p,-2*dt,1)

    electron_num = []
    wsx = sx.copy() # e^(itH_imp)X|GS>
    for i in range(nt):
        p = Observable(nq)
        p.add_operator(1,"Z 0")
        #print(p.get_expectation_value(wsx))
        electron_num.append(p.get_expectation_value(wsx))
        qct.update_quantum_state(wsx)
    return electron_num

electron_num = time_evolution_imp(state_X, 1e-2, 1000)
electron_num_1 = time_evolution_imp(state_X, 1e-2, 5000)
electron_num_iter = [i for i in range(len(electron_num))]
electron_num_iter_1 = [i for i in range(len(electron_num_1))]

plt.figure(figsize=(18,12))
plt.subplot(211)
plt.plot(electron_num_iter,electron_num)

plt.subplot(212)
plt.plot(electron_num_iter_1,electron_num_1)
plt.show()


'''
m_imp = 0.5(1+sigma_z)

We use 
H = H_atomic + H_imp
m_imp = c_up^t c_up + c_down^t c_down

do we take c = sigma_x + i sigma_y or take c(t) = e^-iHt c e^iHt 

<m_imp> = <phi(t)|m_imp|phi(t)>
|phi(t)> = e^iH_2t|Gs> where Gs is of H_atomic

'''