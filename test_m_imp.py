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
eb = np.array([1,-1]) # 1 , -1 2 , -2
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

def  vqe(d_theta, norb):
    circ = cost(d_theta)
    cost_history = []
    cost_history.append(cost(d_theta))
    method = "BFGS"
    options = {"disp": True, "maxiter": 50, "gtol": 1e-6}
    opt = minimize(cost, d_theta, method=method, callback=lambda x: cost_history.append(cost(x)))
    update_theta = opt.x
    state_X = func.get_state(update_theta, norb)
    return update_theta, cost_history, state_X

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
        p.add_operator(1,"Z 3")
        #print(p.get_expectation_value(wsx))
        electron_num.append(p.get_expectation_value(wsx))
        qct.update_quantum_state(wsx)
    return electron_num

def time_evolution_imp_up(sx,dt,nt) :

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

def time_evolution_imp_down(sx,dt,nt) :

    qct = QuantumCircuit(nq)
    for j in range(H_imp.get_term_count()):
        p = Observable(nq)
        p.add_operator(H_imp.get_term(j))
        qct.add_observable_rotation_gate(p,-2*dt,1)

    electron_num = []
    wsx = sx.copy() # e^(itH_imp)X|GS>
    for i in range(nt):
        p = Observable(nq)
        p.add_operator(1,"Z 3")
        #print(p.get_expectation_value(wsx))
        electron_num.append(p.get_expectation_value(wsx))
        qct.update_quantum_state(wsx)
    return electron_num

d_theta = np.random.random(nq+ norb * (nq+nq-2))*1e-1
d_theta_ = np.random.random(nq+ norb * (nq+nq-2))*1e-1

update_theta, cost_history, state_X = vqe(d_theta, norb)
update_theta_, cost_history_, state_X_ = vqe(d_theta_, norb)

electron_num = time_evolution_imp(state_X, 1e-2, 2000)
electron_num_plus_one = [(1 + i/2) for i in electron_num]

up = time_evolution_imp_up(state_X, 1e-2, 2000)
down = time_evolution_imp_down(state_X, 1e-2, 2000)

N_up_down = up + down
electron_up_down = [(1 + i/2) for i in electron_num]

electron_num_iter = [i for i in range(len(electron_num))]

fig, axs = plt.subplots(2, sharex=False, sharey=False)
fig.suptitle('Testing different time scales')
axs[0].plot(electron_num_iter,electron_num_plus_one, linestyle="dashdot", color="blue")
axs[1].plot(electron_num_iter,electron_up_down, linestyle="dashdot", color="blue")
plt.show()