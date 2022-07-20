from aim import *
from qulacs.circuit import QuantumCircuitOptimizer
from hf import *
import sys
sys.path.append(".")
import h5py
from ctypes import sizeof
from cirq import QuantumState, quantum_state
from qiskit import QuantumCircuit
from qulacs import *
from openfermion.ops import FermionOperator
from openfermion import *
import numpy as np
from qulacs import ParametricQuantumCircuit
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import scipy as sci
from scipy.optimize import minimize
import math
from qulacs.gate import *
import function_file as func

def Hed(U, eimp, inv=False):
    """
    contruct H from parameters read in param_3
    """
    norb = eimp.shape[0]
    N = norb
    nimp = U.shape[0]
    from itertools import product

    hi, h0 = 0, 0
    if inv:
        for i, j in product(range(nimp), range(nimp)):
            if i != j:
                hi += FermionOperator(f"{i}^  {j}^  {j} {i} ", U[i, j] / 2.0)
                hi += FermionOperator(f"{i+N}^  {j+N}^  {j+N} {i+N} ", U[i, j] / 2.0)
                hi += FermionOperator(f"{i}^  {N+j}^  {N+j} {i} ", U[i, j])
            else:
                hi += FermionOperator(f"{i}^  {N+j}^  {N+j} {i} ", U[i, j])
        for i, j in product(range(norb), range(norb)):
            h0 += FermionOperator(f"{i}^ {j} ", eimp[i, j])
            h0 += FermionOperator(f"{i+N}^ {j+N} ", eimp[i, j])
    else:
        for i, j in product(range(nimp), range(nimp)):

            if i != j:
                hi += FermionOperator(
                    f"{N-1-i}^  {N-1-j}^  {N-1-j} {N-1-i} ", U[i, j] / 2.0
                )
                hi += FermionOperator(f"{i+N}^  {j+N}^  {j+N} {i+N} ", U[i, j] / 2.0)
                hi += FermionOperator(f"{N-1-i}^  {N+j}^  {N+j} {N-1-i} ", U[i, j])
            else:
                hi += FermionOperator(f"{N-1-i}^  {N+j}^  {N+j} {N-1-i} ", U[i, j])
        for i, j in product(range(norb), range(norb)):
            h0 += FermionOperator(f"{N-1-i}^ {N-1-j} ", eimp[i, j])
            h0 += FermionOperator(f"{i+N}^ {j+N} ", eimp[i, j])
    return h0, hi

#norb = 2
#qubit number
# nq = 2*norb
# U = np.zeros((norb,norb))
# U[0,0] = 8
# e = np.zeros((norb,norb))
# e[0,0] = -U[0,0]/2.
# eb = np.array([-i for i in range(norb-1)])
# v = np.array([2*(i+1) for i  in range(norb-1)])
# v = v/np.sum(v**2)**0.5
# for i in range(1,norb) :
#     e[i,i] = eb[i-1]
#     e[i,0] = v[i-1]
#     e[0,i] = v[i-1]
# h0,hi =  func.Hed(U,e)
# h = h0 + hi
# hjw = jordan_wigner(h)
# #print(hjw)<
# H =  quantum_operator.create_quantum_operator_from_openfermion_text(f"{hjw}")

f = h5py.File("aim12.h5", "r")
e = np.array(f["e"])
et = np.array(f["et"])
U = np.array(f["U"])
f.close()
h0, hi = Hed(U, et,inv=True)
norb = e.shape[0]
nq = 2*norb
anderson_impurity = AIM(e, et, U)
H = h0 + hi
hjw = jordan_wigner(h0+hi)
e,v = eigh(get_sparse_operator(hjw).todense())
egs = e[0]
print(e[:10])

nq = anderson_impurity.nq
norb = anderson_impurity.norb
noc1=3
noc2=3
loc_up = [i for i in range(noc1)]
loc_down = [i for i in range(noc2)]

q0 = anderson_impurity.ground_state_h0(loc_up, loc_down)

H =  quantum_operator.create_quantum_operator_from_openfermion_text(f"{hjw}")
H2 =  quantum_operator.create_quantum_operator_from_openfermion_text(f"{hjw}"+f"\n(1e-8+0j) [Z{nq}]")
cst = float(f"{hjw}".split()[0].split('+')[0].replace('(','').replace('[','').replace('\'',''))

#VQE 
#-----------------------------------------------------

def cost(d_theta):
    state = QuantumState(nq) #Prepare |00000>
    circuit = func.min_ansatz(nq, norb, d_theta) #Construct quantum circuit
    circuit.update_quantum_state(state) #Operate quantum circuit on state
    return H.get_expectation_value(state).real #Calculate expectation value of Hamiltonian

def get_state(d_theta,norb):
    nq = 2 * norb
    state = QuantumState(nq+1) #Prepare |00000>
    circuit = func.min_ansatz_1(nq, norb, d_theta) #Construct quantum circuit
    circuit.update_quantum_state(state) #Operate quantum circuit on state
    return state #Output the current state 

dt = np.random.random(nq+ norb * (nq+nq-2))*1e-1
circ = cost(dt)
cost_history = []
d_theta = np.random.random(nq+ norb * (nq+nq-2))*1e-2
cost_history.append(cost(d_theta))
method = "BFGS"
options = {"disp": True, "maxiter": 50, "gtol": 1e-6}
opt = minimize(cost, d_theta, method=method, callback=lambda x: cost_history.append(cost(x)))
update_theta = opt.x
q0 = get_state(update_theta, norb)
print(cost_history)

#print(q1.get_qubit_count())
#-----------------------------------------------------

def ctrotter(dt) :
    qct = QuantumCircuit(nq+1)
    for j in range(H.get_term_count()):
        term = H.get_term(j)
        pauli_index = term.get_index_list()
        id_index = term.get_pauli_id_list()
        ceof = term.get_coef()
        if pauli_index == [] or id_index == []:
            pauli_index == [0 for i in range(nq)]
            id_index == [0 for i in range(nq)]

#            continue

        if ceof.imag != 0.0:
            raise ValueError('Pauli gates coefficient is imaginary cannot preform real time evolution')

        test_gate_ = gate.PauliRotation( pauli_index, id_index,-2*dt*ceof.real)

        test_gate = gate.to_matrix_gate(test_gate_)
        test_gate.add_control_qubit(nq,1)
        qct.add_gate(test_gate)
    return qct

#Setting random state
#q0 = QuantumState(nq+1)
#q0.set_Haar_random_state()

#Function that creates the gaussian random values
def create_gaussian_values(standard_deviation, cycles):
    rand_list = (np.random.normal(0.0, standard_deviation, cycles)).tolist()
    return rand_list

#variables that determine the number of energies to test
spacing = 0.001
min = -2.3
max = -2.2

#setting the maximum values a trotter evolution is allowed to use
resolution = 0.01

times = create_gaussian_values(10,30)
#times = [0.1*i for i in range(-30,30)]#create_gaussian_values(1,1)

exact_value = []
qs_list = []
prob_list = []
Energy = np.arange(min,max,spacing) #range of energies to testd

for k in range(len(Energy)):

    probability_list = []
    qs = QuantumState(nq+1)
#    qs.set_Haar_random_state() # random state preperation
    # X(1).update_quantum_state(qs)
    # X(0).update_quantum_state(qs)
    # X(norb).update_quantum_state(qs)
    qs= q0.copy()
#    X(norb+1).update_quantum_state(qs)

    qc = QuantumCircuit(nq+1)

    P0(nq).update_quantum_state(qs) # projection of ancilla qubit to the 0 state
    qs.normalize(qs.get_squared_norm())

    prob=1

    for i in range(len(times)):

        nstep = int(abs(times[i])/(resolution))+1
        dt = times[i]/nstep

        qct = ctrotter(dt)
        gate.H(nq).update_quantum_state(qs)
        for j in range(nstep) :
            qct.update_quantum_state(qs)
        U1(nq, Energy[k] * times[i]).update_quantum_state(qs)
        gate.H(nq).update_quantum_state(qs)

        prob = prob*(qs.get_zero_probability(nq))
        P0(nq).update_quantum_state(qs)
        qs.normalize(qs.get_squared_norm())
    print(Energy[k], H2.get_expectation_value(qs).real,prob)
    prob_list.append(prob)
    exact_value.append(H2.get_expectation_value(qs).real)

max1 = prob_list[0]    

count = 0

for i in range(0, len(prob_list)):  
    #Compare elements of array with max    
    if(prob_list[i] > max1): 
        count += 1     
        max1 = prob_list[i]; 
print('These are the largest values')
print(Energy[count],exact_value[count],max1)

plt.plot(Energy,prob_list)
plt.show()