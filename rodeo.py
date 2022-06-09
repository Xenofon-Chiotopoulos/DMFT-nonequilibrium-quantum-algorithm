from cirq import QuantumState
from pyparsing import quoted_string, quotedString
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
import function_file as func
from qulacs import gate

norb = 2
nq = 2*norb 
U = np.zeros((norb,norb))
U[0,0] = 8 
e = np.zeros((norb,norb))
e[0,0] = -U[0,0]/2. 
eb = np.array([-i for i in range(norb-1)])
v = np.array([2*(i+1) for i  in range(norb-1)])
v = v/np.sum(v**2)**0.5
for i in range(1,norb) :
    e[i,i] = eb[i-1] 
    e[i,0] = v[i-1] 
    e[0,i] = v[i-1]
h0,hi =  func.Hed(U,e)
h = h0 + hi
hjw = jordan_wigner(h)
#print(hjw) 
H =  quantum_operator.create_quantum_operator_from_openfermion_text(f"{hjw}") 
e,v = eigh(get_sparse_operator(hjw).todense())
egs = e[0]
print(egs)

def create_gaussian_values(standard_deviation, cycles):
    rand_list = abs(np.random.normal(0.0, standard_deviation, cycles)).tolist()
    return rand_list

def time_evo_rodeo(H, time, resolution, nq, qs):

    qc = QuantumCircuit(nq)
    for j in range(H.get_term_count()):
        p = Observable(nq)
        p.add_operator(H.get_term(j))
        #print(p.get_expectation_value(qs))
        qc.add_observable_rotation_gate(p,-2*resolution,1)
    for k in range(int(time/resolution)):
        qc.update_quantum_state(qs)
    return qs

def rodeo(H, times, Energy, nq, resolution= 0.1):
    qs_save = []
    result = []
    qs = QuantumState(nq+1)
    qc = QuantumCircuit(nq+1)
    for i in range(len(times)):
        
        qc.add_H_gate(0)
        if(times[i] > 0.1):
            reps = int(times[i]/0.1)
            for trot_reps in range(reps):
                for j in range(H.get_term_count()):

                    term = H.get_term(j)
                    pauli_index = term.get_index_list()
                    id_index = term.get_pauli_id_list()
                    ceof = term.get_coef()

                    if pauli_index == [] or id_index == []:
                        continue
                    if ceof.imag != 0.0:
                        raise ValueError('Pauli gates coefficient is imaginary cannot preform real time evolution')
                        
                    test_gate_ = gate.PauliRotation( pauli_index, id_index,-2*resolution*ceof.real)
                    test_gate = gate.to_matrix_gate(test_gate_)
                    test_gate.add_control_qubit(nq,0)
                    qc.add_gate(test_gate)
                for k in range(int(reps/resolution)):
                    qc.update_quantum_state(qs)

        else:
            for j in range(H.get_term_count()):

                term = H.get_term(j)
                pauli_index = term.get_index_list()
                id_index = term.get_pauli_id_list()
                ceof = term.get_coef()

                if pauli_index == [] or id_index == []:
                    continue
                if ceof.imag != 0.0:
                    raise ValueError('Pauli gates coefficient is imaginary cannot preform real time evolution')
                    
                test_gate_ = gate.PauliRotation( pauli_index, id_index,-2*resolution*ceof.real)
                test_gate = gate.to_matrix_gate(test_gate_)
                test_gate.add_control_qubit(nq,0)
                qc.add_gate(test_gate)
            for k in range(int(times[i]/resolution)):
                qc.update_quantum_state(qs)
            
        qc.add_U1_gate(0, Energy * times[i])
        qc.add_H_gate(0)
        qc.update_quantum_state(qs)
        qs_save.append(qs)

    return qs_save

def test_energy_range(min, max, times, spacing = 0.1, resolution = 0.1):
    qs_list = []
    Energy = np.linspace(min,max,0.1)
    for i in range(len(Energy)):
        qs_save, result = rodeo( H, times, Energy, nq, resolution) 
        qs_list.append(qs_save)  
    return qs_list

def initialize_random_state(seed, nq):
    state = QuantumState(nq)
    state.set_Haar_random_state(seed)
    return state

test = create_gaussian_values(1,1)
test_list = rodeo( H, [0.5], -4.82842712474619, nq, 0.5)
print(test_list[0].get_vector)


#Test area to add control to pauli rotation 
'''
for j in range(H.get_term_count()):
            p = Observable(nq)
            print(H.get_term(j))
            p.add_operator(H.get_term(j))
            qc.add_observable_rotation_gate(p,-2*resolution,1)

qs = QuantumState(2)
qc = QuantumCircuit(2)
X_gate = gate.PauliRotation([0,1],[2,3],-2*0.5)
#X_gate = gate.Pauli([0,1],[2,3])
#X_gate = gate.X(0)
X_mat_gate = gate.to_matrix_gate(X_gate)
#X_mat_gate.add_control_qubit(1,0)
print(qs.get_vector())

for j in range(H.get_term_count()):
    p = Observable(nq)
    p.add_operator(H.get_term(j))
    qc.add_observable_rotation_gate(p,-2*0.5,1)


qc.add_gate(X_mat_gate)
X_mat_gate.update_quantum_state(qs)
print(qs.get_vector())

qs.set_zero_state()
qc.add_multi_Pauli_rotation_gate([0,1],[2,3],-2*0.5)
qc.update_quantum_state(qs)
print(qs.get_vector())

#####
index = 0
x_gate = X(index)
x_mat_gate = to_matrix_gate(x_gate)
X_mat_gate.add_control_qubit(1,0)

qc = QuantumCircuit(2)
qc.add_X_gate(0)
qc.add_X_gate(1)
qs = QuantumState(2)
qs.set_zero_state()
qc.update_quantum_state(qs)
print(qs.get_vector())


qs = QuantumState(3)
qc = QuantumCircuit(3)
X_gate = gate.PauliRotation([0,1],[2,3],-2*0.5)
X_mat_gate = gate.to_matrix_gate(X_gate)
X_mat_gate.add_control_qubit(1,0)
print(qs.get_vector())

qc.add_gate(X_mat_gate)
X_mat_gate.update_quantum_state(qs)
print(qs.get_vector())

qc.add_multi_Pauli_rotation_gate([0,1],[2,3],-2*0.5)
qc.update_quantum_state(qs)
print(qs.get_vector())

'''