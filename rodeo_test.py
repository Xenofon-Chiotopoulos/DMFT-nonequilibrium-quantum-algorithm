from ctypes import sizeof
from cirq import QuantumState, quantum_state
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

norb = 3
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

def create_gaussian_values(standard_deviation, cycles):
    rand_list = abs(np.random.normal(0.0, standard_deviation, cycles)).tolist()
    return rand_list


times = create_gaussian_values(0.1,10)
print(times)
qs_list = []
prob_list = []
Energy = np.linspace(-10,10,5000)
qs = QuantumState(nq+1)
for k in range(len(Energy)):
    probability_list = []
    qc = QuantumCircuit(nq+1)
    qc.add_P0_gate(nq)
    for i in range(len(times)): 
        qc.add_H_gate(nq)
        if(times[i] > 0.1):
            reps = np.arange(0,times[i],0.1)
            for trot_reps in reps:
                for j in range(H.get_term_count()):
                    term = H.get_term(j)
                    pauli_index = term.get_index_list()
                    id_index = term.get_pauli_id_list()
                    ceof = term.get_coef()

                    if pauli_index == [] or id_index == []:
                        continue
                    if ceof.imag != 0.0:
                        raise ValueError('Pauli gates coefficient is imaginary cannot preform real time evolution')
                        
                    test_gate_ = gate.PauliRotation( pauli_index, id_index,-2*0.1*ceof.real)
                    test_gate = gate.to_matrix_gate(test_gate_)
                    test_gate.add_control_qubit(nq,0)
                    qc.add_gate(test_gate)
                for k in range(int(trot_reps/0.1)):
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
                    
                test_gate_ = gate.PauliRotation( pauli_index, id_index,-2*0.1*ceof.real)
                test_gate = gate.to_matrix_gate(test_gate_)
                test_gate.add_control_qubit(nq,0)
                qc.add_gate(test_gate)
            for k in range(int(times[i]/0.1)):
                qc.update_quantum_state(qs)
        
        qc.add_U1_gate(nq, Energy[k] * times[i])
        qc.add_H_gate(nq)
        qc.update_quantum_state(qs)
        probability_list.append(qs.get_zero_probability(nq)) 
        print(qs.get_zero_probability(nq))

    qs_list.append(qs)
    prob_list.append(probability_list) 

W = np.linspace(-10,10,5000)

plt.plot(W,prob_list)
plt.show()