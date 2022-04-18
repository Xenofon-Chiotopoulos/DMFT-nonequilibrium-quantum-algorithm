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

eta=0.05
W = np.linspace(-10,10,10000)+1j*eta

norb = 3 # nb of orbital (impurity + bath)

nq = 2*norb # nb of qubits, 2* norb due to spin

U = np.zeros((norb,norb))

U[0,0] = 8 # We  put a U just on the first orbital which is the impurity
#I belive this is in Ev

e = np.zeros((norb,norb))

e[0,0] = -U[0,0]/2. # Double counting, we can talk about that next time

#set off diagonal part of e to 0

eb = np.array([-1,1,2])#[-i for i in range(norb-1)])
v = np.array([2*(i+1) for i  in range(norb-1)])
v = v/np.sum(v**2)**0.5


for i in range(1,norb) :
    e[i,i] = eb[i-1] #energy of bath i
    e[i,0] = v[i-1] #hopping amplitude between impurity and site i
    e[0,i] = v[i-1]

h0,hi =  func.Hed(U,e)
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

def cost(d_theta):
    state = QuantumState(nq) #Prepare |00000>
    circuit = func.circuit_ansatz(nq, norb, d_theta) #Construct quantum circuit
    circuit.update_quantum_state(state) #Operate quantum circuit on state
    return H.get_expectation_value(state).real #Calculate expectation value of Hamiltonian

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


state_X = func.get_state(update_theta, norb)
X(0).update_quantum_state(state_X) 
state_Y = func.get_state(update_theta, norb)
Y(0).update_quantum_state(state_Y)

#Good fidelity appears at 5000 iterations
#
dt_test = [10000] # Change this to change the number of total iterations
Gwp_list1 = []
Gwh_list1 = []
for i in dt_test:
  Gwp, Gwh, T = func.Gwp_Gwh_creation(state_X, state_Y, cost_history, nq, H, 1e-2,i)
  Gwp_list1.append(Gwp)
  Gwh_list1.append(Gwh)
list = [i for i in range(len(T))]
#plt.plot( list, T)

plt.figure(figsize=(12,8))
#plt.subplot(211)
'''
plt.plot(W.real,-Ge.imag,'k--', label='U = 1')
plt.plot(W.real,-Ge1.imag,'b--')
plt.legend(prop={'size': 14})
plt.xlabel('Frequency')
plt.ylabel('Imaginary value of G(w)')
plt.title('Comparing Classical and Quantum algorithms to find the DOS')
'''
#plt.subplot(212)
plt.xlabel('Frequency')
plt.ylabel('Imaginary value of G(w)')
plt.plot(W.real,((Gwp_list1[0]+Gwh_list1[0])).imag,'k--')
#plt.plot(W.real,((Gwp_list1[1]+Gwh_list1[1])).imag,'b--',label='U = 8')
#plt.legend(prop={'size': 14})
plt.show()