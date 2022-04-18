from sympy import N
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
norb = 3
nq = 2*norb
U = np.zeros((norb,norb))
U[0,0] = 8
e = np.zeros((norb,norb))
e[0,0] = -U[0,0]/2
eb = np.array([-i for i in range(norb-1)])
v = np.array([2*(i+1) for i  in range(norb-1)])
v = v/np.sum(v**2)**0.5

for i in range(1,norb) :
    e[i,i] = eb[i-1] #energy of bath i
    e[i,0] = v[i-1] #hopping amplitude between impurity and site i
    e[0,i] = v[i-1]

h0,hi =  func.Hed(U,e)
h = h0 + hi
hjw = jordan_wigner(h)
H =  create_quantum_operator_from_openfermion_text(f"{hjw}") 
e,v = eigh(get_sparse_operator(hjw).todense())
egs = e[0]
print(egs)
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
n_lanczos=20

Ge1=0
for ic,c in enumerate([cp,cm]):
    u = (c.dot(gs)).T
    norm = u.T.dot(u)[0,0]**0.5
    print(norm)
    u = u/norm
    print((-1)**ic)
    a,b=func.lanc(n_lanczos,u,(-1)**ic*h)
    Ge1+=norm**2*func.f(W,a,b**2)

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
opt = minimize(cost, d_theta, method=method, callback=lambda x: cost_history.append(cost(x)))
update_theta = opt.x

'''
plt.rcParams["font.size"] = 18
plt.figure(figsize=(14,8))
plt.plot((cost_history-egs), color="red", label="VQE")
plt.xlabel("Iteration")
plt.ylabel("Error in measurment")
plt.yscale('log')
plt.title('Error in the ground state per iteration')
plt.legend()
plt.show()
'''
state_X = func.get_state(update_theta, norb)
X(0).update_quantum_state(state_X) 
state_Y = func.get_state(update_theta, norb)
Y(0).update_quantum_state(state_Y)

Gwp, Gwh, T = func.Gwp_Gwh_creation(state_X, state_Y, cost_history, nq, H, 1e-2,5000)

plt.figure(figsize=(18,12))

plt.subplot(211)
plt.plot(W.real,-Ge1.imag,'b--')
plt.xlabel('Frequency')
plt.ylabel('Imaginary value of G(w)')
plt.title('Comparing Classical and Quantum algorithms to find the DOS')

plt.subplot(212)
plt.xlabel('Frequency')
plt.ylabel('Imaginary value of G(w)')
plt.plot(W.real,(Gwp+Gwh).imag,'k--')

plt.show()