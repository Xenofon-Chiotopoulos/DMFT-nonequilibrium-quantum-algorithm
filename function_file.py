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

def get_state(d_theta,norb):
    nq = 2 * norb
    state = QuantumState(nq) #Prepare |00000>
    circuit = circuit_ansatz(nq, norb, d_theta) #Construct quantum circuit
    circuit.update_quantum_state(state) #Operate quantum circuit on state
    return state #Output the current state 

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

def time_evolutionf(sx,sy,dt,nt,nq,H) :
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

def time_evolution_log(sx,sy,dt,nt,nq,H) :
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

def Gwp_Gwh_creation(state_X, state_Y, cost_history, nq, H, dt = 0.5,nt = 20000):
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

    Gp, Gh = time_evolution_log(state_X,state_Y,t,nt,nq,H) #uncommet for log
    #Gp, Gh = time_evolutionf(state_X,state_Y,dt,nt,nq,H) #original time evo function

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