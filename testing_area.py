import numpy as np
from qulacs.gate import to_matrix_gate, X

index = 0
x_gate = X(index)
x_mat_gate = to_matrix_gate(x_gate)

# Operate only when 1st-qubit is 0
control_index = 1
control_with_value = 0
x_mat_gate.add_control_qubit(control_index, control_with_value)
print(x_mat_gate)

from qulacs import QuantumState
state = QuantumState(3)
#state.load(np.arange(2**3))
print(state.get_vector())

x_mat_gate.update_quantum_state(state)
print(state.get_vector())

state.set_Haar_random_state(1)
print(state.get_vector())
