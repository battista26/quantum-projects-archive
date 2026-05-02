import pennylane as qml
import numpy as np

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)

def bell_state():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0,1])
    return qml.state()

state = bell_state()
print("Bell state vector:\n", state)

# Full density matrix
rho_full = np.outer(state, np.conj(state))
print("\nFull density matrix:\n", rho_full.round(3))

# Reduced density matrix for qubit 0 using partial trace
# Reshape to (2,2,2,2) and trace over qubit 1
rho_full_reshaped = rho_full.reshape(2,2,2,2)

### CHECK the correctness of indices
rho_A = np.trace(rho_full_reshaped, axis1=1, axis2=3) # sum over indices 1 and 3
print("\nReduced density matrix for qubit 0:\n", rho_A.round(3))
print("Purity of rho_A:", np.trace(rho_A @ rho_A).real)

# Entropy of entanglement

from scipy.linalg import eigvalsh
evals = eigvalsh(rho_A)
evals = np.clip(evals, 0, 1) # remove numerical negatives
entropy = -np.sum(evals * np.log2(evals + 1e-12))
print("Entropy of entanglement:", entropy)

# Non‑maximally entangled state

@qml.qnode(dev)
def non_max_state():
    qml.RY(1.2, wires=0) # create non-uniform amplitudes
    qml.CNOT(wires=[0,1])
    return qml.state()

state2 = non_max_state()

rho_full2 = np.outer(state2, np.conj(state2))
rho_A2 = np.trace(rho_full2.reshape(2,2,2,2), axis1=2, axis2=3)

evals2 = eigvalsh(rho_A2)
evals2 = np.clip(evals2, 0, 1)

entropy2 = -np.sum(evals2 * np.log2(evals2 + 1e-12))
print("Entropy of entanglement for non-max state:", entropy2)