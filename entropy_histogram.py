import pennylane as qml
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # If i don't use this it crashed in my PC idk why, not really necessary
import matplotlib.pyplot as plt


dev = qml.device('default.qubit', wires = 2)


@qml.qnode(dev)

def random_state_entropy():
    # Dimension for 2 qubits is 2^2 = 4

    dim = 4

    # Haar random state

    real_part = np.random.normal(size = dim)
    imag_part = np.random.normal(size = dim)
    state_vector = real_part + 1j * imag_part


    # Normalize

    state_vector = state_vector / np.linalg.norm(state_vector)


    # Prepare in the circuit

    qml.StatePrep(state_vector, wires = [0, 1])


    # Measure and return von Neumann entropy of first qubit (wire 0)
    # Measure entropy in bits

    return qml.vn_entropy(wires = [0], log_base = 2)


num_samples = 5000

entropies = [random_state_entropy() for _ in range(num_samples)]


plt.figure(figsize = (8, 5))
plt.hist(entropies, bins=50, density=True, color='teal', edgecolor='black', alpha=0.7)
plt.title(f"Entanglement Entropy Histogram ({num_samples} Random 2-Qubit States)")
plt.xlabel("Von Neumann Entropy (bits)")
plt.ylabel("Probability Density")
plt.grid(axis='y', alpha=0.5)
plt.show()