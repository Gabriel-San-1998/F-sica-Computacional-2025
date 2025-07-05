import numpy as np
import matplotlib.pyplot as plt

def build_hamiltonian(k, G_values, alpha=0.0, method="diag"):
    N = len(G_values)
    kinetic = np.array([(k + G)**2 for G in G_values])
    
    if method == "diag":
        H = np.diag(kinetic)
        if alpha != 0:
            off_diag = np.full(N - 1, alpha)
            H += np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return H
    else:
        raise ValueError("Only 'diag' method implemented for now.")

# Parâmetros do sistema
num_G = 5  # número de ondas planas de cada lado
G_vals = 2 * np.pi * np.arange(-num_G, num_G + 1)  # múltiplos de 2π
k_vals = np.linspace(-np.pi, np.pi, 500)
bands = []

# Calcula as energias para cada k
for k in k_vals:
    H = build_hamiltonian(k, G_vals, alpha=0.0, method="diag")
    eigvals = np.linalg.eigvalsh(H)
    bands.append(np.sort(eigvals))

bands = np.array(bands)

# Plot
plt.figure(figsize=(10, 6))
for i in range(len(G_vals)):
    plt.plot(k_vals, bands[:, i], color="black", linewidth=1)

plt.title("Estrutura de bandas (Rede vazia) - Figura 5.14")
plt.xlabel(r"$k$")
plt.ylabel(r"$\varepsilon(k)$")
plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi],
           [r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"])
plt.grid(True)
plt.axvline(x=-np.pi, linestyle='--', color='gray')
plt.axvline(x=np.pi, linestyle='--', color='gray')
plt.axhline(y=0, linestyle='-', color='black')
plt.xlim(-2*np.pi, 2*np.pi)
plt.ylim(0, 80)
plt.show()

