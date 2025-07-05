import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def V_gaussian_single(x, A=1.0, gamma=0.05):
    """Single Gaussian centred at x=0."""
    return -A * np.exp(- (x**2) / (2 * gamma**2))

def V_gaussian_double(x, A=1.0, B=1.0, gamma=0.05):
    """Two Gaussians centred at x=1/4 and x=-1/4 (equiv. 3/4)."""
    return (-A * np.exp(- ((x - 0.25)**2) / (2 * gamma**2))
            -B * np.exp(- ((x + 0.25)**2) / (2 * gamma**2)))

def compute_Vmn(V_func, G_vals):
    """Numerical integration of V(x) for all (m,n) pairs."""
    N_G = len(G_vals)
    Vmn = np.zeros((N_G, N_G), dtype=complex)
    for m in range(N_G):
        for n in range(N_G):
            phase = lambda x: V_func(x) * np.exp(-1j * (G_vals[m] - G_vals[n]) * x)
            Vmn[m, n], _ = quad(phase, 0.0, 1.0, limit=200)
    # Force Hermitian (small numerical noise)
    Vmn = 0.5 * (Vmn + Vmn.conj().T)
    return np.real_if_close(Vmn)

def build_Hk(k, G_vals, Vmn):
    kinetic = 0.5 * (k + G_vals)**2
    return np.diag(kinetic) + Vmn

def compute_bands(k_vals, G_vals, Vmn, num_bands):
    bands = []
    for k in k_vals:
        Hk = build_Hk(k, G_vals, Vmn)
        eigvals = np.linalg.eigvalsh(Hk)
        bands.append(eigvals[:num_bands])
    return np.array(bands)

# ------------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------------
N = 5                           # G indices from -5 to 5  → matrix 11×11
G_vals = 2 * np.pi * np.arange(-N, N + 1)
k_vals = np.linspace(-np.pi, np.pi, 400)
num_bands = 8                   # display lowest bands
gamma = 0.05                    # Gaussian width
A = 1.0                         # amplitude
B = 1.0

# ------------------------------------------------------------------
# 1) Single Gaussian potential -------------------------------------
# ------------------------------------------------------------------
Vmn_single = compute_Vmn(lambda x: V_gaussian_single(x, A, gamma), G_vals)
bands_single = compute_bands(k_vals, G_vals, Vmn_single, num_bands)

# Plot potential
x_plot = np.linspace(0, 1, 500)
plt.figure(figsize=(5, 3))
plt.plot(x_plot, V_gaussian_single(x_plot, A, gamma))
plt.title("Single Gaussian potential")
plt.xlabel("x")
plt.ylabel("V(x)")
plt.grid(True)
plt.tight_layout()
plt.savefig('potential_single_gaussian.png')
plt.close()

# Plot bands
plt.figure(figsize=(6, 4))
for n in range(num_bands):
    plt.plot(k_vals, bands_single[:, n])
plt.title("Bands – Single Gaussian")
plt.xlabel("k")
plt.ylabel("Energy")
plt.xlim(-np.pi, np.pi)
plt.tight_layout()
plt.savefig('bands_single_gaussian.png')
plt.close()

# ------------------------------------------------------------------
# 2) Double Gaussian potential -------------------------------------
# ------------------------------------------------------------------
Vmn_double = compute_Vmn(lambda x: V_gaussian_double(x, A, B, gamma), G_vals)
bands_double = compute_bands(k_vals, G_vals, Vmn_double, num_bands)

# Plot potential
plt.figure(figsize=(5, 3))
plt.plot(x_plot, V_gaussian_double(x_plot, A, B, gamma))
plt.title("Double Gaussian potential")
plt.xlabel("x")
plt.ylabel("V(x)")
plt.grid(True)
plt.tight_layout()
plt.savefig('potential_double_gaussian.png')
plt.close()

# Plot bands
plt.figure(figsize=(6, 4))
for n in range(num_bands):
    plt.plot(k_vals, bands_double[:, n])
plt.title("Bands – Double Gaussian")
plt.xlabel("k")
plt.ylabel("Energy")
plt.xlim(-np.pi, np.pi)
plt.tight_layout()
plt.savefig('bands_double_gaussian.png')
plt.close()

# Return matrix values for inspection (rounded)
import pandas as pd
print("Matriz Vmn para o potencial gaussiano simples:")
print(pd.DataFrame(np.round(Vmn_single, 4)))

"Plots saved: potential_single_gaussian.png, bands_single_gaussian.png, potential_double_gaussian.png, bands_double_gaussian.png"

