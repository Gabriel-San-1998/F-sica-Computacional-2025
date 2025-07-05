import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Parâmetros iniciais
N = 9  # número de ondas planas (número de Gs)
G_vals = np.arange(-N//2, N//2 + 1) * 2 * np.pi  # vetores de rede recíproca
k_vals = np.linspace(-np.pi, np.pi, 300)  # 1ª zona de Brillouin
q = 1  # frequência do cosseno
A = 1  # amplitude do potencial

# Potencial periódico V(x) = A cos(2π q x)
def V_cos(x, A=1, q=1):
    return A * np.cos(2 * np.pi * q * x)

# Matriz de potencial Vmn via transformada de Fourier discreta
def compute_Vmn(V_func, G_vals):
    N_G = len(G_vals)
    Vmn = np.zeros((N_G, N_G), dtype=complex)

    for m in range(N_G):
        for n in range(N_G):
            integrand = lambda x: V_func(x) * np.exp(-1j * 2 * np.pi * (G_vals[m] - G_vals[n]) * x)
            Vmn[m, n], _ = quad(integrand, 0, 1)  # integração no intervalo [0, 1]
    return Vmn

# Construir matriz Hamiltoniana H(k) = T + V
def build_Hk(k, G_vals, Vmn):
    N_G = len(G_vals)
    T = np.diag(0.5 * (k + G_vals)**2)  # energia cinética
    return T + Vmn

# Calcula bandas
def compute_bands(k_vals, G_vals, Vmn):
    bands = []
    for k in k_vals:
        Hk = build_Hk(k, G_vals, Vmn)
        eigvals = np.linalg.eigvalsh(Hk)
        bands.append(eigvals)
    return np.array(bands)

# Calcular matriz Vmn e imprimir
Vmn_cos = compute_Vmn(V_cos, G_vals)
Vmn_cos_real = np.real_if_close(Vmn_cos)

# Calcular bandas
bands_cos = compute_bands(k_vals, G_vals, Vmn_cos_real)

# Plot do potencial
x_plot = np.linspace(0, 1, 500)
plt.figure(figsize=(6, 3))
plt.plot(x_plot, V_cos(x_plot, A, q))
plt.title('Potencial V(x) = A cos(2πqx)')
plt.xlabel('x')
plt.ylabel('V(x)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot das bandas
plt.figure(figsize=(8, 6))
for i in range(len(G_vals)):
    plt.plot(k_vals, bands_cos[:, i], label=f'Banda {i+1}')
plt.title('Bandas de energia com V(x) = A cos(2πqx)')
plt.xlabel('k')
plt.ylabel('Energia')
plt.grid(True)
plt.tight_layout()
plt.show()

Vmn_cos_real.round(3)  # exibir a matriz Vmn com 3 casas decimais


