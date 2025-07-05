import numpy as np
import matplotlib.pyplot as plt

def build_hamiltonian(k, N, alpha, method="loop"):
    """
    Hamiltoniano tridiagonal para o cristal 1D.
      method="loop" → mesmo algoritmo do primeiro código
      method="diag" → construção compacta com np.diag (estilo imagem)
    """
    dim = 2 * N + 1
    G = 2.0 * np.pi * np.arange(-N, N + 1)          # vetores de rede recíproca
    kin = (k + G) ** 2                               # termo cinético (diagonal)

    if method == "loop":
        H = np.zeros((dim, dim))
        np.fill_diagonal(H, kin) 
        for i in range(dim - 1):
            H[i, i + 1] = H[i + 1, i] = alpha
    elif method == "diag":
        off = np.full(dim - 1, alpha)
        H = (np.diag(kin) +
             np.diag(off,  k=1) +
             np.diag(off,  k=-1))
    else:
        raise ValueError("method deve ser 'loop' ou 'diag'")
    return H

def compute_bands(N=3, alpha=1.0, num_k=400, num_bands=4, method="loop"):
    k_vals = np.linspace(-np.pi, np.pi, num_k, endpoint=True)
    bands  = np.empty((num_bands, num_k))

    for j, k in enumerate(k_vals):
        H = build_hamiltonian(k, N, alpha, method=method)
        eigvals = np.linalg.eigvalsh(H)
        bands[:, j] = eigvals[:num_bands]            # pegar as mais baixas
    return k_vals, bands

# -------- parâmetros ajustáveis --------
N          = 7    # número de vetores G de cada lado
alpha      = 70  # intensidade do potencial periódico
num_k      = 400    # resolução em k
num_bands  = 7     # quantas bandas mostrar
method     = "diag" # "loop" ou "diag"
# ---------------------------------------

k_vals, bands = compute_bands(N, alpha, num_k, num_bands, method)

# ------- gráfico -------
plt.figure(figsize=(6, 4))
for n in range(num_bands):
    plt.plot(k_vals, bands[n])
plt.xlim(-np.pi, np.pi)
plt.xlabel(r"$k$")
plt.ylabel("Energia")
plt.ylim(-5, 300)
plt.title(f"Bandas 1D  (method = {method}, α = {alpha}, N = {N})")
plt.axvline(-np.pi, ls="--", lw=0.6, color='gray')
plt.axvline( np.pi, ls="--", lw=0.6, color='gray')
plt.grid(True, ls=":", lw=0.4)
plt.tight_layout()

plt.savefig('9.4f.png')
plt.show()
#Adicionar mais diagonais
