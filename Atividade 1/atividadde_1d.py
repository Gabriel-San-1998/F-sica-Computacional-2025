import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Função de duas variáveis
def U(x, y):
    return np.sin(x) * np.cos(y) + 2 * (x * y)**2 / 1000

# Derivadas parciais de primeira ordem
def dU_dx(x, y):
    return x * y**2 / 250 + np.cos(x)*np.cos(y)

def dU_dy(x, y):
    return y * x**2 / 250 - np.sin(x) * np.sin(y)

# Derivadas parciais de segunda ordem
def d2U_dx2(x, y):
    return  y**2 / 250 - np.sin(x) * np.cos(x)

def d2U_dy2(x, y):
    return  x**2 / 250 - np.sin(x) * np.cos(y)

# Parâmetros
step = 0.1
tolerancia = 1e-5
max_iter = 10000

# Lista de chutes iniciais [(x0, y0), ...]
chutes_iniciais = [(-4, 2), (2, 3), (-2, 1), (4, 0), (-4, -3), (2, -3)]

# Grava caminhos e mínimos
minimos = []
trajetos_x = []
trajetos_y = []

# Descida do gradiente para cada chute
for x0, y0 in chutes_iniciais:
    x, y = x0, y0
    caminho_x = [x]
    caminho_y = [y]

    for _ in range(max_iter):
        grad_x = dU_dx(x, y)
        grad_y = dU_dy(x, y)

        if abs(grad_x) < tolerancia and abs(grad_y) < tolerancia:
            break

        # Gradiente descendente
        x -= step * grad_x
        y -= step * grad_y

        caminho_x.append(x)
        caminho_y.append(y)

    minimos.append((x, y, U(x, y)))
    trajetos_x.append(caminho_x)
    trajetos_y.append(caminho_y)

# ---------- BUSCA EM GRADE PARA O MÍNIMO GLOBAL ----------
x_vals = np.linspace(-6, 6, 500)
y_vals = np.linspace(-6, 6, 500)
X, Y = np.meshgrid(x_vals, y_vals)
Z = U(X, Y)

# Localizar índice do mínimo global
min_index = np.unravel_index(np.argmin(Z), Z.shape)
x_min_global = X[min_index]
y_min_global = Y[min_index]
u_min_global = Z[min_index]

# ---------- PLOT ----------
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, 100, cmap='coolwarm')
plt.colorbar(label='U(x, y)')

# Trajetórias dos gradientes
for i in range(6):
    plt.plot(trajetos_x[i], trajetos_y[i], marker='o', markersize=3, label=f"Trajeto {i+1}")

# Mínimos locais encontrados por gradiente
for i, (x_min, y_min, u_min) in enumerate(minimos):
    plt.plot(x_min, y_min, 'ko')  # ponto final
    print(f"Mínimo {i+1}: x = {x_min:.4f}, y = {y_min:.4f}, U = {u_min:.4f}")
# Comparar os mínimos locais com o mínimo global
print("\n📊 Comparação entre mínimos locais e o mínimo global:\n")
for i, (x_min, y_min, u_min) in enumerate(minimos):
    dx = abs(x_min - x_min_global)
    dy = abs(y_min - y_min_global)
    du = abs(u_min - u_min_global)

    print(f"Comparando com mínimo {i+1}:")
    print(f"  Δx = {dx:.6f}")
    print(f"  Δy = {dy:.6f}")
    print(f"  ΔU = {du:.6e}")
    if dx < 1e-3 and dy < 1e-3:
        print(" Coincide com o mínimo global!")
    print()


# Mínimo global encontrado por busca em grade
plt.plot(x_min_global, y_min_global, 'o', color='purple', markersize=10, label='Mínimo Global')
print(f" MÍNIMO GLOBAL: x = {x_min_global:.4f}, y = {y_min_global:.4f}, U = {u_min_global:.4f}")

plt.title("Mapa de calor com trajetos e mínimo global")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.savefig(r'/home/gabas/Área de trabalho/F-sica-Computacional-2025/Atividade 1/1atv1d.png')
plt.show()
