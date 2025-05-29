import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
T_amb = 25          # Temperatura ambiente (°C)
r = 0.005           # Taxa de resfriamento (1/s)
T0 = 90             # Temperatura inicial do café (°C)
t0 = 0              # Tempo inicial (s)
tf = 5000           # Tempo final (s)
dt = 10             # Passo de tempo (s)

# Função da EDO
def dTdt(t, T):
    return r * (T_amb - T)

# Solução analítica
def T_analitica(t):
    return T_amb - (T_amb - T0) * np.exp(-r * t)

# Método de Runge-Kutta de 4ª ordem
def rk4(f, T0, t0, tf, dt):
    N = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, N)
    T = np.zeros(N)
    T[0] = T0

    for i in range(1, N):
        k1 = f(t[i-1], T[i-1])
        k2 = f(t[i-1] + dt/2, T[i-1] + dt*k1/2)
        k3 = f(t[i-1] + dt/2, T[i-1] + dt*k2/2)
        k4 = f(t[i-1] + dt, T[i-1] + dt*k3)
        T[i] = T[i-1] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t, T

# Executar RK4
tempo, T_rk4 = rk4(dTdt, T0, t0, tf, dt)

# Solução analítica nos mesmos pontos de tempo
T_exata = T_analitica(tempo)

# Plotagem
plt.figure(figsize=(9, 5))
plt.plot(tempo, T_rk4, 'o-', label='RK4 (numérica)', color='brown', markersize=4)
plt.plot(tempo, T_exata, '--', label='Solução analítica', color='blue')
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Comparação: RK4 vs Solução Analítica')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
