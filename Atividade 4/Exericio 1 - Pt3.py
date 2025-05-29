import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do modelo
T_amb = 25       # Temperatura ambiente (°C)
r = 0.005        # Taxa de resfriamento (1/s)
T0 = 90          # Temperatura inicial (°C)

# Solução analítica
def T_analitica(t):
    return T_amb - (T_amb - T0) * np.exp(-r * t)

# Gerar 10 pontos no intervalo de 0 a 200 s
np.random.seed(0)  # Para reprodutibilidade
t_dados = np.linspace(0, 200, 10)
T_puros = T_analitica(t_dados)

# Adicionar ruído gaussiano (média = 0, desvio padrão = 0.5)
ruido = np.random.normal(0, 0.5, size=t_dados.shape)
T_ruidosos = T_puros + ruido

# Exibir os dados sintéticos
for i in range(len(t_dados)):
    print(f"t = {t_dados[i]:6.1f} s\tT = {T_ruidosos[i]:6.2f} °C")

# Visualização
plt.figure(figsize=(8, 5))
plt.plot(t_dados, T_puros, '--', label='Solução analítica (sem ruído)', color='blue')
plt.plot(t_dados, T_ruidosos, 'o', label='Dados sintéticos (com ruído)', color='red')
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Dados Sintéticos para Treinamento da PINN')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

