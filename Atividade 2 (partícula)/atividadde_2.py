import numpy as np
import matplotlib.pyplot as plt

# Listas para armazenar os dados
lista_x = []
lista_y = []
lista_vx = []
lista_vy = []
lista_tempo = []

class Particle:
    def __init__(self, x, y, vx, vy, massa):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.massa = massa

    def newton(self, fx, fy, dt):
        ax = fx / self.massa
        ay = fy / self.massa

        # Atualiza posições com aceleração
        self.x += self.vx * dt + 0.5 * ax * dt**2
        self.y += self.vy * dt + 0.5 * ay * dt**2

        # Atualiza velocidades
        self.vx += ax * dt
        self.vy += ay * dt

# Condições iniciais
dt = 0.01
tempo = 0.0
particula1 = Particle(0, 0, 10, 10, 1)

# Loop até a partícula atingir o solo novamente (y <= 0), ignorando o tempo inicial
while True:
    # Salva os dados
    lista_x.append(particula1.x)
    lista_y.append(particula1.y)
    lista_vx.append(particula1.vx)
    lista_vy.append(particula1.vy)
    lista_tempo.append(tempo)

    # Atualiza o estado da partícula
    particula1.newton(0, -9.8, dt)
    tempo += dt

    # Condição de parada: partícula atinge o chão (mas não no ponto inicial)
    if tempo > 0 and particula1.y <= 0:
        break

# Exemplo de como acessar os dados:
print(f"Última posição: x = {lista_x[-1]:.2f}, y = {lista_y[-1]:.2f}")
print(lista_x)
print(lista_y)
print(lista_vx)
print(lista_vy)

# Gráfico paramétrico da trajetória
plt.figure(figsize=(8, 6))
plt.plot(lista_x, lista_y, linestyle='-', marker='o', markersize=3, color='blue', label='Trajetória')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trajetória da Partícula (x vs y)')
plt.grid(True)
plt.legend()
plt.axis('equal')  # Para manter proporções reais da trajetória
plt.show()

