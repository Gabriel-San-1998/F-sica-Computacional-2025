import numpy as np
from matplotlib import pyplot as plt
import pylab
import random

# Ajuste de visualização
pylab.rcParams['figure.figsize'] = (10.0, 10.0)

# Listas para armazenar os pontos visitados
a = []
b = []

# Definição da função e suas derivadas
def f(x):
    return (x**2)*(x - 1)*(x + 1)  # x^4 - x^2

def df(x):
    return 4*x**3 - 2*x  # derivada correta

def ddf(x):
    return 12*x**2 - 2   # segunda derivada

# Chute inicial
x_0 = 5

# Parâmetros do algoritmo
step_inicial = 1
step = step_inicial
tolerancia = 0.1
contador = 1

# Loop de busca do mínimo
while not (abs(df(x_0)) < tolerancia and ddf(x_0) > 0) and contador < 10000:
    a.append(x_0)
    b.append(f(x_0))

    if df(x_0) > 0:
        x_0 -= step
    elif df(x_0) < 0:
        x_0 += step
    else:
        #print("Temos um ponto de máximo localizado em", "x =", x_0, "e y =", f(x_0))
        sinal = random.choice([-1, 1])
        x_0 += sinal * step  # Deslocamento aleatório

    # Redução adaptativa do passo
    if step > 1:
        step = max(tolerancia, step * 0.9)

    contador += 1

# Armazena o ponto final
a.append(x_0)
b.append(f(x_0))

# Resultados finais
print("O mínimo está localizado em", "x =", x_0, "e y =", f(x_0))
print("Quantidade de iterações:", contador)
#print("Caminho seguido (x):", a)
#print("Caminho seguido (y):", b)

# Gráfico da função e do caminho percorrido
x = np.linspace(-2.5, 2.5, 100)
plt.grid()
plt.plot(x, f(x), color='r', label='f(x)')
plt.plot(a, b, color='k', linestyle='--', label='Trajetória')

plt.title('Atividade 1b', fontsize=18)
plt.ylabel('f(x)', fontsize=16)
plt.xlabel('x', fontsize=16)
plt.legend()

# Salvar a imagem
plt.savefig(r'/home/gabas/Área de trabalho/F-sica-Computacional-2025/Atividade 1/1atv1b.png')
plt.show()