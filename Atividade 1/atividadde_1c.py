import numpy as np
import math
from matplotlib import pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 10.0)

a = []
b = []

def f(x):
    return (x**2)*(x - 1)*(x + 1) + (x/4)  # função

def df(x):
    return 4*x**3 - 2*x + 0.25  # derivada primeira

def ddf(x):
    return 12*x**2 - 2  # derivada segunda

# chute inicial
x_0 = 5

# passo inicial
step_inicial = 3

# tolerância para critério de parada
tolerancia = 0.1

contador = 1
step = step_inicial

# loop com critério de parada por tolerância e máximo de iterações
while not (abs(df(x_0)) < tolerancia and ddf(x_0) > 0) and contador < 10000:
    a.append(x_0)
    b.append(f(x_0))

    if df(x_0) > 0:
        x_0 -= step
    elif df(x_0) < 0:
        x_0 += step

    # Redução controlada do passo
    step = max(tolerancia, step_inicial - (contador / 2))
    contador += 1

# adiciona o ponto final
a.append(x_0)
b.append(f(x_0))

print("O mínimo está localizado em", "x =", x_0, "e y =", f(x_0))
print('Quantidade de iterações:', contador)

x = np.linspace(-2.5, 2.5, 100)
plt.grid()
plt.plot(x, f(x), color='r')
plt.plot(a, b, color='k', linestyle='--')
plt.title('Atividade 1c', fontsize=18)
plt.ylabel(r'f(x)', fontsize=16)
plt.xlabel('x', fontsize=16)
# plt.savefig('atv1c.png')
plt.savefig(r'/home/gabas/Área de trabalho/F-sica-Computacional-2025/Atividade 1/1atv1c.png')
plt.show()