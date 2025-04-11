import numpy as np
import math
from matplotlib import pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 10.0)

a = []
b = []

def f(x):
    return (x**2) - 1

def df(x):
    return 2 * x

def ddf(x):
    return 2

# chute inicial
x_0 = 5

# passo utilizado
step = 4

max_iter = 1000
contador = 0

if step <= 3:
    while df(x_0) != 0 and contador < max_iter:
        a.append(x_0)
        b.append(f(x_0))

        if df(x_0) > 0:
            x_0 -= step
        else:
            x_0 += step
        
        contador += 1

    a.append(x_0)
    b.append(f(x_0))
    print("O mínimo está localizado em", "x=", x_0, "e y=", f(x_0))
    print("Iterações:", contador)

    x = np.linspace(-3, 3, 100)
    plt.grid()
    plt.plot(x, f(x), color='r')
    plt.plot(a, b, color='k')
    plt.title('Atividade 1a', fontsize=18)
    plt.ylabel(r'f(x)', fontsize=16)
    plt.xlabel('x', fontsize=16)
    plt.savefig(r'/home/gabas/Área de trabalho/F-sica-Computacional-2025/Atividade 1/1atv1a.png')

else:
    while df(x_0) != 0 and contador < max_iter:
        a.append(x_0)
        b.append(f(x_0))

        if df(x_0) > 0:
            x_0 -= step
        else:
            x_0 += step

        step -= 0.5
        contador += 1

    a.append(x_0)
    b.append(f(x_0))
    print("O mínimo está localizado em", "x=", x_0, "e y=", f(x_0))
    print("Iterações:", contador)

    x = np.linspace(-3, 3, 100)
    plt.grid()
    plt.plot(x, f(x), color='r')
    plt.plot(a, b, color='k')
    plt.title('Atividade 1a', fontsize=18)
    plt.ylabel(r'f(x)', fontsize=16)
    plt.xlabel('x', fontsize=16)
    plt.savefig(r'/home/gabas/Área de trabalho/F-sica-Computacional-2025/Atividade 1/1atv1a.png')

      
