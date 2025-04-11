import numpy as np
import math
from matplotlib import pyplot as plt
import pylab
import random
pylab.rcParams['figure.figsize'] = (10.0, 10.0)

a=[]
b=[]
c=[]
d=[]


def f(x):
    return (x**2)*(x - 1)*(x + 1)  # x^4 - x^2

def df(x):
    return 4*x**3 - 2*x  # derivada correta

def ddf(x):
    return 12*x**2 - 2   # segunda derivada


#chute inicial
x_0 = 2

#chute inicial simétrico para encontrar o segundo mínimo

#x_1 = -x_0


#Passo utilizado (o professor quer um pequeno e um maior)

step_inicial = 3 #valores do step menores que 1 só funcionam se for 0.50, 0.25, 0.125 e assim por diante

#Se o step for menor que 1 ele não precisa ser alterado durante a iteração, porém se for maior ele precisa reduzir conforme iteramos ou o algoritmo entra em loop e não termina de rodar nunca
tolerancia = 0.1

contador = 1 #Necessário para restaurar o valor do step indicado no inicio pois ele é diminuido na geração do primeiro gráfico.

step = step_inicial

while not (df(x_0)==0 and ddf(x_0) > 0) and contador <10000:

            a.append(x_0)
            b.append(f(x_0))

            if df(x_0) > 0:
                x_0 -= step

            elif df(x_0) < 0:
                x_0 += step

            elif df(x_0)==0:
              print("Temos um ponto de máximo localizado em", "x=", x_0, "e y=", f(x_0))
              sinal = random.choice([-1, 1])  # Escolhe aleatoriamente -1 ou 1
             # x_0 +=  step   # Multiplica pelo passo ou qualquer número

            #step -= (contador / 2)
            step = max(tolerancia, (step_inicial- ( contador/2)))
            contador +=1


a.append(x_0)
b.append(f(x_0))


print("O mínimo está localizado em", "x=", x_0, "e y=", f(x_0))

x=np.linspace(-2.5,2.5,100)
print(a)
print(b)
print('Quantidade de iterações:', contador)
plt.grid()
plt.plot(x, f(x),color='r')
plt.plot(a, b,color='k', linestyle='--')

plt.title('Atividade 1b',fontsize=18)
plt.ylabel(r'f(x)',fontsize=16)
plt.xlabel('x',fontsize=16)
plt.savefig(r'/home/gabas/Área de trabalho/F-sica-Computacional-2025/Atividade 1/1atv1b.png')


