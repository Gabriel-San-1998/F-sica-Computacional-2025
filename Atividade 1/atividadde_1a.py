import numpy as np
import math
from matplotlib import pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 10.0)

a=[]
b=[]

def f(x):
 return (x**2)-1

def df(x):
  return 2*x

def ddf(x):
  return 2

#chute inicial
x_0 = 3

#Passo utilizado (o professor quer um pequeno e um maior)

step = 0.25

if step <= 3:

      while df(x_0) != 0:

        a.append(x_0)
        b.append(f(x_0))

        if df(x_0) > 0:
            x_0 -= step
        else:
            x_0 += step

      #esse append fora do loop é necessário para que o valor minimo também seja adicionado as listas a e b
      a.append(x_0)
      b.append(f(x_0))
      print("O mínimo está localizado em", "x=", x_0, "e y=", f(x_0))
      x=np.linspace(-3,3,100)
      print(a)
      print(b)
      plt.grid()
      plt.plot(x, f(x),color='r')
      plt.plot(a, b,color='k')
      #plt.plot(a, b,color='orange')
      plt.title('Atividade 1a',fontsize=18)
      plt.ylabel(r'f(x)',fontsize=16)
      plt.xlabel('x',fontsize=16)
      plt.savefig('atv1b.png')
      plt.savefig('atv1a.png')


else:
      while df(x_0) != 0:

            a.append(x_0)
            b.append(f(x_0))

            if df(x_0) > 0:
                x_0 -= step
            else:
                x_0 += step

            step -=0.5

      #esse append fora do loop é necessário para que o valor minimo também seja adicionado as listas a e b
      a.append(x_0)
      b.append(f(x_0))
      print("O mínimo está localizado em", "x=", x_0, "e y=", f(x_0))
      x=np.linspace(-3,3,100)
      print(a)
      print(b)
      plt.grid()
      plt.plot(x, f(x),color='r')
      plt.plot(a, b,color='k')
     # plt.savefig(atv1a)
      #plt.plot(a, b,color='orange')
      plt.title('Atividade 1a',fontsize=18)
      plt.ylabel(r'f(x)',fontsize=16)
      plt.xlabel('x',fontsize=16)
      plt.show()
     # plt.savefig('atv1b.png')
      plt.savefig('atv1a.png')
      
