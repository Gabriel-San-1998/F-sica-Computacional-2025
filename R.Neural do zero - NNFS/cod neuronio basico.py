import numpy as np
import math
from matplotlib import pyplot as plt
import pylab

inputs= [1.2, 5.1, 2.1] #entradas provenientes de neuronios anteriores 

weigths= [3.1, 2.1, 8.7] #peso associdado a cada entrada

bias= 3 #viés desse neuronio

output= inputs[0]*weigths[0] + inputs[1]*weigths[1] + inputs[2]*weigths[2] + bias #informação que ele passará adiante

print(output)



