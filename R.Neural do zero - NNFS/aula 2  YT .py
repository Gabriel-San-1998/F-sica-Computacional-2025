#Cod para 3 neuronios

import numpy as np
import math
from matplotlib import pyplot as plt
import pylab

inputs= [1,2, 3, 2.5]#entradas provenientes de neuronios anteriores 

weigths1= [0.2, 0.8, -0.5, 1] #peso associdado a cada entrada
weigths2= [0.5, -0.91, 0.26, -0.5]
weigths3= [-0.26, -0.27, 0.17, .87]

bias1= 2 #viés desse neuronio
bias2= 3 
bias3= 0.5 

output= [inputs[0]*weigths1[0] + inputs[1]*weigths1[1] + inputs[2]*weigths1[2] + inputs[3]*weigths1[3]  + bias1,

inputs[0]*weigths2[0] + inputs[1]*weigths2[1] + inputs[2]*weigths2[2] + inputs[3]*weigths2[3]  + bias2,

inputs[0]*weigths3[0] + inputs[1]*weigths3[1] + inputs[2]*weigths3[2] + inputs[3]*weigths3[3]  + bias3, ] #informação que ele passará adiante

print(output)

#video 2YT

