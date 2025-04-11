#Cod para 2 neuronio com np

import numpy as np

inputs= [1,2, 3, 2.5]#entradas provenientes de neuronios anteriores 

weigths= [0.2, 0.8, -0.5, 1]

biases= 2

output= np.dot(weigths, inputs) + biases

print(output)

#video 3YT pt2

