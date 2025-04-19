#Vamos criar alguns "lotes" de entrada, pois quanto mais entrada de informação,mais a rede é otimizada


import numpy as np


inputs= [[1,2, 3, 2.5],#entradas provenientes de neuronios anteriores 
        [2, 5, -1, 2],
        [-1.5, 2.7, 3.3, -0.8]
        ]
weigths= [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]

biases= [2, 3, 0.5]

layer_outputs = []

output= np.dot(inputs, np.array(weigths).T)+biases
print(output)

#video 4YT pt1

