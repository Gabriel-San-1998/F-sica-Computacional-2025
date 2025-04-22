#Vamos criar alguns "lotes" de entrada, pois quanto mais entrada de informação,mais a rede é otimizada


import numpy as np


inputs= [[1,2, 3, 2.5],#entradas provenientes de neuronios anteriores 
        [2, 5, -1, 2],
        [-1.5, 2.7, 3.3, -0.8]
        ]
weigths= [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]

biases= [2, 3, 0.5]


weigths2= [[0.1, -0.14, 0.5], 
           [-0.5, 0.12, -0.33], 
           [-0.44, 0.73, -0.13]]

biases2= [-1, 2, -0.5]
layer1_outputs = np.dot(inputs, np.array(weigths).T)+biases

layer2_outputs= np.dot(layer1_outputs, np.array(weigths2).T)+biases2
print(layer2_outputs)

#video 4YT pt1

