#Vamos criar alguns "lotes" de entrada, pois quanto mais entrada de informação,mais a rede é otimizada


import numpy as np

np.random.seed(0)

X = [[1,2, 3, 2.5],#Dados de entrada, perceba que são 3 lotes, portanto iremos obter 3 lotes de resultados
     [2, 5, -1, 2],#lote 2
     [-1.5, 2.7, 3.3, -0.8]] #lote 3

class Layer_Dense: #definindo uma classe para facilitar a criação de camadas
    
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+ self.biases

layer1 =  Layer_Dense(4,5) # Os parametros entre parenteses são o formato dessa camada como é solicitado no __init__
layer2 = Layer_Dense(5,2) # O primeiro número faz referencia a quantidade de entradas que a camada tem, no layer1 é 4 pq temos 4 entradas iniciais em cada lote e na layer 2 é 5 pq a layer 1 apresenta 5 saídas
                          #O segundo número faz referência a quantidade de neuronios da nossa camada e consequentemente da quantodade de saídas 

layer1.forward(X) #aplicando a função avanço na layer1 com os dados de entrada
#print(layer1.output)

layer2.forward(layer1.output) #aplicando a função avanço na layer2 com os dados de saida da layer1
print(layer2.output)
