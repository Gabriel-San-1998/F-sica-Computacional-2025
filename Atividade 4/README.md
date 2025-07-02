# Atividade 04
Nesta atividade comparamos o desempenho de uma NN e de uma PINN na descrição do resfriamento de uma caneca de café, principalmente no que diz respeito a extrapolação.
![ derivadas](Exercicio 1 - Pt1 e 2.py)
A rede neural funciona muito bem para os dados treinados, porém falha miseravelmente assim que a extrapolação começa.
![ derivadas](Exercicio 1 - Pt3.py)
A PINN funciona muito bem aplicando tanto a perda física quanto a função perda dos dados, principalmente quando a taxa de resfrimento é informada.
![ derivadas](Exercicio 1 - Pt4.py)
Quando a taxa de resfriamento não é informada o peso da perda física se torna muito importante 
e para o algoritmo apresentado, só conseguimos bons resultados para um coeficiente que proporcionalmente valoriza 
mais a informação física em relação a perda dos dados.
![ derivadas](Exercicio 1 - Pt5.py)
![ derivadas](Exercicio 1 - Pt6.py)
(OBS: A função de ativação tan não apresenta bons resultados, foi necessário usar a ReLU)
