import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# 1. Gerar dados de treinamento
np.random.seed(42)
num_samples = 100
angles_train = np.random.uniform(0, 4 * np.pi, num_samples).reshape(-1, 1)
sin_values_train = np.sin(angles_train)

# Adicionar ruído
noise = np.random.normal(0, 0.1, sin_values_train.shape)
sin_values_train += noise

# 2. Criar o modelo FCNN com Keras
model = Sequential([
    Input(shape=(1,)),               # Entrada explícita, conforme boas práticas
    Dense(20, activation='tanh'),
    Dense(20, activation='tanh'),
    Dense(10, activation='tanh'),
    Dense(10, activation='tanh'),
    Dense(1)                         # Saída com um único valor (regressão)
])

# 3. Compilar o modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'  # erro quadrático médio
)

# 4. Treinar o modelo
model.fit(angles_train, sin_values_train, epochs=1000, verbose=0)

# 5. Gerar dados de teste
num_test_samples = 50
angles_test = np.linspace(0, 6 * np.pi, num_test_samples).reshape(-1, 1)
sin_values_true = np.sin(angles_test)

# 6. Fazer previsões
sin_values_predicted = model.predict(angles_test)

# 7. Avaliar o modelo
mse = mean_squared_error(sin_values_true, sin_values_predicted)
print(f"Mean Squared Error on Test Data: {mse}")

# 8. Visualizar os resultados
plt.figure(figsize=(10, 6))
plt.scatter(angles_train, sin_values_train, label='Training Data', alpha=0.5)
plt.plot(angles_test, sin_values_true, label='True sin(theta)', color='blue')
plt.plot(angles_test, sin_values_predicted, label='Predicted sin(theta)', color='red')
plt.xlabel('Angle (radians)')
plt.ylabel('sin(theta)')
plt.title('FCNN Interpolation of sin(theta) with TensorFlow/Keras')
plt.legend()
plt.grid(True)
plt.show()
