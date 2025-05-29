import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# 1. Gerar dados de treinamento (intervalo seguro evitando assíntotas da tangente)
np.random.seed(42)
num_samples = 100
angles_train = np.random.uniform(-1.4, 1.4, num_samples).reshape(-1, 1)
tan_values_train = np.tan(angles_train)

# Adicionar ruído
noise = np.random.normal(0, 0.1, tan_values_train.shape)
tan_values_train += noise

# 2. Criar o modelo FCNN com Keras
model = Sequential([
    Input(shape=(1,)),
    Dense(20, activation='tanh'),
    Dense(20, activation='tanh'),
    Dense(10, activation='tanh'),
    Dense(10, activation='tanh'),
    Dense(1)
])

# 3. Compilar o modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)

# 4. Treinar o modelo
model.fit(angles_train, tan_values_train, epochs=1000, verbose=0)

# 5. Gerar dados de teste (também em intervalo seguro)
num_test_samples = 200
angles_test = np.linspace(-1.5, 1.5, num_test_samples).reshape(-1, 1)
tan_values_true = np.tan(angles_test)

# 6. Fazer previsões
tan_values_predicted = model.predict(angles_test)

# 7. Avaliar o modelo
mse = mean_squared_error(tan_values_true, tan_values_predicted)
print(f"Mean Squared Error on Test Data: {mse:.4f}")

# 8. Visualizar os resultados
plt.figure(figsize=(10, 6))
plt.scatter(angles_train, tan_values_train, label='Training Data', alpha=0.5)
plt.plot(angles_test, tan_values_true, label='True tan(theta)', color='blue')
plt.plot(angles_test, tan_values_predicted, label='Predicted tan(theta)', color='red')
plt.xlabel('Angle (radians)')
plt.ylabel('tan(theta)')
plt.title('FCNN Interpolation of tan(theta) with TensorFlow/Keras')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

