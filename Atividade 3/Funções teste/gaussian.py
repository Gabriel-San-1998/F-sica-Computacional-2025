import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# 1. Gerar dados de treinamento
np.random.seed(42)
num_samples = 200
x_train = np.linspace(-10, 10, num_samples).reshape(-1, 1)

# Função gaussiana f(x) = e^(-x^2)
def gaussian(x):
    return np.exp(-x**2)

y_train = gaussian(x_train)

# Adicionar ruído opcional
noise = np.random.normal(0, 0.01, y_train.shape)
y_train += noise

# 2. Criar o modelo FCNN
model = Sequential([
    Input(shape=(1,)),
    Dense(64, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(32, activation='tanh'),
    Dense(1)
])

# 3. Compilar o modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)

# 4. Treinar o modelo
model.fit(x_train, y_train, epochs=500, verbose=0)

# 5. Gerar dados de teste
x_test = np.linspace(-10, 10, 500).reshape(-1, 1)
y_true = gaussian(x_test)

# 6. Fazer previsões
y_pred = model.predict(x_test)

# 7. Avaliar o modelo
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error on Test Data: {mse:.6f}")

# 8. Visualizar os resultados
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_true, label='True Gaussian (e^(-x²))', color='blue')
plt.plot(x_test, y_pred, label='Predicted Gaussian', color='red')
plt.scatter(x_train, y_train, label='Training Data', alpha=0.3, color='gray', s=10)
plt.xlabel('x')
plt.ylabel('f(x) = e^(-x²)')
plt.title('FCNN Approximation of Gaussian Function')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
