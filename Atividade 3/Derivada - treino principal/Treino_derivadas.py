import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

def generate_data(nx, qtde, pmax):
    x = np.linspace(-1, 1, nx).reshape(-1, 1)
    y = []
    dy = []

    for _ in range(qtde):
        # Decide par/par ou ímpar/ímpar aleatoriamente
        parity_type = np.random.choice(['even', 'odd'])

        if parity_type == 'even':
            # Expoentes pares
            powers = np.arange(0, pmax + 1, 2)
            # Coeficientes pares: 2 * inteiro
            coeffs = 2 * np.random.randint(-5, 6, size=len(powers))
        else:
            # Expoentes ímpares
            powers = np.arange(1, pmax + 1, 2)
            # Coeficientes ímpares: 2 * inteiro + 1
            coeffs = 2 * np.random.randint(-5, 6, size=len(powers)) + 1

        # Polinômio sem ruído
        poly = sum(c * x**p for c, p in zip(coeffs, powers))
        scale_y = np.max(np.abs(poly))
        scale_y = scale_y if scale_y > 1e-8 else 1.0
        y_i = poly / scale_y
        y.append(y_i)

        # Derivada sem ruído
        deriv = sum(c * p * x**(p - 1) for c, p in zip(coeffs, powers))
        scale_dy = np.max(np.abs(deriv))
        scale_dy = scale_dy if scale_dy > 1e-8 else 1.0
        dy_i = deriv / scale_dy
        dy.append(dy_i)

    y = np.hstack(y).T
    dy = np.hstack(dy).T
    return y, dy

# Geração de dados
y, dy = generate_data(50, 10000, 5)
print(y.shape)
print(dy.shape)

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(y, dy, test_size=0.2, random_state=42)

# Modelo MLP
neurons = 10
layers = 10

model = MLPRegressor(
    hidden_layer_sizes=tuple([neurons] * layers),
    activation='tanh',
    solver='adam',
    max_iter=100000,
    random_state=42,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    n_iter_no_change=50,
    tol=1e-8,
    verbose=True
)

# Treinamento
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# Testes com funções conhecidas
plt.figure(figsize=(15, 4))
new_x = np.linspace(-1/2, 1/2, y.shape[1]).reshape(1, -1)

# Teste 1: sin(2πx)
plt.subplot(131)
new_y = np.sin(2 * np.pi * new_x)
new_dy = np.cos(2 * np.pi * new_x)
predicted_derivative = model.predict(new_y)
plt.plot(new_x[0], new_y[0], label='Input', color='black')
plt.plot(new_x[0], new_dy[0], label='True dy/dx', color='blue')
plt.plot(new_x[0], predicted_derivative[0], label='Predicted dy/dx', color='red', linestyle='dashed')
plt.title('sin(2πx)')
plt.ylim(-2, 2)
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend()
plt.grid(True)

# Teste 2: cos(2πx)
plt.subplot(132)
new_y = np.cos(2 * np.pi * new_x)
new_dy = -np.sin(2 * np.pi * new_x)
predicted_derivative = model.predict(new_y)
plt.plot(new_x[0], new_y[0], label='Input', color='black')
plt.plot(new_x[0], new_dy[0], label='True dy/dx', color='blue')
plt.plot(new_x[0], predicted_derivative[0], label='Predicted dy/dx', color='red', linestyle='dashed')
plt.title('cos(2πx)')
plt.ylim(-2, 2)
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend()
plt.grid(True)

# Teste 3: x^2
plt.subplot(133)
new_y = new_x ** 2
new_dy = 2 * new_x
predicted_derivative = model.predict(new_y)
plt.plot(new_x[0], new_y[0], label='Input', color='black')
plt.plot(new_x[0], new_dy[0], label='True dy/dx', color='blue')
plt.plot(new_x[0], predicted_derivative[0], label='Predicted dy/dx', color='red', linestyle='dashed')
plt.title('x^2')
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('derivadas.png')
plt.show()
