import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- Parâmetros físicos ----------
T_amb = 25
T0 = 90
r = 0.005

# ---------- Solução analítica ----------
def T_analitica(t):
    return T_amb - (T_amb - T0) * np.exp(-r * t)

# ---------- Dados sintéticos ----------
np.random.seed(0)
t_dados = np.linspace(0, 200, 10)
T_puros = T_analitica(t_dados)
ruido = np.random.normal(0, 0.5, size=t_dados.shape)
T_ruidosos = T_puros + ruido

# ---------- Preparação dos dados para PyTorch ----------
t_tensor = torch.tensor(t_dados, dtype=torch.float32).unsqueeze(1)
T_tensor = torch.tensor(T_ruidosos, dtype=torch.float32).unsqueeze(1)

# ---------- Definição da rede neural ----------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------- Inicialização e treinamento ----------
model = SimpleNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Treinamento
epochs = 5000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(t_tensor)
    loss = loss_fn(pred, T_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.5f}")

# ---------- Avaliação: extrapolação até 1000 s ----------
t_test = np.linspace(0, 1000, 200)
t_test_tensor = torch.tensor(t_test, dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    T_pred = model(t_test_tensor).squeeze().numpy()

# Solução analítica para comparação
T_real = T_analitica(t_test)

# ---------- Plot ----------
plt.figure(figsize=(10, 5))
plt.plot(t_test, T_real, label='Solução analítica', linestyle='--', color='blue')
plt.plot(t_test, T_pred, label='Rede Neural (NN)', color='red')
plt.scatter(t_dados, T_ruidosos, label='Dados sintéticos', color='black', zorder=5)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Ajuste e Extrapolação com Rede Neural')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
