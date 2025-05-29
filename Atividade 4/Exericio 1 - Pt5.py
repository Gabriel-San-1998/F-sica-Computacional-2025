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

# ---------- Conversão para tensores PyTorch ----------
t_tensor = torch.tensor(t_dados, dtype=torch.float32).unsqueeze(1)
T_tensor = torch.tensor(T_ruidosos, dtype=torch.float32).unsqueeze(1)

# ---------- Rede Neural tradicional ----------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------- Treinamento da NN tradicional ----------
model_nn = SimpleNN()
optimizer_nn = torch.optim.Adam(model_nn.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(5000):
    model_nn.train()
    optimizer_nn.zero_grad()
    pred_nn = model_nn(t_tensor)
    loss_nn = loss_fn(pred_nn, T_tensor)
    loss_nn.backward()
    optimizer_nn.step()
    if epoch % 500 == 0:
        print(f"[NN]   Epoch {epoch}: Loss = {loss_nn.item():.5f}")

# ---------- Rede PINN ----------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)[0]

def physics_loss(model):
    ts = torch.linspace(0, 1000, steps=1000).view(-1, 1).requires_grad_(True)
    temps = model(ts)
    dT = grad(temps, ts)
    ode = dT + r * (temps - T_amb)
    return torch.mean(ode**2)

def data_loss(model):
    pred = model(t_tensor)
    return torch.mean((pred - T_tensor)**2)

# ---------- Treinamento da PINN ----------
model_pinn = PINN()
optimizer_pinn = torch.optim.Adam(model_pinn.parameters(), lr=1e-3)
lambda_phy = 7.5

for epoch in range(5000):
    optimizer_pinn.zero_grad()
    loss = data_loss(model_pinn) + lambda_phy * physics_loss(model_pinn)
    loss.backward()
    optimizer_pinn.step()
    if epoch % 500 == 0:
        print(f"[PINN] Epoch {epoch}: Loss = {loss.item():.5f}")

# ---------- Avaliação ----------
t_test = np.linspace(0, 1000, 1000)
t_test_tensor = torch.tensor(t_test, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    T_pred_nn = model_nn(t_test_tensor).squeeze().numpy()
    T_pred_pinn = model_pinn(t_test_tensor).squeeze().numpy()

T_real = T_analitica(t_test)

# ---------- Plot ----------
plt.figure(figsize=(10, 5))
plt.plot(t_test, T_real, '--', label='Solução Analítica', color='black')
plt.plot(t_test, T_pred_nn, label='NN Tradicional (ReLU)', color='red')
plt.plot(t_test, T_pred_pinn, label='PINN (ReLU)', color='blue')
plt.scatter(t_dados, T_ruidosos, label='Dados Sintéticos', color='r', zorder=5)
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Comparação: Solução Analítica, NN (ReLU) e PINN (ReLU)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('PINN x NN.png')
plt.show()

