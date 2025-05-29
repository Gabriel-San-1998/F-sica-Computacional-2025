import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- Parâmetros físicos ----------
T_amb = 25
T0 = 90
r_real = 0.005  # usado apenas para gerar os dados sintéticos

# ---------- Solução analítica ----------
def T_analitica(t):
    return T_amb - (T_amb - T0) * np.exp(-r_real * t)

# ---------- Dados sintéticos ----------
np.random.seed(0)
t_dados = np.linspace(0, 200, 20)
T_puros = T_analitica(t_dados)
ruido = np.random.normal(0, 0.5, size=t_dados.shape)
T_ruidosos = T_puros + ruido

# ---------- Conversão para tensores PyTorch ----------
t_tensor = torch.tensor(t_dados, dtype=torch.float32).unsqueeze(1)
T_tensor = torch.tensor(T_ruidosos, dtype=torch.float32).unsqueeze(1)

# ---------- Rede Neural tradicional (mantida para comparação) ----------
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

print("Iniciando treinamento da NN Tradicional...")
for epoch in range(5000):
    model_nn.train()
    optimizer_nn.zero_grad()
    pred_nn = model_nn(t_tensor)
    loss_nn = loss_fn(pred_nn, T_tensor)
    loss_nn.backward()
    optimizer_nn.step()
    if epoch % 500 == 0:
        print(f"[NN]   Epoch {epoch}: Loss = {loss_nn.item():.5f}")
print("Treinamento da NN Tradicional concluído.")

# ---------- Rede PINN com r treinável (Método 3: otimizando log(r)) ----------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Otimizar o logaritmo de r
        # Inicializamos log(r) com o logaritmo do valor que esperamos (0.005)
        self._log_r = nn.Parameter(torch.tensor([torch.log(torch.tensor(0.01))], dtype=torch.float32))

    def forward(self, x):
        return self.net(x)

    # Método para obter o valor positivo de r (exponencial do logaritmo)
    def get_positive_r(self):
        return torch.exp(self._log_r)

# Gradiente via autograd
def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)[0]

# Loss física: ODE usando o r positivo
def physics_loss_discovery(model):
    # Amostra pontos de tempo no domínio (incluindo extrapolação)
    ts = torch.linspace(0, 1000, steps=1000).view(-1, 1).requires_grad_(True)
    temps = model(ts)
    dT = grad(temps, ts)

    # Equação diferencial: dT/dt = r * (T_amb - T)  => dT/dt - r * (T_amb - T) = 0
    # Ou: dT/dt + r * (T - T_amb) = 0
    # Usamos model.get_positive_r() aqui
    pde = dT + model.get_positive_r() * (temps - T_amb)
    return torch.mean(pde**2)

# Loss de dados
def data_loss(model):
    pred = model(t_tensor)
    return torch.mean((pred - T_tensor)**2)

# ---------- Treinamento da PINN ----------
model_pinn = PINN()
# Otimizamos todos os parâmetros, incluindo _log_r
optimizer_pinn = torch.optim.Adam(model_pinn.parameters(), lr=1e-3)
lambda_phy = 500 # Peso da perda física

print("\nIniciando treinamento da PINN (otimizando log(r))...")
for epoch in range(5000):
    optimizer_pinn.zero_grad()

    # A perda total é a soma da perda de dados e a perda física ponderada
    loss = data_loss(model_pinn) + lambda_phy * physics_loss_discovery(model_pinn)

    loss.backward()
    optimizer_pinn.step()

    if epoch % 500 == 0:
        # Imprimimos o valor positivo de r
        print(f"[PINN] Epoch {epoch}: Loss = {loss.item():.5f} | r = {model_pinn.get_positive_r().item():.5f}")
print("Treinamento da PINN concluído.")
print(f"Valor final de r descoberto pela PINN: {model_pinn.get_positive_r().item():.5f}")
print(f"Valor real de r: {r_real:.5f}")

# ---------- Avaliação ----------
t_test = np.linspace(0, 1000, 1000)
t_test_tensor = torch.tensor(t_test, dtype=torch.float32).unsqueeze(1)

# Desativar cálculo de gradientes para avaliação
with torch.no_grad():
    T_pred_nn = model_nn(t_test_tensor).squeeze().numpy()
    T_pred_pinn = model_pinn(t_test_tensor).squeeze().numpy()

T_real = T_analitica(t_test)

# ---------- Plot ----------
plt.figure(figsize=(10, 6))
plt.plot(t_test, T_real, '--', label='Solução Analítica (r real)', color='black')
plt.plot(t_test, T_pred_nn, label='NN Tradicional (ReLU)', color='red', alpha=0.8)
plt.plot(t_test, T_pred_pinn, label='PINN (ReLU) - r aprendido (log)', color='blue', alpha=0.8)
plt.scatter(t_dados, T_ruidosos, label='Dados Sintéticos', color='r', zorder=5, s=20) # s é o tamanho do marcador
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (°C)')
plt.title('Comparação: Solução Analítica, NN Tradicional e PINN com r aprendido (otimizando log(r))')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('descobrindo r.png')
plt.show()
