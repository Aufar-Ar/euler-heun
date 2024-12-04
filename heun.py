import numpy as np
import matplotlib.pyplot as plt
# Parameter
A = 0.100
beta = 0.80
gamma = 0.60
sigma = 0.50
mu = 0.12
# Kondisi awal
S0 = 10
I0 = 5
R0 = 2
h = 0.1
steps = 300
# Fungsi untuk menghitung perubahan dalam model SIR
def dS_dt(S, I, N):
    return A - mu * S - beta * S * I / N

def dI_dt(S, I, N):
    return beta * S * I / N - (mu + gamma) * I

def dR_dt(I, R):
    return gamma * I - (mu + sigma) * R
# Metode Heun untuk menyelesaikan model SIR
def heun_method(S0, I0, R0, N, h, steps):
    S, I, R = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    S[0], I[0], R[0] = S0, I0, R0

    for t in range(1, steps):
        # Prediktor
        S_star = S[t-1] + h * dS_dt(S[t-1], I[t-1], N)
        I_star = I[t-1] + h * dI_dt(S[t-1], I[t-1], N)
        R_star = R[t-1] + h * dR_dt(I[t-1], R[t-1])

        # Korektor
        S[t] = S[t-1] + (h / 2) * (dS_dt(S[t-1], I[t-1], N) + dS_dt(S_star, I_star, N))
        I[t] = I[t-1] + (h / 2) * (dI_dt(S[t-1], I[t-1], N) + dI_dt(S_star, I_star, N))
        R[t] = R[t-1] + (h / 2) * (dR_dt(I[t-1], R[t-1]) + dR_dt(I_star, R_star))

    return S, I, R
# Simulasi untuk total populasi
N = S0 + I0 + R0

# Menjalankan simulasi dengan metode Heun
S_heun, I_heun, R_heun = heun_method(S0, I0, R0, N, h, steps)
# Plot hasil
plt.figure(figsize=(10, 6))
time = np.arange(steps) * h

plt.plot(time, S_heun, label='Susceptible (S)', color='yellow')
plt.plot(time, I_heun, label='Infected (I)', color='black')
plt.plot(time, R_heun, label='Recovered (R)', color='green')

plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using Heun\'s Method')
plt.legend()
plt.grid(True)
plt.show()