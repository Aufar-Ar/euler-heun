import numpy as np
import matplotlib.pyplot as plt

# Parameter
delta_t = 0.1
A = 0.100
beta = 0.80
gamma = 0.60
delta = 0.50
mu = 0.12

# Fungsi perubahan
def dS_dt(S, I, N):
    return A - mu * S - (beta * S * I) / N

def dI_dt(S, I, N):
    return (beta * S * I) / N - (mu + gamma) * I

def dR_dt(I, R):
    return gamma * I - (mu + delta) * R
# Metode Euler
def euler_method(S0, I0, R0, N, dt, steps):
    S, I, R = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    S[0], I[0], R[0] = S0, I0, R0

    for t in range(1, steps):
        dS = dS_dt(S[t-1], I[t-1], N) * dt
        dI = dI_dt(S[t-1], I[t-1], N) * dt
        dR = dR_dt(I[t-1], R[t-1]) * dt

        S[t] = S[t-1] + dS
        I[t] = I[t-1] + dI
        R[t] = R[t-1] + dR

    return S, I, R
# Parameter simulasi
S0, I0, R0 = 10, 5, 0  # Kondisi awal
N = S0 + I0 + R0  # Total populasi
dt = 0.1  # Langkah waktu
steps = 300  # Jumlah langkah waktu

# Simulasi
S_euler, I_euler, R_euler = euler_method(S0, I0, R0, N, dt, steps)

# Plot hasil
plt.figure(figsize=(10, 6))
plt.plot(np.arange(steps) * dt, S_euler, 'y*-', label='S(t) adalah susceptible')
plt.plot(np.arange(steps) * dt, I_euler, 'k-', label='I(t) adalah infected')
plt.plot(np.arange(steps) * dt, R_euler, 'go-', label='R(t) adalah recovered')
plt.xlabel('Waktu (t)')
plt.ylabel('Nilai S(t), I(t), R(t)')
plt.title('Metode Euler dalam Menyelesaikan Sistem Transmisi HIV-AIDS Model Epidemi')
plt.legend()
plt.grid(True)
plt.show()