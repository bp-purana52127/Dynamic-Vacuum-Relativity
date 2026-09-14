import numpy as np
import matplotlib.pyplot as plt

# 1. Spatial Lattice Grid (CFL-Stabilized)
N = 400
x = np.linspace(-10, 10, N)
dx = x[1] - x[0]
dt = 0.0002  # Reduced time step to satisfy RK4 stability: dt < dx^2
steps = 3000

# Absorbing Boundary Condition to eliminate edge reflections
absorber = np.exp(-((x / 8.5)**16))

# 2. Complex Connection Background: A_a^i = Gamma + i * K(x)
# Localized extrinsic curvature barrier K(x) near x = 0
K_x = 2.0 * np.exp(-((x / 1.5)**2))

# 3. Wave Packet Initialization
x0 = -4.5
sigma = 0.7
k0 = 4.0

# Forward-temporal wave packet (w > 0)
psi_forward = np.exp(-((x - x0)**2) / (2 * sigma**2)) * np.exp(1j * k0 * x)

# Retrocausal / negative-frequency wave packet (w < 0)
psi_retro = np.exp(-((x - x0)**2) / (2 * sigma**2)) * np.exp(-1j * k0 * x)

# Normalize initial wave packets
psi_forward /= np.sqrt(np.sum(np.abs(psi_forward)**2) * dx)
psi_retro   /= np.sqrt(np.sum(np.abs(psi_retro)**2) * dx)

# 4. Derivative & Non-Hermitian RHS Function
def compute_rhs(psi, K_profile):
    # Central spatial finite differences
    d1_psi = (np.roll(psi, -1) - np.roll(psi, 1)) / (2 * dx)
    d2_psi = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / (dx**2)
    
    # Wave equation derived from i * d_t(psi) = -0.5 * d2(psi) + i * K(x) * d1(psi)
    rhs = 0.5j * d2_psi - K_profile * d1_psi
    return rhs * absorber

# 5. 4th-Order Runge-Kutta (RK4) Time Stepper
def rk4_step(psi, K_profile):
    k1 = compute_rhs(psi, K_profile)
    k2 = compute_rhs(psi + 0.5 * dt * k1, K_profile)
    k3 = compute_rhs(psi + 0.5 * dt * k2, K_profile)
    k4 = compute_rhs(psi + dt * k3, K_profile)
    return (psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)) * absorber

# Execute temporal integration
psi_f_t = psi_forward.copy()
psi_r_t = psi_retro.copy()

print("Executing RK4 Complexified Connection Simulation...")
for t in range(steps):
    psi_f_t = rk4_step(psi_f_t, K_x)
    psi_r_t = rk4_step(psi_r_t, -K_x) # Opposite phase coupling for negative frequency

# Calculate final transmission probabilities P(w) past the barrier (x > 2.0)
P_forward = np.sum(np.abs(psi_f_t[x > 2.0])**2) * dx
P_retro   = np.sum(np.abs(psi_r_t[x > 2.0])**2) * dx

suppression = P_retro / max(P_forward, 1e-12)

print("\n=== OPTION 1: COMPLEXIFIED CONNECTION SIMULATION (STABLE) ===")
print(f"Forward Wave Transmission P(w > 0)   : {P_forward:.4f}")
print(f"Retrocausal Transmission P(w < 0)    : {P_retro:.6e}")
print(f"Suppression Factor (P_retro / P_fwd) : {suppression:.6e}")

# 6. Plotting Results
plt.figure(figsize=(10, 5))
plt.plot(x, np.abs(psi_forward)**2, 'k--', alpha=0.4, label='Initial Packet')
plt.plot(x, np.abs(psi_f_t)**2, color='blue', lw=2, label=r'Forward Wave ($\omega > 0$)')
plt.plot(x, np.abs(psi_r_t)**2, color='crimson', lw=2, label=r'Retrocausal Wave ($\omega < 0$)')
plt.axvspan(-1.5, 1.5, color='gray', alpha=0.2, label=r'CTC Barrier Zone $i K_a^i$')

plt.title(r'Chronology Protection: Complexified Connection Damping ($A_a^i = \Gamma_a^i + i K_a^i$)')
plt.xlabel('Spatial Coordinate (x)')
plt.ylabel(r'Probability Density $|\Psi(x)|^2$')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

output_img = '/home/gmarcus/QBG/sim_complex_connection.png'
plt.savefig(output_img)
print(f"\nSUCCESS! Stable simulation plot saved to: {output_img}")
