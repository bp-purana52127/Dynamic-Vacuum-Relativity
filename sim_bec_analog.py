import numpy as np
import matplotlib.pyplot as plt

# 1. Spatial Domain & Acoustic Horizon Geometry
N = 500
x = np.linspace(-10, 10, N)
dx = x[1] - x[0]
dt = 0.002
steps = 800

# Background Condensate Profile: Density n(x) drops, Velocity v(x) increases
# Acoustic Horizon sits at x = 0 where v(x) = c_s(x)
n0 = 1.0 - 0.4 * np.tanh(x / 2.0)    # Condensate density
c_s = np.sqrt(n0)                    # Local sound speed
v_fluid = 0.8 + 0.5 * np.tanh(x / 2.0) # Local flow velocity

mach_number = v_fluid / c_s          # Mach > 1 is supersonic acoustic BH

# 2. Phonon Wave Packet (Injected upstream at x = -6, traveling right)
x_p = -6.0
sigma_p = 0.6
k_p = 6.0
phi_phonon = np.exp(-((x - x_p)**2) / (2 * sigma_p**2)) * np.exp(1j * k_p * x)

# 3. Acoustic Wave Equation in Curved Background Metric
# d2_phi/dt2 = c_s^2 * d2_phi/dx2 - 2*v*d2_phi/dtdx
phi = phi_phonon.copy()
phi_prev = phi_phonon.copy()

trans_history = []

for t in range(steps):
    # Spatial derivatives
    d2_phi = (np.roll(phi, -1) - 2 * phi + np.roll(phi, 1)) / (dx**2)
    d1_phi = (np.roll(phi, -1) - np.roll(phi, 1)) / (2 * dx)
    
    # Time derivative estimate
    d_dt_phi = (phi - phi_prev) / dt
    d1_d_dt_phi = (np.roll(d_dt_phi, -1) - np.roll(d_dt_phi, 1)) / (2 * dx)
    
    # Curved metric acoustic evolution
    d2_dt2_phi = (c_s**2) * d2_phi - 2 * v_fluid * d1_d_dt_phi
    
    # Velocity Verlet update
    phi_next = 2 * phi - phi_prev + (dt**2) * d2_dt2_phi
    phi_prev = phi.copy()
    phi = phi_next.copy()
    
    # Measure transmission amplitude beyond acoustic horizon (x > 3.0)
    trans_amplitude = np.sum(np.abs(phi[x > 3.0])**2) * dx
    trans_history.append(trans_amplitude)

initial_power = np.sum(np.abs(phi_phonon)**2) * dx
final_power = trans_history[-1]
transmission_ratio = final_power / max(initial_power, 1e-12)

print("\n=== OPTION 2: ANALOG GRAVITY BEC HORIZON SIMULATION ===")
print(f"Supersonic Transition Point  : x = 0.0 (Mach = 1.0)")
print(f"Initial Phonon Wave Power    : {initial_power:.4f}")
print(f"Transmitted Phonon Power     : {final_power:.4f}")
print(f"Acoustic Transmission Ratio  : {transmission_ratio:.4f}")

# 4. Graphical Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top Panel: Condensate Flow Profile
ax1.plot(x, c_s, 'b-', lw=2, label=r'Local Sound Speed $c_s(x)$')
ax1.plot(x, v_fluid, 'r-', lw=2, label=r'Fluid Flow Velocity $v(x)$')
ax1.axhline(1.0, color='gray', linestyle=':')
ax1.axvline(0.0, color='black', linestyle='--', label='Acoustic Horizon (Mach = 1)')
ax1.set_ylabel('Velocity')
ax1.set_title('BEC Acoustic Black Hole Horizon Geometry')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# Bottom Panel: Phonon Wave Damping across Horizon
ax2.plot(x, np.abs(phi_phonon)**2, 'g--', alpha=0.5, label='Initial Phonon Packet')
ax2.plot(x, np.abs(phi)**2, 'crimson', lw=2, label='Phonon Packet at Horizon Boundary')
ax2.axvspan(0.0, 10.0, color='red', alpha=0.1, label='Supersonic Region ($v > c_s$)')
ax2.set_xlabel('Spatial Coordinate (x)')
ax2.set_ylabel(r'Phonon Amplitude $|\phi(x)|^2$')
ax2.set_title('Phonon Phase Damping across Supersonic Horizon')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
output_img = '/home/gmarcus/QBG/sim_bec_analog.png'
plt.savefig(output_img)
print(f"SUCCESS! Output plot saved to: {output_img}")
