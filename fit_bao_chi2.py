import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Published DESI BAO Consensus Points [z_eff, DM/r_s, error_sigma]
desi_bao_data = np.array([
    [0.30,  9.23,  0.22],  # BGS
    [0.51, 13.62,  0.25],  # LRG1
    [0.71, 16.85,  0.30],  # LRG2
    [0.93, 21.71,  0.35],  # LRG3 + ELG1
    [1.32, 27.79,  0.53],  # ELG2
    [1.49, 30.20,  1.10],  # QSO
    [2.33, 39.71,  0.95]   # Lyman-alpha
])

z_obs     = desi_bao_data[:, 0]
DM_obs    = desi_bao_data[:, 1]
sigma_obs = desi_bao_data[:, 2]

# 2. Cosmological Parameters
H0  = 67.4      # km/s/Mpc
c   = 299792.458 # km/s
r_s = 147.5     # Sound horizon at drag epoch in Mpc
Om  = 0.315
Ode = 0.685

# Model A: Standard Static Lambda-CDM (w0 = -1.0, wa = 0.0)
def E_lcdm(z):
    return np.sqrt(Om * (1 + z)**3 + Ode)

# Model B: Paper's Dynamic Vacuum Potential (w0 = -0.82, wa = -0.41)
def E_dynamic_phi(z, w0=-0.82, wa=-0.41):
    f_z = ((1 + z)**(3 * (1 + w0 + wa))) * np.exp(-3 * wa * z / (1 + z))
    return np.sqrt(Om * (1 + z)**3 + Ode * f_z)

def DM_over_rs(z, E_func):
    integral, _ = quad(lambda z_: 1.0 / E_func(z_), 0, z)
    DM = (c / H0) * integral
    return DM / r_s

# 3. Compute Predictions & Reduced Chi-Squared Statistics
DM_lcdm_pred = np.array([DM_over_rs(z, E_lcdm) for z in z_obs])
DM_phi_pred  = np.array([DM_over_rs(z, E_dynamic_phi) for z in z_obs])

chi2_lcdm = np.sum(((DM_obs - DM_lcdm_pred) / sigma_obs)**2)
chi2_phi  = np.sum(((DM_obs - DM_phi_pred) / sigma_obs)**2)

dof = len(z_obs) - 2 # Degrees of freedom
red_chi2_lcdm = chi2_lcdm / dof
red_chi2_phi  = chi2_phi / dof

print("\n=== STATISTICAL GOODNESS-OF-FIT EVALUATION ===")
print(f"Standard Lambda-CDM  : Chi^2 = {chi2_lcdm:.2f} | Reduced Chi^2 = {red_chi2_lcdm:.3f}")
print(f"Paper Dynamic Vacuum : Chi^2 = {chi2_phi:.2f} | Reduced Chi^2 = {red_chi2_phi:.3f}")

# 4. Generate Graphical Fit
z_grid = np.linspace(0.1, 2.5, 100)
curve_lcdm = [DM_over_rs(z, E_lcdm) for z in z_grid]
curve_phi  = [DM_over_rs(z, E_dynamic_phi) for z in z_grid]

plt.figure(figsize=(10, 6))
plt.errorbar(z_obs, DM_obs, yerr=sigma_obs, fmt='o', color='black', ecolor='gray', elinewidth=2, capsize=4, label='DESI BAO Observed Data')
plt.plot(z_grid, curve_lcdm, color='blue', linestyle='--', lw=2, label=fr'Standard $\Lambda$CDM ($\chi^2_{{red}} = {red_chi2_lcdm:.2f}$)')
plt.plot(z_grid, curve_phi, color='crimson', lw=2.5, label=fr'Dynamic Vacuum $\Phi(x^\mu)$ ($\chi^2_{{red}} = {red_chi2_phi:.2f}$)')

plt.xlabel('Redshift (z)')
plt.ylabel(r'$D_M(z) / r_s$ (Comoving Distance / Sound Horizon)')
plt.title(r'DESI BAO Observational Data vs. Dynamic Vacuum Potential $V(\Phi)$')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

output_path = '/home/gmarcus/QBG/desi_bao_chi2_fit.png'
plt.savefig(output_path)
print(f"\nSUCCESS! Plot saved to: {output_path}")
