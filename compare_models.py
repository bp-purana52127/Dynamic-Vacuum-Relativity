import numpy as np
from scipy.integrate import quad

z_obs     = np.array([0.30,  0.51,  0.71,  0.93,  1.32,  1.49,  2.33])
d_obs     = np.array([9.23, 13.62, 16.85, 21.71, 27.79, 30.20, 39.71])
sigma_obs = np.array([0.22,  0.25,  0.30,  0.35,  0.53,  1.10,  0.95])
N = len(z_obs)

C = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            C[i, j] = sigma_obs[i]**2
        else:
            rho = 0.20 * np.exp(-abs(z_obs[i] - z_obs[j]) / 0.5)
            C[i, j] = rho * sigma_obs[i] * sigma_obs[j]

C_inv = np.linalg.inv(C)
c, r_s = 299792.458, 147.5

def joint_chi2(w0, wa, H0, Om):
    Ode = 1.0 - Om
    def E_func(z_):
        f_z = ((1 + z_)**(3 * (1 + w0 + wa))) * np.exp(-3 * wa * z_ / (1 + z_))
        return np.sqrt(Om * (1 + z_)**3 + Ode * f_z)
    
    m = np.array([(c / H0) * quad(lambda z_: 1.0 / E_func(z_), 0, z)[0] / r_s for z in z_obs])
    delta = d_obs - m
    chi2_bao = delta.T @ C_inv @ delta
    chi2_prior = ((H0 - 67.4) / 0.5)**2 + ((Om - 0.315) / 0.007)**2
    return chi2_bao + chi2_prior

# Baseline Lambda-CDM (w0 = -1.0, wa = 0.0, H0 = 67.4, Om = 0.315)
chi2_lcdm = joint_chi2(-1.0, 0.0, 67.4, 0.315)
chi2_dynamic = 25.45 # From your joint optimization output

delta_chi2 = chi2_lcdm - chi2_dynamic

print("=== MODEL COMPARISON SUMMARY ===")
print(f"Standard Lambda-CDM Chi^2 : {chi2_lcdm:.2f}")
print(f"Dynamic Scalar Field Chi^2: {chi2_dynamic:.2f}")
print(f"Delta Chi^2 Improvement   : {delta_chi2:.2f}")

if delta_chi2 > 0:
    print(f"RESULT: Your dynamic vacuum model improves the fit by Delta Chi^2 = {delta_chi2:.2f} over Lambda-CDM.")
else:
    print(f"RESULT: Standard Lambda-CDM provides a better fit by Delta Chi^2 = {abs(delta_chi2):.2f}.")
