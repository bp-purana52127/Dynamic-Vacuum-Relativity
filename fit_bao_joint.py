import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

# 1. DESI BAO Data Vector [z_obs, DM/r_s, 1D_sigma]
z_obs     = np.array([0.30,  0.51,  0.71,  0.93,  1.32,  1.49,  2.33])
d_obs     = np.array([9.23, 13.62, 16.85, 21.71, 27.79, 30.20, 39.71])
sigma_obs = np.array([0.22,  0.25,  0.30,  0.35,  0.53,  1.10,  0.95])
N = len(z_obs)

# 2. Covariance Matrix C with off-diagonal bin correlations
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

def model_DM_rs(z, w0, wa, H0, Om):
    Ode = 1.0 - Om
    def E_func(z_):
        f_z = ((1 + z_)**(3 * (1 + w0 + wa))) * np.exp(-3 * wa * z_ / (1 + z_))
        return np.sqrt(Om * (1 + z_)**3 + Ode * f_z)
    
    integral, _ = quad(lambda z_: 1.0 / E_func(z_), 0, z)
    return (c / H0) * integral / r_s

# 3. Joint Chi-Square Objective (BAO + Planck CMB Priors)
def joint_chi2(params):
    w0, wa, H0, Om = params
    
    # BAO Likelihood
    m = np.array([model_DM_rs(z, w0, wa, H0, Om) for z in z_obs])
    delta = d_obs - m
    chi2_bao = delta.T @ C_inv @ delta
    
    # Planck CMB Gaussian Priors: H0 = 67.4 +/- 0.5, Om = 0.315 +/- 0.007
    chi2_prior = ((H0 - 67.4) / 0.5)**2 + ((Om - 0.315) / 0.007)**2
    
    return chi2_bao + chi2_prior

# 4. Run Optimization with Realistic Dark Energy Bounds
initial_guess = [-0.82, -0.41, 67.4, 0.315]
bounds = [(-1.5, -0.5), (-2.0, 1.0), (60.0, 75.0), (0.25, 0.40)]

res = minimize(joint_chi2, initial_guess, method='L-BFGS-B', bounds=bounds)

w0_fit, wa_fit, H0_fit, Om_fit = res.x

print("\n=== JOINT PLANCK + DESI BAO OPTIMIZATION ===")
print(f"Optimized w0   : {w0_fit:.3f}")
print(f"Optimized wa   : {wa_fit:.3f}")
print(f"Optimized H0   : {H0_fit:.2f} km/s/Mpc")
print(f"Optimized Om   : {Om_fit:.3f}")
print(f"Total Chi^2    : {res.fun:.2f}")
print(f"Reduced Chi^2  : {res.fun / (N - 2):.3f}")
