import os
import urllib.request
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# 1. Download official Pantheon+ Data Catalog if not already cached
sne_file = '/home/gmarcus/GoogleDrive/QBG/Pantheon+SH0ES.dat'
if not os.path.exists(sne_file):
    print("Downloading Pantheon+ Type Ia Supernova catalog...")
    url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
    urllib.request.urlretrieve(url, sne_file)

# Load SNe data (filter z > 0.01 to eliminate local peculiar velocity noise)
df = pd.read_csv(sne_file, sep=r'\s+', comment='#')
valid = (df['zHD'] > 0.01) & (df['MU_SH0ES'].notnull()) & (df['MU_SH0ES_ERR_DIAG'] > 0)
z_sne = df['zHD'][valid].values
mu_obs = df['MU_SH0ES'][valid].values
sigma_sne = df['MU_SH0ES_ERR_DIAG'][valid].values

# 2. DESI BAO Data Vector & Covariance Matrix
z_bao     = np.array([0.30,  0.51,  0.71,  0.93,  1.32,  1.49,  2.33])
d_bao     = np.array([9.23, 13.62, 16.85, 21.71, 27.79, 30.20, 39.71])
sigma_bao = np.array([0.22,  0.25,  0.30,  0.35,  0.53,  1.10,  0.95])
N_bao = len(z_bao)

C_bao = np.zeros((N_bao, N_bao))
for i in range(N_bao):
    for j in range(N_bao):
        if i == j:
            C_bao[i, j] = sigma_bao[i]**2
        else:
            C_bao[i, j] = 0.20 * np.exp(-abs(z_bao[i] - z_bao[j]) / 0.5) * sigma_bao[i] * sigma_bao[j]
C_bao_inv = np.linalg.inv(C_bao)

c, r_s = 299792.458, 147.5

# 3. High-Performance Cosmological Distance Solver
def compute_chi2(params, is_lcdm=False):
    if is_lcdm:
        H0, Om, M_offset = params
        w0, wa = -1.0, 0.0
    else:
        w0, wa, H0, Om, M_offset = params

    Ode = 1.0 - Om
    def E_func(z_):
        f_z = ((1 + z_)**(3 * (1 + w0 + wa))) * np.exp(-3 * wa * z_ / (1 + z_))
        return np.sqrt(Om * (1 + z_)**3 + Ode * f_z)

    # Grid interpolation for instantaneous vectorized SNe evaluation
    z_grid = np.linspace(0.001, 2.5, 150)
    comoving_grid = np.array([(c / H0) * quad(lambda z_: 1.0 / E_func(z_), 0, z)[0] for z in z_grid])
    
    # Distance Modulus mu(z) = 5*log10(D_L) + 25 + M_offset
    dL_grid = (1.0 + z_grid) * comoving_grid
    mu_grid = 5.0 * np.log10(np.maximum(dL_grid, 1e-5)) + 25.0 + M_offset
    
    interp_mu = interp1d(z_grid, mu_grid, kind='cubic')
    mu_pred = interp_mu(z_sne)
    
    # SNe Likelihood
    chi2_sne = np.sum(((mu_obs - mu_pred) / sigma_sne)**2)

    # BAO Likelihood
    interp_DM = interp1d(z_grid, comoving_grid, kind='cubic')
    d_bao_pred = interp_DM(z_bao) / r_s
    delta_bao = d_bao - d_bao_pred
    chi2_bao = delta_bao.T @ C_bao_inv @ delta_bao

    # Planck CMB Priors
    chi2_prior = ((H0 - 67.4) / 0.5)**2 + ((Om - 0.315) / 0.007)**2

    return chi2_sne + chi2_bao + chi2_prior

# 4. Perform Optimizations
print(f"Loaded {len(z_sne):,} Pantheon+ Type Ia Supernovae.")
print("Running Joint SNe + BAO + Planck Optimization...")

# Model A: Dynamic Vacuum Field
init_phi = [-1.0, 0.0, 67.4, 0.315, 0.0]
bounds_phi = [(-2.0, 0.0), (-3.0, 3.0), (60.0, 75.0), (0.2, 0.4), (-1.0, 1.0)]
res_phi = minimize(lambda p: compute_chi2(p, False), init_phi, method='L-BFGS-B', bounds=bounds_phi)

# Model B: Baseline Static Lambda-CDM
init_lcdm = [67.4, 0.315, 0.0]
bounds_lcdm = [(60.0, 75.0), (0.2, 0.4), (-1.0, 1.0)]
res_lcdm = minimize(lambda p: compute_chi2(p, True), init_lcdm, method='L-BFGS-B', bounds=bounds_lcdm)

w0_f, wa_f, H0_f, Om_f, M_f = res_phi.x
delta_chi2 = res_lcdm.fun - res_phi.fun
sigma_significance = np.sqrt(max(0, delta_chi2))

print("\n=== JOINT SNe Ia + BAO + PLANCK RESULTS ===")
print(f"Lambda-CDM Total Chi^2   : {res_lcdm.fun:.2f}")
print(f"Dynamic Field Total Chi^2: {res_phi.fun:.2f}")
print(f"Delta Chi^2 Improvement   : {delta_chi2:.2f}")
print(f"Statistical Significance : {sigma_significance:.2f} sigma")
print(f"\nBest-fit parameters:")
print(f"w0 = {w0_f:.3f} | wa = {wa_f:.3f} | H0 = {H0_f:.2f} km/s/Mpc | Om = {Om_f:.3f}")
