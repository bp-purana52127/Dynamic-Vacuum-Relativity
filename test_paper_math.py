import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from astropy.io import fits
from astropy.table import Table

# 1. Load the 1.42 million galaxy redshifts processed earlier
desi_path = '/home/gmarcus/GoogleDrive/QBG/DESI/zall-pix-fuji.fits'
print("Loading galaxy catalog...")
with fits.open(desi_path, memmap=True) as hdul:
    data = Table(hdul[1].data)

valid_mask = (data['Z'] > 0) & (data['Z'] < 3.5) & (data['ZWARN'] == 0)
redshifts = data['Z'][valid_mask]

# 2. Cosmological Parameters (H0 = 67.4 km/s/Mpc, Omega_m = 0.315, Omega_de = 0.685)
H0 = 67.4
Om = 0.315
Ode = 0.685

# Model A: Standard Static Lambda-CDM (w0 = -1.0, wa = 0.0)
def E_lcdm(z):
    return np.sqrt(Om * (1 + z)**3 + Ode)

# Model B: Paper's Dynamic Vacuum Field Phi(x) (w0 = -0.82, wa = -0.41 as favored by DESI)
def E_dynamic_phi(z, w0=-0.82, wa=-0.41):
    f_z = ((1 + z)**(3 * (1 + w0 + wa))) * np.exp(-3 * wa * z / (1 + z))
    return np.sqrt(Om * (1 + z)**3 + Ode * f_z)

# 3. Compute Comoving Distance D_M(z) for both models
z_grid = np.linspace(0.01, 3.5, 100)
c = 299792.458 # Speed of light in km/s

D_M_lcdm = [ (c / H0) * quad(lambda z_: 1.0 / E_lcdm(z_), 0, z)[0] for z in z_grid ]
D_M_phi  = [ (c / H0) * quad(lambda z_: 1.0 / E_dynamic_phi(z_), 0, z)[0] for z in z_grid ]

# Percentage deviation in galaxy distances predicted by your scalar field
pct_diff = 100 * (np.array(D_M_phi) - np.array(D_M_lcdm)) / np.array(D_M_lcdm)

# 4. Plot DESI galaxy counts against the mathematical divergence of your paper
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Top Panel: DESI Redshift Histogram
ax1.hist(redshifts, bins=100, color='indigo', alpha=0.6, label='DESI Galaxy Sample (1.42M Objects)')
ax1.set_ylabel('Galaxy Count')
ax1.set_title('Testing Dynamic Vacuum Potential $V(\\Phi)$ against DESI Redshifts')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)

# Bottom Panel: Mathematical Divergence (Dynamic Vacuum vs. Static Lambda)
ax2.plot(z_grid, pct_diff, color='crimson', lw=2.5, label='$\Delta D_M$: Dynamic Vacuum $\Phi(x^\mu)$ vs. Static $\Lambda$')
ax2.axhline(0, color='black', linestyle=':', alpha=0.7)
ax2.set_xlabel('Redshift (z)')
ax2.set_ylabel('% Distance Shift')
ax2.legend(loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
output_file = '/home/gmarcus/QBG/paper_math_vs_desi.png'
plt.savefig(output_file)
print(f"SUCCESS! Mathematical comparison plotted and saved to: {output_file}")
