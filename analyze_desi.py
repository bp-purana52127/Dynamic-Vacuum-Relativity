import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

print("Loading DESI 2.0 GB dataset from Google Drive...")
desi_path = '/home/gmarcus/GoogleDrive/QBG/DESI/zall-pix-fuji.fits'

# Memory-map the FITS table to avoid crashing RAM
with fits.open(desi_path, memmap=True) as hdul:
    data = Table(hdul[1].data)

print("Filtering valid galactic spectroscopic redshifts...")
valid_mask = (data['Z'] > 0) & (data['Z'] < 4.0) & (data['ZWARN'] == 0)
redshifts = data['Z'][valid_mask]

print("Generating redshift distribution plot...")
plt.figure(figsize=(10, 5))
plt.hist(redshifts, bins=100, color='crimson', histtype='stepfilled', alpha=0.7)
plt.title('DESI EDR Galaxy & QSO Redshift Distribution N(z)')
plt.xlabel('Redshift (z)')
plt.ylabel('Object Count')
plt.grid(True, linestyle='--', alpha=0.5)

output_img = '/home/gmarcus/QBG/desi_redshift_distribution.png'
plt.savefig(output_img)
print(f"SUCCESS! Processed {len(redshifts):,} galaxies.")
print(f"Plot saved directly to disk at: {output_img}")
