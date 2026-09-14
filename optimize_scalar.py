import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

# DESI BAO Data Points [z, DM/r_s, error]
data = np.array([
    [0.30,  9.23,  0.22], [0.51, 13.62, 0.25], [0.71, 16.85, 0.30],
    [0.93, 21.71,  0.35], [1.32, 27.79, 0.53], [1.49, 30.20, 1.10], [2.33, 39.71, 0.95]
])
z_obs, DM_obs, sigma_obs = data[:, 0], data[:, 1], data[:, 2]

c, r_s = 299792.458, 147.5

# Define Chi-Square Objective Function
def chi2_func(params):
    w0, wa, H0, Om = params
    Ode = 1.0 - Om
    
    def E_func(z):
        f_z = ((1 + z)**(3 * (1 + w0 + wa))) * np.exp(-3 * wa * z / (1 + z))
        return np.sqrt(Om * (1 + z)**3 + Ode * f_z)
    
    model_preds = []
    for z in z_obs:
        integral, _ = quad(lambda z_: 1.0 / E_func(z_), 0, z)
        model_preds.append((c / H0) * integral / r_s)
        
    return np.sum(((DM_obs - np.array(model_preds)) / sigma_obs)**2)

# Initial guess: [w0, wa, H0, Om]
initial_guess = [-0.82, -0.41, 67.4, 0.315]
bounds = [(-2.0, 0.0), (-3.0, 1.0), (60.0, 75.0), (0.2, 0.4)]

res = minimize(chi2_func, initial_guess, method='L-BFGS-B', bounds=bounds)

print("=== OPTIMIZED SCALAR FIELD FIT ===")
print(f"Best-fit w0 : {res.x[0]:.3f}")
print(f"Best-fit wa : {res.x[1]:.3f}")
print(f"Best-fit H0 : {res.x[2]:.2f} km/s/Mpc")
print(f"Best-fit Om : {res.x[3]:.3f}")
print(f"Min Chi^2   : {res.fun:.2f} (Reduced Chi^2 = {res.fun / (len(z_obs)-4):.3f})")
