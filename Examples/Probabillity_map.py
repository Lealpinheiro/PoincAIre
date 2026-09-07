##############################
# Python packages & ML model #
##############################

# Packages
import numpy as np 
import matplotlib.pyplot as plt  
import joblib 

# Load the trained XGBoost model
model = joblib.load("xgboost_model.pkl")

###########################
# Parameter Configuration #
###########################

# Settings
omega = 0.8  # Asymmetry parameter
theta = 0.  # Angular orientation
ecc_vals = np.linspace(0., 0.2, 300)  # Eccentricity range
thetadot_vals = np.linspace(-1., 4., 300)  # Angular velocity range
E, T = np.meshgrid(ecc_vals, thetadot_vals)  # Parameter grid

grid = np.c_[E.ravel(), T.ravel(), np.full(E.size, omega), np.full(E.size, theta)]
    
# Probabilities for each dynamical class
probs = model.predict_proba(grid)
P_res = probs[:, 0].reshape(E.shape)    # Resonant class
P_circ = probs[:, 1].reshape(E.shape)   # Circulation class
P_chaos = probs[:, 2].reshape(E.shape)  # Chaos class

#################################
# Generate the Probability maps #
#################################

# Plot
fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
maps = [P_res, P_circ, P_chaos]  # Data array collection
class_names = ["resonant", "circulation", "chaos"]

for i, ax in enumerate(axes):

    im = ax.pcolormesh(T, E, maps[i], shading="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_ylabel("e")
    
    if i == len(axes) - 1:
        ax.set_xlabel(r"$\dot{\theta}$")
        
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Probability ({class_names[i]})")
        
plt.tight_layout()
plt.savefig(f"probability_{omega}.png")
plt.close()                                                                              