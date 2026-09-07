########################################
#           Entropy maps               #
########################################
# Tiago Francisco, Giovana Ramon, Lucas Soares e Othon Winter
# Data: 18/01/2024
# Python3
# Read me: This example demonstrates how to quantify classification 
# uncertainty using normalized Shannon entropy.

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

# Predict the probability of belonging to each dynamical class
probs = model.predict_proba(grid)
    
# Calculate normalized Shannon Entropy for a 3 class 
eps = 1e-12  # Threshold to avoid log(0) error
entropy = -np.sum(probs * np.log(probs + eps), axis=1)  # Entropy sum
max_entropy = np.log(3) 
entropy_normalized = entropy / max_entropy  
entropy_map = entropy_normalized.reshape(E.shape) 
cbar_kwargs = {"orientation": "horizontal", "pad": 0.15, "aspect": 40}
save_kwargs = {"dpi": 300, "bbox_inches": "tight"}

#############################
# Generate the Entropy maps #
#############################

grid = np.c_[E.ravel(), T.ravel(), np.full(E.size, omega), np.full(E.size, theta)]
    
# Predict the probability of belonging to each dynamical class
probs = model.predict_proba(grid)
pred_class = np.argmax(probs, axis=1).reshape(E.shape)

# Plot
plt.figure(figsize=(6,5))
img = plt.pcolormesh(T, E, entropy_map, shading="auto", cmap="inferno", vmin=0, vmax=1)
    
# Set the colorbar
cbar = plt.colorbar(img, **cbar_kwargs)
cbar.set_label("Entropy", fontsize=14)
plt.xlabel(r"$\dot{\theta}$", fontsize=16)
plt.ylabel("e", fontsize=16)
    
plt.tight_layout()
plt.savefig(f"uncertainty_{omega}.png", **save_kwargs)
plt.close()                                                                                    