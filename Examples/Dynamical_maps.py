##############################
# Python packages & ML model #
##############################

# Packages
import numpy as np 
import matplotlib.pyplot as plt  
import joblib 
from matplotlib.colors import ListedColormap, BoundaryNorm  

# Load the trained XGBoost model
model = joblib.load("xgboost_model.pkl")

###########################
# Parameter Configuration #
###########################

# Settings
omega = 1.34  # Asymmetry parameter
theta = 0.  # Angular orientation
ecc_vals = np.linspace(0., 0.2, 300)  # Eccentricity range
thetadot_vals = np.linspace(-1., 4., 300)  # Angular velocity range
E, T = np.meshgrid(ecc_vals, thetadot_vals)  # Parameter grid

cmap_regimes = ListedColormap(["blue", "red", "green"])  # Colormap for the dynamical regimes
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_regimes.N)  # Define boundaries for the discrete dynamical regimes
class_names = ["resonant", "circulation", "chaos"]  # Class labels
cbar_kwargs = {"orientation": "horizontal", "pad": 0.15, "aspect": 40}  # Colorbar layout
save_kwargs = {"dpi": 300, "bbox_inches": "tight"}  # Figure export settings

###############################
# Generate the dynamical maps #
###############################

grid = np.c_[E.ravel(), T.ravel(), np.full(E.size, omega), np.full(E.size, theta)]
    
# Predict the probability of belonging to each dynamical class
probs = model.predict_proba(grid)
pred_class = np.argmax(probs, axis=1).reshape(E.shape)

# Plot
plt.figure(figsize=(6,5))
img = plt.pcolormesh(T, E, pred_class, cmap=cmap_regimes, norm=norm, shading="auto")
plt.xlabel(r"$\dot{\theta}$", fontsize=16)
plt.ylabel("e", fontsize=16)
    
# Set the colorbar
cbar = plt.colorbar(img, **cbar_kwargs)
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(class_names)
cbar.set_label("Dynamical regime", fontsize=14)
    
plt.tight_layout()
plt.savefig(f"diagram_{omega}.png", **save_kwargs)
plt.close()
