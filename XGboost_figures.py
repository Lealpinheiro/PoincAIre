################################ 
# XGBoost Classifier           #
################################
# Tiago Francisco, Giovana Ramon, Lucas Soares, Othon Winter 
# Data: 06/2026
# Python3 
# Read me: This code generate figures for the paper using XGBoost trained model


# ========
# Packages
# ========
import numpy as np
import matplotlib.pyplot as plt
import joblib
import matplotlib.animation as animation
import os

from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.cluster import KMeans

# ==========
# Load model
# ==========
model = joblib.load("xgboost_model.pkl")

# ========
# Settings
# ========
omega_list = [1.34]
theta_sel = 0.

ecc_vals = np.linspace(0., 0.2, 300)
thetadot_vals = np.linspace(-1., 4., 300)

E, T = np.meshgrid(ecc_vals, thetadot_vals)

# Class color map
cmap_regimes = ListedColormap(["blue", "red", "green"])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_regimes.N)

# Class labels 
class_names = ["resonance", "circularization", "chaos"]

# Common figure parameters
fig_size = (6, 5)
cbar_kwargs = {"orientation": "horizontal", "pad": 0.15, "aspect": 40}
save_kwargs = {"dpi": 300, "bbox_inches": "tight"}

# ===============
# Loop over omega
# ===============
for omega_sel in omega_list:

    print(f"Omega = {omega_sel}")

    grid = np.c_[E.ravel(), T.ravel(), np.full(E.size, omega_sel), np.full(E.size, theta_sel)]

    # Predict probabilities
    probs = model.predict_proba(grid)

    # reshape to 2D maps (each class)
    P_res = probs[:, 0].reshape(E.shape)
    P_circ = probs[:, 1].reshape(E.shape)
    P_chaos = probs[:, 2].reshape(E.shape)

    # Dynamical map
    pred_class = np.argmax(probs, axis=1).reshape(E.shape)
    
    plt.figure(figsize=fig_size)

    img = plt.pcolormesh(T, E, pred_class, cmap=cmap_regimes, norm=norm, shading="auto")

    # Add horizontal gray lines 
    target_e_vals = [0.02, 0.06, 0.10, 0.14]
    for e in target_e_vals:
        plt.axhline(y=e, color="black", linestyle="-", linewidth=2, alpha=0.8)
        plt.text(x=-0.9, y=e + 0.003, s=f"e = {e}", color="black", fontsize=8, weight="bold")

    # Box and write the resonances
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75)
    annotations = [
        (-0.2, 0.125, r"1:1$\alpha$"),
        (2.0, 0.16,  r"1:1$\beta$"),
        (2.5, 0.003,  r"2:1"),
        (2.82, 0.01, r"5:2"), 
        (3.3, 0.04, r"3:1"),
        (3.75, 0.06, r"7:2")
    ]   
    for x_pos, y_pos, label_text in annotations:
        plt.text(x_pos, y_pos, label_text, color="black", fontsize=8, 
                 weight="bold", ha="center", va="center", bbox=bbox_props)
                  
    plt.xlabel(r"$\dot{\theta}$", fontsize=16)
    plt.ylabel("e", fontsize=16)
    
    cbar = plt.colorbar(img, **cbar_kwargs)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(class_names)
    cbar.set_label("Dynamical regime", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"diagram_{omega_sel}.png", **save_kwargs)
    plt.close()

    # Entropy map
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    # maximum value of entropy in the case of 3 class
    max_entropy = np.log(3)  
    entropy_normalized = entropy / max_entropy
    entropy_map = entropy_normalized.reshape(E.shape)

    plt.figure(figsize=fig_size)

    img = plt.pcolormesh(T, E, entropy_map, shading="auto", cmap="inferno", vmin=0, vmax=1)

    cbar = plt.colorbar(img, **cbar_kwargs)
    cbar.set_label("Entropy", fontsize=14)
  
    plt.xlabel(r"$\dot{\theta}$", fontsize=16)
    plt.ylabel("e", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"uncertainty_{omega_sel}.png", **save_kwargs)
    plt.close()
    
    # Probability map
    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

    maps = [P_res, P_circ, P_chaos]

    for i, ax in enumerate(axes):
        im = ax.pcolormesh(T, E, maps[i], shading="auto",cmap="viridis",vmin=0,vmax=1)

        ax.set_ylabel("e")

        if i == len(axes) - 1:
            ax.set_xlabel(r"$\dot{\theta}$")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"Probability ({class_names[i]})")

    plt.tight_layout()
    plt.savefig(f"probability_{omega_sel}.png")
    plt.close()
    
# ===============
# GIF
# ===============
omega_vals = np.linspace(0.3, 1.7, 50)

theta_sel = 0.

ecc_vals = np.linspace(0.0, 0.2, 300)
thetadot_vals = np.linspace(-1.0, 4.0, 300)

E, T = np.meshgrid(ecc_vals, thetadot_vals)

maps = []
blue_masks = []

# =======================
# Generate all omega maps
# =======================
for omega_sel in omega_vals:

    grid = np.c_[
        E.ravel(),
        T.ravel(),
        np.full(E.size, omega_sel),
        np.full(E.size, theta_sel)
    ]

    probs = model.predict_proba(grid)
    pred_class = np.argmax(probs, axis=1).reshape(E.shape)

    maps.append(pred_class)

    # reference class
    blue_masks.append(pred_class == 0)

maps = np.array(maps)
blue_masks = np.array(blue_masks)


output_dir = "omega_frames"
os.makedirs(output_dir, exist_ok=True)

for i, omega_sel in enumerate(omega_vals):

    fig, ax = plt.subplots(figsize=(6,5))

    im = ax.pcolormesh(
        T, E, maps[i],
        cmap=ListedColormap(["blue", "red", "green"]),
        shading="auto"
    )

    ax.set_title(f"omega = {omega_sel:.3f}")
    ax.set_xlabel(r"$\dot{\theta}$")
    ax.set_ylabel("e")

    filename = os.path.join(output_dir, f"omega_{omega_sel:.5f}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
# ===============
# Animation
# ===============
fig, ax = plt.subplots(figsize=(6,5))

def update(i):
    ax.clear()

    ax.pcolormesh(
        T, E, maps[i],
        cmap=ListedColormap(["blue", "red", "green"]),
        shading="auto"
    )

    ax.set_title(f"omega = {omega_vals[i]:.3f}")
    ax.set_xlabel(r"$\dot{\theta}$")
    ax.set_ylabel("e")

ani = animation.FuncAnimation(fig, update, frames=len(omega_vals))

ani.save("omega_evolution.gif", fps=5)

plt.close()

# ===============
# Mossaic
# ===============
omega_list = [0.3, 0.4, 0.5, 0.6, 0.7,
              0.9, 1.0, 1.1, 1.2, 1.3,
              1.34, 1.4, 1.5, 1.6, 1.7]

theta_sel = 0.0

ecc_vals = np.linspace(0.0, 0.2, 300)
thetadot_vals = np.linspace(-1.0, 4.0, 300)

E, T = np.meshgrid(ecc_vals, thetadot_vals)

cmap_regimes = ListedColormap(["blue", "red", "green"])

# Set figure
nrows, ncols = 5, 3

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(10, 12),
    sharex=True,
    sharey=True
)

axes = axes.ravel()

# Loop over omega
for i, omega_sel in enumerate(omega_list):

    grid = np.c_[
        E.ravel(),
        T.ravel(),
        np.full(E.size, omega_sel),
        np.full(E.size, theta_sel)
    ]

    probs = model.predict_proba(grid)
    pred_class = np.argmax(probs, axis=1).reshape(E.shape)

    ax = axes[i]

    ax.pcolormesh(
        T, E, pred_class,
        cmap=cmap_regimes,
        shading="auto"
    )

    # Label box
    ax.text(
        0.03, 0.93,
        rf"$\omega = {omega_sel}$",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="square", facecolor="white", edgecolor="black", pad=0.2)
    )

# setting the axis labels-
for ax in axes:
    ax.label_outer() 

for i, ax in enumerate(axes):
    row = i // ncols
    col = i % ncols

    if col == 0:
        ax.tick_params(axis='y', labelleft=True)

    if row == nrows - 1:
        ax.tick_params(axis='x', labelbottom=True)

fig.supxlabel(r"$\dot{\theta}$", fontsize=14)
fig.supylabel("e", fontsize=14)

# set the panels configuration
plt.subplots_adjust(
    left=0.10,
    right=0.98,
    top=0.98,
    bottom=0.10,
    wspace=0.06,
    hspace=0.08
)
plt.savefig("mosaic_omega.png", dpi=300)
