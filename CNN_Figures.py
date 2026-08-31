import matplotlib
# Use "Agg" if running headlessly (no display), or comment out to display interactively
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model

# -----------------------
# Input configuration
# -----------------------
model_path = "poincare_classifier.keras"
n_points_per_curve = 1001
img_size = (128, 128)

# Class names mapping (alphabetical order from your original script)
# Index 0: chaos, Index 1: close, Index 2: open
class_names = ["chaos", "close", "open"]

# Color mapping requested by user
color_map = {
    "close": "blue",
    "open": "red",
    "chaos": "green"
}

n_ic = 40
target_files = ["ecc_00000.txt",  "ecc_00550.txt",  "ecc_01100.txt",  "ecc_01650.txt",
                "ecc_00050.txt",  "ecc_00600.txt",  "ecc_01150.txt",  "ecc_01700.txt",
                "ecc_00100.txt",  "ecc_00650.txt",  "ecc_01200.txt",  "ecc_01750.txt",
                "ecc_00150.txt",  "ecc_00700.txt",  "ecc_01250.txt",  "ecc_01800.txt",
                "ecc_00200.txt",  "ecc_00750.txt",  "ecc_01300.txt",  "ecc_01850.txt",
                "ecc_00250.txt",  "ecc_00800.txt",  "ecc_01350.txt",  "ecc_01900.txt",
                "ecc_00300.txt",  "ecc_00850.txt",  "ecc_01400.txt",  "ecc_01950.txt",
                "ecc_00350.txt",  "ecc_00900.txt",  "ecc_01450.txt",  "ecc_02000.txt",
                "ecc_00400.txt",  "ecc_00950.txt",  "ecc_01500.txt",  "ecc_00450.txt",  
                "ecc_01000.txt",  "ecc_01550.txt",  "ecc_00500.txt",  "ecc_01050.txt",  
                "ecc_01600.txt"]


# -----------------------
# Load ML model
# -----------------------
print("Loading model...")
model = load_model(model_path)

# -----------------------
# Helper Functions
# -----------------------
def normalize_theta(theta):
    return ((theta + np.pi/2) % np.pi) - np.pi/2

def figure_to_input(theta, theta_dot, img_size=(128,128), dpi=150):
    fig, ax = plt.subplots(figsize=(3,3), dpi=dpi)
    ax.scatter(theta, theta_dot, s=1, c="black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    img = Image.fromarray(img).resize(img_size)
    img = np.array(img).astype("float32") / 255.
    img = np.expand_dims(img, axis=0)
    return img

omega_value = 0.3

# -----------------------
# Main Processing & Plotting
# -----------------------
for ecc_file in target_files:
    if not os.path.exists(ecc_file):
        print(f"File not found: {ecc_file}, skipping...")
        continue

    print(f"\nProcessing file: {ecc_file}")
    ecc_data = np.loadtxt(ecc_file)
    
    # Extract eccentricity from filename 
    # Example: ecc_00550.txt -> e = 0.0550 
    ecc_string = os.path.splitext(ecc_file)[0].replace("ecc_", "") 
    eccentricity_value = float(ecc_string) / 10000

    # Initialize lists to store curve data and their classified labels
    curves_data = []
    batch_imgs = []

    # 1. Extract data and prepare CNN inputs for all curves in the file
    for ci in range(n_ic):
        start = ci * n_points_per_curve
        end   = start + n_points_per_curve
        curve = ecc_data[start:end]

        theta_vals = normalize_theta(curve[:, 2])
        theta_dot_vals = curve[:, 3]
        
        curves_data.append((theta_vals, theta_dot_vals))

        # Generate image representing this single curve for the CNN
        img = figure_to_input(theta_vals, theta_dot_vals)
        batch_imgs.append(img[0])

    # 2. Predict classes in batch
    batch_imgs = np.array(batch_imgs)
    probs_batch = model.predict(batch_imgs, verbose=0)
    
    # 3. Create the final Poincaré Plot for this file
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    for ci in range(n_ic):
        theta_vals, theta_dot_vals = curves_data[ci]
        probs = probs_batch[ci]
        
        # Get the class with the highest probability
        pred_idx = np.argmax(probs)
        pred_class = class_names[pred_idx]
        plot_color = color_map[pred_class]

        # Plot this individual trajectory
        ax.scatter(theta_vals, theta_dot_vals, s=2, c=plot_color, alpha=0.7)
        
    # --- Added green vertical line at theta = 0 ---
    #ax.axvline(x=0, color='gray', linestyle='-', linewidth=2, zorder=1)
    
    ax.set_title(rf"$\omega = {omega_value:.2f}$, $e = {eccentricity_value:.3f}$", fontsize=14)
    
    # Decorate and save the surface of section plot
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel(r"$\dot{\theta}$", fontsize=12)

    # Create custom legend handles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8)
    ]

    # Save output plot image
    output_filename = f"plot_{os.path.splitext(ecc_file)[0]}.png"
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
