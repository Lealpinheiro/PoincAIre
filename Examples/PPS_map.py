########################################
#              PPS maps                #
########################################
# Tiago Francisco, Giovana Ramon, Lucas Soares e Othon Winter
# Data: 18/01/2024
# Python3
''' Read me: This example demonstrates how to perform visual classification 
of Poincaré surface-of-section (PSS) maps using a convolutional neural network (CNN) model. 
The script loads an input file containing four columns corresponding to time, true anomaly, 
angular parameter (theta and theta_dot). 
The data are obtained from numerical integrations performed for 40 different 
initial conditions, with 1,000 PSS points generated for each initial condition. 
Thus, the input file contains a total of 40,000 lines. 
The data are organized sequentially, with the first 1,000 lines corresponding 
to the first initial condition, the next 1,000 lines corresponding to the 
second initial condition, and so on until all 40 initial conditions have been processed. 
For each initial condition, the corresponding points are extracted and 
converted into an image representation of the Poincaré surface of section. 
These images are then provided as input to the trained CNN model, 
which predicts the dynamical class of each initial condition: resonant, circulation, or chaos.'''


##############################
# Python packages & ML model #
##############################

# Packages
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from tensorflow.keras.models import load_model
from matplotlib.lines import Line2D

# Path to the trained CNN model
model_path = "poincare_classifier.keras"

###########################
# Parameter Configuration #
###########################

# Asymmetry parameter
omega_value = 0.3
# Eccentricity parameter
eccentricity_value = 0.2
# Number of initial conditions in the input file
n_ic = 40
# Number of points per curve
n_points_per_curve = 1000

# Image size
img_size = (128, 128)
# Class names
class_names = ["chaos", "resonant", "circulation"]
# Color assigned to each class
color_map = {"resonant": "blue", "circulation": "red", "chaos": "green"}
# Input file
target_files = ["example.txt"]

model = load_model(model_path)

###########################
#       Functions         #
###########################

# Normalize theta to the interval [-pi/2, pi/2]
def normalize_theta(theta):
    return ((theta + np.pi/2) % np.pi) - np.pi/2

# Create a figure representing the Poincaré surface of section
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

#################################################
# Generate the Poincaré surface of section maps #
#################################################

for file in target_files:

    ecc_data = np.loadtxt(file)
    
    # Lists to store the curve data and their predicted class labels
    curves_data = []
    batch_imgs = []

    # Extract the data and prepare the CNN inputs for all curves in the input file
    for ci in range(n_ic):
        
        start = ci * (n_points_per_curve + 1)
        end   = start + n_points_per_curve + 1
        curve = ecc_data[start:end]

        theta_vals = normalize_theta(curve[:, 2])
        theta_dot_vals = curve[:, 3]
        
        curves_data.append((theta_vals, theta_dot_vals))

        # Generate an image representation for each curve
        img = figure_to_input(theta_vals, theta_dot_vals)
        batch_imgs.append(img[0])

    # Predict the class probabilities for all curves
    batch_imgs = np.array(batch_imgs)
    probs_batch = model.predict(batch_imgs, verbose=0)
    
    # Create the final Poincaré surface of section plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    for ci in range(n_ic):
        theta_vals, theta_dot_vals = curves_data[ci]
        probs = probs_batch[ci]
        
        # Get the class with the highest predicted probability
        pred_idx = np.argmax(probs)
        pred_class = class_names[pred_idx]
        plot_color = color_map[pred_class]

        # Plot the trajectory using the color assigned to its predicted class
        ax.scatter(theta_vals, theta_dot_vals, s=2, c=plot_color, alpha=0.7)
            
    ax.set_title(rf"$\omega = {omega_value:.2f}$, $e = {eccentricity_value:.3f}$", fontsize=14)
    
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel(r"$\dot{\theta}$", fontsize=12)

    # Create custom legend handles
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8)]

    plt.savefig("plot_example.png", bbox_inches='tight')
    plt.close(fig)