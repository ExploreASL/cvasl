import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from collections import defaultdict

# Define the root directory where your models are stored
root_dir = "."  # Change this to your actual directory

# Identify model folders
model_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

# Dictionary to store images by filename across models
images_by_filename = defaultdict(list)

# Collect PNG file paths grouped by their filename
for model in model_folders:
    model_path = os.path.join(root_dir, model)
    for root, _, files in os.walk(model_path):
        for file in files:
            if file.endswith(".png"):
                images_by_filename[file].append((model, os.path.join(root, file)))

# Create 3x3 grid plots for each unique PNG filename
output_dir = os.path.join(root_dir, "combined_plots")
os.makedirs(output_dir, exist_ok=True)

for filename, image_paths in images_by_filename.items():
    if len(image_paths) < 9:
        continue  # Skip if there are fewer than 9 images

    fig, axes = plt.subplots(3, 3, figsize=(20, 9), dpi=300)

    for ax, (model_name, img_path) in zip(axes.flatten(), image_paths[:9]):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(model_name[0:50], fontsize=8, fontweight='bold', color='lightgreen', pad=1)
        ax.axis("off")
        ax.spines['top'].set_linewidth(2)
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_linewidth(2)
        ax.spines['right'].set_color('black')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"combined_{filename}"))
    plt.close()

# Identify model folders
model_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

# Collect images by filename within each subfolder across models
images_by_subfolder = defaultdict(lambda: defaultdict(list))

for model in model_folders:
    model_path = os.path.join(root_dir, model)
    for subfolder in os.listdir(model_path):
        subfolder_path = os.path.join(model_path, subfolder)
        if os.path.isdir(subfolder_path):
            for root, _, files in os.walk(subfolder_path):
                for file in files:
                    if file.endswith(".png"):
                        images_by_subfolder[subfolder][file].append((model, os.path.join(root, file)))

# Create combined plots for each subfolder
output_root = os.path.join(root_dir, "combined_plots")
os.makedirs(output_root, exist_ok=True)

for subfolder, images_by_filename in images_by_subfolder.items():
    subfolder_output_dir = os.path.join(output_root, subfolder)
    os.makedirs(subfolder_output_dir, exist_ok=True)

    for filename, image_paths in images_by_filename.items():
        if len(image_paths) < 9:
            continue  # Skip if there are fewer than 9 images

        fig, axes = plt.subplots(3, 3, figsize=(20, 9), dpi=300)

        for ax, (model_name, img_path) in zip(axes.flatten(), image_paths[:9]):
            img = mpimg.imread(img_path)
            img_cropped = img
            # # Crop the top 10% of the image
            # crop_height = int(img.shape[0] * 0.1)
            # img_cropped = img[crop_height:, :]
            
            ax.imshow(img_cropped)
            ax.set_title(model_name[0:50], fontsize=8, fontweight='bold', color='lightgreen', pad=1)
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.margins(0)
            for spine in ax.spines.values():
                spine.set_linewidth(1)
                spine.set_color('black')

        plt.tight_layout()
        plt.savefig(os.path.join(subfolder_output_dir, f"combined_{filename}"), bbox_inches='tight', pad_inches=0)
        plt.close()
