import os
import pandas as pd
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, XGradCAM, AblationCAM, LayerCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import logging
import csv
from tqdm import tqdm
import re
import gc
import argparse
from math import ceil
torch.backends.cudnn.benchmark = True  # Optimizes CUDA operations
torch.cuda.empty_cache()
import traceback
import cv2
import scipy
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    XGradCAM,
    AblationCAM,
    EigenGradCAM,
    LayerCAM
)
import matplotlib.colors as mcolors
from models.cnn import Large3DCNN
from models.densenet3d import DenseNet3D
from models.efficientnet3d import EfficientNet3D
from models.improvedcnn3d import Improved3DCNN
from models.resnet3d import ResNet3D
from models.resnext3d import ResNeXt3D
from data import BrainAgeDataset
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import pearsonr
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set a seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)




class BrainAgeWrapper(torch.nn.Module):
    """Wrapper class to handle the demographic input"""
    def __init__(self, model, demographics):
        super().__init__()
        self.model = model
        self.demographics = demographics

    def forward(self, x):
        return self.model(x, self.demographics)

def create_visualization_dirs(base_output_dir, methods_to_run):
    """Create directories for specified visualization methods"""
    all_methods = {
        'gradcam': GradCAM,
        'hirescam': HiResCAM,
        'gradcam++': GradCAMPlusPlus,
        'xgradcam': XGradCAM,
        'eigencam': EigenGradCAM,
        'layercam': LayerCAM
    }

    # Filter methods if specific ones are requested
    if 'all' not in methods_to_run:
        all_methods = {k: v for k, v in all_methods.items() if k in methods_to_run}

    for method_name in all_methods.keys():
        method_dir = os.path.join(base_output_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)

    return all_methods

def get_target_layers(model):
    """Get the target layers for visualization based on model type."""
    model_name = model.model.__class__.__name__ # Access the wrapped model

    if model_name == 'Large3DCNN':
        return [model.model.conv_layers[-1]]  # Last Conv3d layer
    elif model_name == 'DenseNet3D':
        return [model.model.trans2[1]] # Transition layer before last avg pool
    elif model_name == 'EfficientNet3D':
        return [model.model.conv_head] # Head convolution before avg pool
    elif model_name == 'Improved3DCNN':
        # Access the last layer in the sequential conv_layers
        if isinstance(model.model.conv_layers[-1], nn.MaxPool3d): # check if last layer is pool
            return [model.model.conv_layers[-5]] # Target the conv layer before pool and relu and SE block
        else:
            return [model.model.conv_layers[-2]] # Target the conv layer before relu and SE block
    elif model_name == 'ResNet3D':
        return [model.model.layer3[-1].conv2] # Last conv layer in last ResNet block of layer3
    elif model_name == 'ResNeXt3D':
        return [model.model.layer3[-1].conv3] # Last conv layer in last ResNeXt block of layer3
    else:
        return None # Default or unknown model type

def load_atlas(atlas_path):
    """Loads a NIfTI atlas and returns the data and header."""
    try:
        atlas_nii = nib.load(atlas_path)
        atlas_data = atlas_nii.get_fdata()
        return atlas_data, atlas_nii.affine
    except Exception as e:
        tb_str = traceback.format_exc()
        logging.error(f"Error loading atlas from {atlas_path}: {e}\nTraceback:\n{tb_str}")
        return None, None

def calculate_regional_intensity(heatmap, atlas_data_resampled):
    """Calculates average heatmap intensity for each region in the atlas."""
    regional_intensities = {}
    unique_region_labels = np.unique(atlas_data_resampled)[1:]

    for region_label in unique_region_labels:
        mask = (atlas_data_resampled == region_label)
        regional_heatmap_values = heatmap[mask]
        if regional_heatmap_values.size > 0: # Handle empty regions
            regional_intensities[int(region_label)] = np.mean(regional_heatmap_values) # Use int for keys
        else:
            regional_intensities[int(region_label)] = np.nan # or some default value

    return regional_intensities

def plot_regional_intensity_distributions(regional_intensity_data, output_dir, method_name, prefix=""):
    """Plots boxplots of heatmap intensity distributions per region and method."""
    df = pd.DataFrame(regional_intensity_data)
    df_melted = pd.melt(df, var_name='region', value_name='intensity') # Reshape for seaborn

    plt.figure(figsize=(12, 6)) # Adjust figure size as needed
    sns.boxplot(x='region', y='intensity', data=df_melted) # Or sns.violinplot
    plt.title(f'{prefix}Heatmap Intensity Distribution per Region - {method_name}')
    plt.xlabel('Atlas Region Label')
    plt.ylabel('Heatmap Intensity')
    plt.xticks(rotation=90) # Rotate x-axis labels if needed
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}{method_name}_regional_intensity_boxplots.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
def plot_average_regional_intensity_barchart(average_regional_intensities, output_dir, method_name, prefix=""):
    """Plots bar chart of average heatmap intensity per region and method."""

    if type(average_regional_intensities) is list:
        average_regional_intensities = average_regional_intensities[0] # Take the first element if it's a list
    regions = list(average_regional_intensities.keys())
    intensities = list(average_regional_intensities.values())

    plt.figure(figsize=(12, 6)) # Adjust figure size as needed
    plt.bar(regions, intensities)
    plt.title(f'{prefix}Average Heatmap Intensity per Region - {method_name}')
    plt.xlabel('Atlas Region Label')
    plt.ylabel('Average Heatmap Intensity')
    plt.xticks(regions, rotation=90) # Ensure x-ticks are region labels and rotate
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}{method_name}_average_regional_intensity_barchart.png"), bbox_inches='tight', dpi=300)
    plt.close()

def calculate_and_plot_regional_correlation(regional_intensity_age_data, output_dir, method_name, prefix=""):
    """Calculates and plots correlation between regional heatmap intensity and age."""
    correlation_results = {}
    for region_label in regional_intensity_age_data:
        intensities = regional_intensity_age_data[region_label]['intensities']
        ages = regional_intensity_age_data[region_label]['ages']
        if len(intensities) > 1: # Need at least 2 points for correlation
            corr, p_value = pearsonr(intensities, ages)
            correlation_results[region_label] = {'correlation': corr, 'p_value': p_value}
        else:
            correlation_results[region_label] = {'correlation': np.nan, 'p_value': np.nan} # Handle cases with insufficient data

    # --- Scatter Plots ---
    for region_label in regional_intensity_age_data:
        intensities = regional_intensity_age_data[region_label]['intensities']
        ages = regional_intensity_age_data[region_label]['ages']

        if len(intensities) > 1: # Plot only if there are data points
            plt.figure(figsize=(8, 6))
            plt.scatter(ages, intensities)
            corr_val = correlation_results[region_label]['correlation']
            plt.title(f'Region {region_label} - Intensity vs. Age (r={corr_val:.2f}) - {method_name}')
            plt.xlabel('Age')
            plt.ylabel('Average Heatmap Intensity')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}{method_name}_region_{region_label}_scatter.png"), bbox_inches='tight', dpi=300)
            plt.close()

    # --- (Optional) Brain Map Visualization of Correlations ---
    # Creating a brain map directly in 3D is complex and beyond a quick snippet.
    # You would need to map correlation values back onto the atlas volume and visualize it.
    # For a simpler representation, you could just print the correlation values for each region.
    print(f"\nCorrelation of Heatmap Intensity with Age - {method_name}:")
    for region_label, results in correlation_results.items():
        print(f"Region {region_label}: Correlation = {results['correlation']:.3f}, p-value = {results['p_value']:.3f}")
        

def normalize_cam(cam, target_size=None):
    """Memory-efficient normalization and resizing"""
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7) # Small value for numerical stability

    if target_size is not None: # Resize if target_size is provided
        cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR) # Use cv2.resize
        return cam_resized.astype(np.float16)  # Use float16 for memory efficiency
    return cam.astype(np.float16)



def plot_slices_with_overlay(image, cam, slice_indices=None, alpha=0.5, actual_age=None, predicted_age=None):
     """Plot original and overlay images for all three anatomical planes"""
     if slice_indices is None:
         # Take middle slices if not specified
         slice_indices = {
             'axial': image.shape[0] // 2,
             'coronal': image.shape[1] // 2,
             'sagittal': image.shape[2] // 2
         }

     fig, axes = plt.subplots(2, 3, figsize=(15, 10))

     # Normalize image for visualization
     image_norm = (image - image.min()) / (image.max() - image.min())
     cam_norm = normalize_cam(cam)

     # Axial view
     axes[0, 0].imshow(image_norm[slice_indices['axial'], :, :], cmap='gray')
     axes[1, 0].imshow(image_norm[slice_indices['axial'], :, :], cmap='gray')
     im = axes[1, 0].imshow(cam_norm[slice_indices['axial'], :, :], cmap='jet', alpha=alpha)
     axes[0, 0].set_title('Axial - Original')
     axes[1, 0].set_title('Axial - Overlay')
     fig.colorbar(im, ax=axes[1,0], label="Heatmap Intensity")

     # Coronal view
     axes[0, 1].imshow(image_norm[:, slice_indices['coronal'], :], cmap='gray')
     axes[1, 1].imshow(image_norm[:, slice_indices['coronal'], :], cmap='gray')
     im = axes[1, 1].imshow(cam_norm[:, slice_indices['coronal'], :], cmap='jet', alpha=alpha)
     axes[0, 1].set_title('Coronal - Original')
     axes[1, 1].set_title('Coronal - Overlay')
     fig.colorbar(im, ax=axes[1,1], label="Heatmap Intensity")


     # Sagittal view
     axes[0, 2].imshow(image_norm[:, :, slice_indices['sagittal']], cmap='gray')
     axes[1, 2].imshow(image_norm[:, :, slice_indices['sagittal']], cmap='gray')
     im = axes[1, 2].imshow(cam_norm[:, :, slice_indices['sagittal']], cmap='jet', alpha=alpha)
     axes[0, 2].set_title('Sagittal - Original')
     axes[1, 2].set_title('Sagittal - Overlay')
     fig.colorbar(im, ax=axes[1,2], label="Heatmap Intensity")


     for ax in axes.flat:
         ax.axis('off')

     # Add age information as text above the plots
     if actual_age is not None and predicted_age is not None:
         age_text = f'Actual Age: {actual_age:.1f} years\nPredicted Age: {predicted_age:.1f} years'
         fig.suptitle(age_text, y=0.98, fontsize=12)

     plt.tight_layout()
     return fig

def plot_all_slices(image, cam, plane='axial', actual_age=None, predicted_age=None, num_slices_per_view=36):
    """Plot all slices for a given view (axial, coronal, or sagittal) in a grid."""
    if plane not in ['axial', 'coronal', 'sagittal']:
        raise ValueError(f"Invalid plane: {plane}. Choose 'axial', 'coronal', or 'sagittal'.")

    num_slices = image.shape[{'axial': 0, 'coronal': 1, 'sagittal': 2}[plane]]
    slice_indices = np.linspace(0, num_slices - 1, num=min(num_slices_per_view, num_slices), dtype=int)

    rows = ceil(np.sqrt(len(slice_indices)))
    cols = ceil(len(slice_indices) / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), dpi=300)

    image_norm = (image - image.min()) / (image.max() - image.min())
    cam_norm = normalize_cam(cam)

    for idx, ax in enumerate(axes.flat):
        if idx < len(slice_indices):
            slice_idx = slice_indices[idx]
            if plane == 'axial':
                img_slice = image_norm[slice_idx, :, :]
                cam_slice = cam_norm[slice_idx, :, :]
            elif plane == 'coronal':
                img_slice = image_norm[:, slice_idx, :]
                cam_slice = cam_norm[:, slice_idx, :].transpose(1,0)
            elif plane == 'sagittal':
                img_slice = image_norm[:, :, slice_idx]
                cam_slice = cam_norm[:, :, slice_idx].transpose(1,0)

            ax.imshow(img_slice, cmap='gray')
            im = ax.imshow(cam_slice, cmap='jet', alpha=0.5)
            ax.set_title(f'Slice {slice_idx}', fontsize='small')
        ax.axis('off')

    # Add colorbar *after* creating subplots, using figure coordinates.
    if len(slice_indices) > 0:
      fig.subplots_adjust(right=0.8)  # Make space for the colorbar on the right.
      cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # [left, bottom, width, height] in figure coordinates.
      fig.colorbar(im, cax=cax, label="Heatmap Intensity")

    if actual_age is not None and predicted_age is not None:
        fig.suptitle(f'{plane.capitalize()} View - Act: {actual_age:.1f}, Pred: {predicted_age:.1f}', y=0.99, fontsize=12)

    # No need for tight_layout here, we've manually adjusted spacing.
    return fig

def generate_average_heatmaps(avg_image, avg_cam, output_dir, prefix=""):
    """Generate average heatmaps for all planes using plot_all_slices"""

    # Grid plots for each plane
    for plane in ['axial', 'coronal', 'sagittal']:
        fig = plot_all_slices(avg_image, avg_cam, plane=plane)
        fig.savefig(os.path.join(output_dir, f"{prefix}average_{plane}_slices_grid.png"),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)

def plot_all_slices_side_by_side(image, cam, plane='axial', num_slices=25,
                                actual_age=None, predicted_age=None):
    """Plot 25 slices with original and heatmap side-by-side (5 rows x 10 columns) - REMOVED"""
    pass # Removed plotting of individual slices


def plot_average_summary(avg_image, avg_cam, output_dir, prefix=""):
    """Create 2x3 grid showing average image and heatmap across planes, saves to output_dir"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    planes = ['sagittal', 'axial', 'coronal']

    # Normalize averages *here*
    avg_image_norm = (avg_image - avg_image.min()) / (avg_image.max() - avg_image.min())
    avg_cam_norm = (avg_cam - avg_cam.min()) / (avg_cam.max() - avg_cam.min() + 1e-7) # Add small value


    for col, plane in enumerate(planes):
        # Get middle slice indices
        slice_idx = {
            'sagittal': avg_image.shape[0] // 2,
            'axial': avg_image.shape[2] // 2,
            'coronal': avg_image.shape[1] // 2
        }[plane]

        if plane == 'axial':
            img_slice = avg_image_norm[:, :, slice_idx]
            cam_slice = avg_cam_norm[:, :, slice_idx]
        elif plane == 'coronal':
            img_slice = avg_image_norm[:, slice_idx, :]
            cam_slice = avg_cam_norm[:, slice_idx, :]
        elif plane == 'sagittal':
            img_slice = avg_image_norm[slice_idx, :, :]
            cam_slice = avg_cam_norm[slice_idx, :, :]

        # Average image
        axes[0, col].imshow(img_slice, cmap='gray')
        axes[0, col].set_title(f'Average {plane.capitalize()} Image')

        # Average heatmap
        im = axes[1, col].imshow(img_slice, cmap='gray')
        im = axes[1, col].imshow(cam_slice, cmap='jet', alpha=0.5)
        axes[1, col].set_title(f'Average {plane.capitalize()} Heatmap')
        fig.colorbar(im, ax=axes[1, col], label='Heatmap Intensity')
        

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()

    # Save the figure
    filename = os.path.join(output_dir, f"{prefix}average_summary.png")
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_regional_intensity_heatmap(regional_intensity_data, output_dir, method_name, prefix="", colormap='viridis'):
    """
    Plots a heatmap of average regional intensities, using a provided colormap.

    Args:
        regional_intensity_data (dict): Dictionary of regional intensities.  Can be a single dictionary
                                         or a list of dictionaries (e.g., from multiple samples).
        output_dir (str): Directory to save the plot.
        method_name (str): Name of the XAI method.
        prefix (str, optional): Prefix for the filename. Defaults to "".
        colormap (str, optional):  Name of the matplotlib colormap to use. Defaults to 'viridis'.
    """

    if isinstance(regional_intensity_data, list):
        # If it's a list of dictionaries, average them
        df = pd.DataFrame(regional_intensity_data)
        avg_intensities = df.mean(axis=0)
    else:
        # If it's a single dictionary, use it directly
        avg_intensities = pd.Series(regional_intensity_data)

    regions = avg_intensities.index.to_numpy()
    intensities = avg_intensities.to_numpy()

    plt.figure(figsize=(14, 8))  # Adjust figure size for better readability
    plt.imshow([intensities], aspect='auto', cmap=colormap)  # Use imshow for heatmap
    plt.colorbar(label='Average Heatmap Intensity')  # Add colorbar with label
    plt.yticks([])  # Hide y-ticks (no need for row labels)
    plt.xticks(ticks=range(len(regions)), labels=regions, rotation=90)  # Show region labels
    plt.title(f'{prefix}Average Heatmap Intensity per Region - {method_name}')
    plt.xlabel('Atlas Region Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}{method_name}_regional_intensity_heatmap.png"), bbox_inches='tight', dpi=300)
    plt.close()



def plot_regional_correlation_heatmap(regional_intensity_age_data, output_dir, method_name, prefix="", colormap='RdBu_r'):
    """
    Plots a heatmap of correlations between regional intensities and age, using a diverging colormap.

    Args:
        regional_intensity_age_data (dict): Nested dictionary containing regional intensities and ages.
        output_dir (str): Directory to save the plot.
        method_name (str): Name of the XAI method.
        prefix (str, optional): Prefix for the filename. Defaults to "".
        colormap (str, optional):  Name of the diverging matplotlib colormap. Defaults to 'RdBu_r'.
    """

    correlation_results = {}
    for region_label in regional_intensity_age_data:
        intensities = regional_intensity_age_data[region_label]['intensities']
        ages = regional_intensity_age_data[region_label]['ages']
        if len(intensities) > 1:
            corr, _ = pearsonr(intensities, ages)  # Calculate correlation
            correlation_results[region_label] = corr
        else:
            correlation_results[region_label] = np.nan  # Handle cases with insufficient data

    regions = list(correlation_results.keys())
    correlations = list(correlation_results.values())
        
    plt.figure(figsize=(14, 8))  # Adjust for readability
    plt.imshow([correlations], aspect='auto', cmap=colormap, vmin=-1, vmax=1)  # Diverging colormap, centered at 0
    plt.colorbar(label='Pearson Correlation Coefficient')
    plt.yticks([])
    plt.xticks(ticks=range(len(regions)), labels=regions, rotation=90)
    plt.title(f'{prefix}Correlation of Heatmap Intensity with Age - {method_name}')
    plt.xlabel('Atlas Region Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}{method_name}_regional_correlation_heatmap.png"), bbox_inches='tight', dpi=300)
    plt.close()


def plot_regional_intensity_difference_heatmap(regional_intensity_data_bin1, regional_intensity_data_bin2, output_dir, method_name, bin1_label, bin2_label, prefix="", colormap='coolwarm'):
    """
    Plots a heatmap showing the *difference* in average regional intensities between two age bins.

    Args:
        regional_intensity_data_bin1 (dict or list): Regional intensity data for the first age bin.
        regional_intensity_data_bin2 (dict or list): Regional intensity data for the second age bin.
        output_dir (str): Output directory.
        method_name (str): Name of the XAI method.
        bin1_label (str): Label for the first age bin (e.g., "0-10").
        bin2_label (str): Label for the second age bin (e.g., "80-90").
        prefix (str, optional): Prefix for filename. Defaults to "".
        colormap (str): Diverging colormap.
    """

    if isinstance(regional_intensity_data_bin1, list):
        df1 = pd.DataFrame(regional_intensity_data_bin1)
        avg_intensities1 = df1.mean(axis=0)
    else:
        avg_intensities1 = pd.Series(regional_intensity_data_bin1)

    if isinstance(regional_intensity_data_bin2, list):
        df2 = pd.DataFrame(regional_intensity_data_bin2)
        avg_intensities2 = df2.mean(axis=0)
    else:
        avg_intensities2 = pd.Series(regional_intensity_data_bin2)
    
    # Ensure both have same regions, take union, fill missing with NaN, use consistent ordering
    all_regions = sorted(list(set(avg_intensities1.index) | set(avg_intensities2.index)))
    avg_intensities1 = avg_intensities1.reindex(all_regions, fill_value=np.nan)
    avg_intensities2 = avg_intensities2.reindex(all_regions, fill_value=np.nan)

    # Calculate the difference
    intensity_differences = avg_intensities2 - avg_intensities1

    regions = intensity_differences.index.to_numpy()  # Use consistent regions
    differences = intensity_differences.to_numpy()

    plt.figure(figsize=(14, 8))
    plt.imshow([differences], aspect='auto', cmap=colormap)  # Use a diverging colormap
    plt.colorbar(label='Difference in Average Heatmap Intensity')
    plt.yticks([])
    plt.xticks(ticks=range(len(regions)), labels=regions, rotation=90)
    plt.title(f'{prefix}Difference in Average Heatmap Intensity ({bin2_label} - {bin1_label}) - {method_name}')
    plt.xlabel('Atlas Region Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}{method_name}_regional_difference_heatmap_{bin1_label}_{bin2_label}.png"), bbox_inches='tight', dpi=300)
    plt.close()

def plot_regional_intensity_change_across_bins(regional_intensity_data_bins, output_dir, method_name, prefix="", colormap='viridis'):
    """
    Plots a heatmap showing the average regional intensities across multiple age bins.

    Args:
        regional_intensity_data_bins (dict):  A dictionary where keys are bin labels (e.g., "0-10", "10-20")
                                            and values are the regional intensity data for that bin
                                            (either a dict or a list of dicts).
        output_dir (str): Output directory.
        method_name (str):  XAI Method name.
        prefix (str, optional): Prefix for the filename.
        colormap (str, optional): Colormap for the heatmap.
    """

    # Find all unique regions across all bins
    all_regions = set()
    for bin_data in regional_intensity_data_bins.values():
        if isinstance(bin_data, list):
            for sample_data in bin_data:
                all_regions.update(sample_data.keys())
        else:
            all_regions.update(bin_data.keys())
    all_regions = sorted(list(all_regions))

    # Create a 2D array to store the data for the heatmap
    heatmap_data = []
    bin_labels = sorted(regional_intensity_data_bins.keys())  # Ensure consistent bin order

    for bin_label in bin_labels:
        bin_data = regional_intensity_data_bins[bin_label]
        if isinstance(bin_data, list):
            df = pd.DataFrame(bin_data)
            avg_intensities = df.mean(axis=0)
        else:
            avg_intensities = pd.Series(bin_data)
        # Reindex to ensure all regions are present, fill missing with NaN
        avg_intensities = avg_intensities.reindex(all_regions, fill_value=np.nan)
        heatmap_data.append(avg_intensities.to_numpy())

    heatmap_data = np.array(heatmap_data)

    plt.figure(figsize=(16, 10))  # Larger figure for more bins
    plt.imshow(heatmap_data, aspect='auto', cmap=colormap)
    plt.colorbar(label='Average Heatmap Intensity')
    plt.yticks(ticks=range(len(bin_labels)), labels=bin_labels)  # Show bin labels
    plt.xticks(ticks=range(len(all_regions)), labels=all_regions, rotation=90)
    plt.title(f'{prefix}Average Heatmap Intensity Across Age Bins - {method_name}')
    plt.xlabel('Atlas Region Label')
    plt.ylabel('Age Bin')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}{method_name}_regional_intensity_across_bins.png"), bbox_inches='tight', dpi=300)
    plt.close()

def overlay_heatmap_on_brain(image, cam, output_path, plane='axial', slice_index=None, alpha=0.5, colormap='jet'):
    """
    Overlays a heatmap (CAM) on a brain image slice and saves the result.

    Args:
        image (numpy.ndarray): The original brain image (3D).
        cam (numpy.ndarray): The heatmap (CAM) data (3D).  Assumed to be the same shape as image.
        output_path (str): Path to save the overlaid image.
        plane (str):  'axial', 'coronal', or 'sagittal'.
        slice_index (int, optional):  Specific slice to display.  If None, uses the middle slice.
        alpha (float): Transparency of the heatmap overlay (0-1).
        colormap (str):  Matplotlib colormap name.
    """

    # Normalize image and CAM
    image_norm = (image - image.min()) / (image.max() - image.min())
    cam_norm = normalize_cam(cam)  # Use the existing normalize_cam function

    # Determine slice index
    if slice_index is None:
        slice_index = image.shape[{'axial': 0, 'coronal': 1, 'sagittal': 2}[plane]] // 2

    # Extract the slice
    if plane == 'axial':
        img_slice = image_norm[slice_index, :, :]
        cam_slice = cam_norm[slice_index, :, :]
    elif plane == 'coronal':
        img_slice = image_norm[:, slice_index, :]
        cam_slice = cam_norm[:, slice_index, :]  # Corrected: No transpose here
    elif plane == 'sagittal':
        img_slice = image_norm[:, :, slice_index]
        cam_slice = cam_norm[:, :, slice_index]  # Corrected: No transpose here
    else:
        raise ValueError("Invalid plane.  Must be 'axial', 'coronal', or 'sagittal'.")

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_slice, cmap='gray')  # Grayscale brain image
    im = ax.imshow(cam_slice, cmap=colormap, alpha=alpha)  # Overlay heatmap
    fig.colorbar(im, ax=ax, label="Heatmap Intensity") # Add colorbar
    ax.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def generate_average_overlay_heatmaps(avg_image, avg_cam, output_dir, prefix=""):
    """
    Generates overlaid heatmap images for all three planes (axial, coronal, sagittal)
    for the *average* image and CAM.  Saves each plane as a separate image.

    Args:
        avg_image (numpy.ndarray): The average brain image.
        avg_cam (numpy.ndarray): The average heatmap (CAM).
        output_dir (str):  Output directory.
        prefix (str): Prefix for filenames (e.g., "dataset_" or "age_bin_1_").
    """

    for plane in ['axial', 'coronal', 'sagittal']:
        output_path = os.path.join(output_dir, f"{prefix}average_overlay_{plane}.png")
        overlay_heatmap_on_brain(avg_image, avg_cam, output_path, plane=plane)


def generate_average_overlay_heatmaps_grid(avg_image, avg_cam, output_dir, prefix="", num_slices=36):
    """
    Generates overlaid heatmap images for multiple slices in each plane,
    arranged in a grid.  Saves each plane's grid as a separate image.
    """

    for plane in ['axial', 'coronal', 'sagittal']:
        num_slices_in_plane = avg_image.shape[{'axial': 0, 'coronal': 1, 'sagittal': 2}[plane]]
        slice_indices = np.linspace(0, num_slices_in_plane - 1, num=min(num_slices, num_slices_in_plane), dtype=int)

        rows = ceil(np.sqrt(len(slice_indices)))
        cols = ceil(len(slice_indices) / rows)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), dpi=300)

        image_norm = (avg_image - avg_image.min()) / (avg_image.max() - avg_image.min())
        cam_norm = normalize_cam(avg_cam)

        for idx, ax in enumerate(axes.flat):
            if idx < len(slice_indices):
                slice_idx = slice_indices[idx]
                if plane == 'axial':
                    img_slice = image_norm[slice_idx, :, :]
                    cam_slice = cam_norm[slice_idx, :, :]
                elif plane == 'coronal':
                    img_slice = image_norm[:, slice_idx, :]
                    cam_slice = cam_norm[:, slice_idx, :]
                elif plane == 'sagittal':
                    img_slice = image_norm[:, :, slice_idx]
                    cam_slice = cam_norm[:, :, slice_idx]

                ax.imshow(img_slice, cmap='gray')
                im = ax.imshow(cam_slice, cmap='jet', alpha=0.5)
                ax.set_title(f'Slice {slice_idx}', fontsize='small')
            ax.axis('off')
        if len(slice_indices) > 0:
            fig.subplots_adjust(right=0.8)
            cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cax, label = "Heatmap intensity")

        #plt.tight_layout(h_pad=0.1, w_pad=0.1) # No tight_layout needed.
        output_path = os.path.join(output_dir, f"{prefix}average_overlay_{plane}_grid.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

def plot_all_slices_voxelwise(image, cam, plane='axial', num_slices_per_view=36):
    """Plot all slices for a given view (voxel-wise), in a grid."""
    if plane not in ['axial', 'coronal', 'sagittal']:
        raise ValueError(f"Invalid plane: {plane}. Choose 'axial', 'coronal', or 'sagittal'.")

    num_slices = image.shape[{'axial': 0, 'coronal': 1, 'sagittal': 2}[plane]]
    slice_indices = np.linspace(0, num_slices - 1, num=min(num_slices_per_view, num_slices), dtype=int)

    rows = ceil(np.sqrt(len(slice_indices)))
    cols = ceil(len(slice_indices) / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), dpi=300)

    image_norm = (image - image.min()) / (image.max() - image.min())
    cam_norm = normalize_cam(cam)  # Use your existing normalization

    for idx, ax in enumerate(axes.flat):
        if idx < len(slice_indices):
            slice_idx = slice_indices[idx]
            if plane == 'axial':
                img_slice = image_norm[slice_idx, :, :]
                cam_slice = cam_norm[slice_idx, :, :]
            elif plane == 'coronal':
                img_slice = image_norm[:, slice_idx, :]
                cam_slice = cam_norm[:, slice_idx, :].transpose(1,0)
            elif plane == 'sagittal':
                img_slice = image_norm[:, :, slice_idx]
                cam_slice = cam_norm[:, :, slice_idx].transpose(1,0)

            ax.imshow(img_slice, cmap='gray')
            im = ax.imshow(cam_slice, cmap='jet', alpha=0.5)
            ax.set_title(f'Slice {slice_idx}', fontsize='small')
        ax.axis('off')

    # Add colorbar (same as before, adjusted for figure coordinates)
    if len(slice_indices) > 0:
      fig.subplots_adjust(right=0.8)
      cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
      fig.colorbar(im, cax=cax, label="Heatmap Intensity")

    #No tight_layout needed

    return fig

def generate_average_heatmaps_voxelwise(avg_image, avg_cam, output_dir, prefix=""):
    """Generate average heatmaps for all planes (voxel-wise, no atlas)"""

    # Grid plots for each plane (using the new plot_all_slices_voxelwise)
    for plane in ['axial', 'coronal', 'sagittal']:
        fig = plot_all_slices_voxelwise(avg_image, avg_cam, plane=plane)
        fig.savefig(os.path.join(output_dir, f"{prefix}average_{plane}_slices_grid_voxelwise.png"),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)

def plot_average_summary_voxelwise(avg_image, avg_cam, output_dir, prefix=""):
    """Create 2x3 grid showing average image and heatmap (voxel-wise)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    planes = ['sagittal', 'axial', 'coronal']

    # Normalize averages *here*
    avg_image_norm = (avg_image - avg_image.min()) / (avg_image.max() - avg_image.min())
    avg_cam_norm = normalize_cam(avg_cam) # Use your existing normalization

    for col, plane in enumerate(planes):
        # Get middle slice indices
        slice_idx = {
            'sagittal': avg_image.shape[0] // 2,
            'axial': avg_image.shape[2] // 2,
            'coronal': avg_image.shape[1] // 2
        }[plane]

        if plane == 'axial':
            img_slice = avg_image_norm[:, :, slice_idx]
            cam_slice = avg_cam_norm[:, :, slice_idx]
        elif plane == 'coronal':
            img_slice = avg_image_norm[:, slice_idx, :]
            cam_slice = avg_cam_norm[:, slice_idx, :]
        elif plane == 'sagittal':
            img_slice = avg_image_norm[slice_idx, :, :]
            cam_slice = avg_cam_norm[slice_idx, :, :]

        # Average image
        axes[0, col].imshow(img_slice, cmap='gray')
        axes[0, col].set_title(f'Average {plane.capitalize()} Image')

        # Average heatmap
        im = axes[1, col].imshow(img_slice, cmap='gray')
        im = axes[1, col].imshow(cam_slice, cmap='jet', alpha=0.5)
        axes[1, col].set_title(f'Average {plane.capitalize()} Heatmap')
        fig.colorbar(im, ax=axes[1, col], label='Heatmap Intensity')


    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()

    # Save the figure
    filename = os.path.join(output_dir, f"{prefix}average_summary_voxelwise.png")
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

def create_regional_heatmap(regional_intensities, atlas_data_resampled):
    """
    Creates a heatmap where each voxel within a region has the average intensity
    for that region.

    Args:
        regional_intensities (dict): Dictionary of {region_label: average_intensity}.
        atlas_data_resampled (np.ndarray): The resampled atlas data.

    Returns:
        np.ndarray: The regional heatmap.
    """
    regional_heatmap = np.zeros_like(atlas_data_resampled, dtype=np.float32)
    for region_label, intensity in regional_intensities.items():
        if not np.isnan(intensity):  # Handle potential NaN values
            regional_heatmap[atlas_data_resampled == region_label] = intensity
    return regional_heatmap

def generate_xai_visualizations(model, dataset, output_dir, device='cuda', methods_to_run=['all'], atlas_path=None, age_bins = 10):
    """Main visualization function."""
    methods = create_visualization_dirs(output_dir, methods_to_run)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    atlas_data, _ = load_atlas(atlas_path)  # Keep atlas loading

    if atlas_data is None:
        logging.error("Atlas loading failed, regional analysis skipped.")
        # Don't return; we still want voxel-wise plots
    else:
         # --- Keep Dummy Model for Atlas Resampling ---
        dummy_image_batch = next(iter(loader))['image'].unsqueeze(1).to(device)
        dummy_demographics = next(iter(loader))['demographics'].to(device)
        dummy_wrapped_model = BrainAgeWrapper(model, dummy_demographics)
        dummy_target_layers = get_target_layers(dummy_wrapped_model)
        dummy_cam_object = GradCAM(model=dummy_wrapped_model, target_layers=dummy_target_layers)
        with torch.enable_grad():
            dummy_grayscale_cam = dummy_cam_object(input_tensor=dummy_image_batch)[0]
        target_atlas_shape = dummy_grayscale_cam.shape
        print(f"Dummy grayscale_cam shape: {dummy_grayscale_cam.shape}")
        print(f"Target shape for atlas resampling: {target_atlas_shape}")

    # --- Accumulators for BOTH Regional and Voxel-wise ---
    regional_intensity_accumulators = {
    method_name: [] for method_name in methods.keys()
    }
    regional_intensity_age_data_accumulators = {
        method_name: {region_label: {'intensities': [], 'ages': []} for region_label in np.unique(atlas_data)[1:] if region_label != 0}
        for method_name in methods.keys()
    }

    method_accumulators = {method_name: {'image': None, 'cam': None, 'regional_cam': None, 'voxelwise_cam': None, 'count': 0}
                           for method_name in methods.keys()}  # Add 'voxelwise_cam'
    age_bins = np.linspace(0, 100, age_bins + 1)
    atlas_data_resampled = None # Keep
    bin_accumulators = {
        method_name: {i: {'image': None, 'cam': None, 'regional_cam': None, 'voxelwise_cam': None, 'count': 0} for i in range(len(age_bins))}
        for method_name in methods.keys()
    }



    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Processing images")):
            try:
                image = batch["image"].unsqueeze(1).to(device)
                demographics = batch["demographics"].to(device)
                actual_age = batch["age"].item()

                wrapped_model = BrainAgeWrapper(model, demographics)
                predicted_age = wrapped_model(image).detach().item()

                target_layers = get_target_layers(wrapped_model)

                for method_name, method_class in methods.items():
                    with torch.enable_grad():
                        target_layers = get_target_layers(wrapped_model)
                        cam_method_dir = os.path.join(output_dir, method_name)
                        cam_object = method_class(model=wrapped_model, target_layers=target_layers)
                        print('==================',image.shape)
                        _cam = cam_object(input_tensor=image)
                        
                        if _cam.ndim == 5:
                            _cam = _cam[0,0]
                        else:
                            _cam = _cam[0]
                        grayscale_cam = np.squeeze(_cam)

                    orig_image = batch['image'].squeeze().numpy()
                    # --- Keep Target Size for Atlas ---
                    target_size = (target_atlas_shape[2], target_atlas_shape[1])
                    if grayscale_cam is not None:
                        grayscale_cam_copy = grayscale_cam.copy() # Copy for regional analysis
                        normalized_grayscale_cam = normalize_cam(grayscale_cam_copy, target_size=target_size)  # For overlays and potential atlas use

                    # --- Voxel-wise Accumulation (BEFORE resizing) ---
                    if grayscale_cam is not None:
                        if method_accumulators[method_name]['image'] is None:
                            method_accumulators[method_name]['image'] = np.zeros_like(orig_image, dtype=np.float32)
                            method_accumulators[method_name]['voxelwise_cam'] = np.zeros_like(orig_image, dtype=np.float32)  # Voxel-wise CAM
                            # --- For regional analysis, keep accumulating cam ---
                            if atlas_data is not None:
                                 method_accumulators[method_name]['cam'] = np.zeros(target_atlas_shape, dtype=np.float32)
                                 method_accumulators[method_name]['regional_cam'] = np.zeros(target_atlas_shape, dtype=np.float32)
                        method_accumulators[method_name]['image'] += orig_image.astype(np.float32)
                        method_accumulators[method_name]['voxelwise_cam'] += grayscale_cam.astype(np.float32)  # Accumulate raw CAM
                        method_accumulators[method_name]['count'] += 1

                        bin_idx = np.digitize(actual_age, age_bins) - 1
                        bin_idx = max(0, min(bin_idx, 9))
                        if bin_accumulators[method_name][bin_idx]['image'] is None:
                            bin_accumulators[method_name][bin_idx]['image'] = np.zeros_like(orig_image, dtype=np.float32)
                            bin_accumulators[method_name][bin_idx]['voxelwise_cam'] = np.zeros_like(orig_image, dtype=np.float32) # Voxel-wise bin
                            # --- For bins regional analysis, keep accumulating cam ---
                            if atlas_data is not None:
                                bin_accumulators[method_name][bin_idx]['cam'] = np.zeros(target_atlas_shape, dtype=np.float32)
                                bin_accumulators[method_name][bin_idx]['regional_cam'] = np.zeros(target_atlas_shape, dtype=np.float32)
                        bin_accumulators[method_name][bin_idx]['image'] += orig_image.astype(np.float32)
                        bin_accumulators[method_name][bin_idx]['voxelwise_cam'] += grayscale_cam.astype(np.float32)
                        bin_accumulators[method_name][bin_idx]['count'] += 1


                    # --- KEEP Regional/Atlas Code ---
                    if atlas_data is not None:
                        if grayscale_cam is not None:
                            if atlas_data_resampled is None:
                                current_atlas_shape = atlas_data.shape
                                zoom_factors = [ts / as_ for ts, as_ in zip(target_atlas_shape, current_atlas_shape)]
                                atlas_data_resampled = scipy.ndimage.zoom(atlas_data, zoom_factors, order=0)
                            if not isinstance(grayscale_cam, np.ndarray):
                                logging.error(f"Grayscale CAM is not a numpy array, skipping resize and regional analysis for sample {batch_idx}")
                                grayscale_cam = None
                            elif np.isnan(grayscale_cam).any() or np.isinf(grayscale_cam).any():
                                logging.error(f"Grayscale CAM contains NaN or Inf values, skipping resize and regional analysis for sample {batch_idx}")
                                grayscale_cam = None
                            elif grayscale_cam.size == 0:
                                logging.error(f"Grayscale CAM is empty, skipping resize and regional analysis for sample {batch_idx}")
                                grayscale_cam = None
                            elif grayscale_cam.shape != atlas_data_resampled.shape:
                                if grayscale_cam.ndim == 2:
                                    grayscale_cam = cv2.resize(grayscale_cam, (atlas_data_resampled.shape[2], atlas_data_resampled.shape[1]), interpolation=cv2.INTER_LINEAR)
                                    grayscale_cam = np.expand_dims(grayscale_cam, axis=0)
                                elif grayscale_cam.ndim == 3:
                                    resized_slices = []
                                    for d in range(grayscale_cam.shape[2]):
                                        slice_2d = grayscale_cam[:, :, d]
                                        resized_slice = cv2.resize(
                                            slice_2d,
                                            (atlas_data_resampled.shape[2], atlas_data_resampled.shape[1]),
                                            interpolation=cv2.INTER_LINEAR
                                        )
                                        resized_slices.append(resized_slice)
                                    grayscale_cam = np.stack(resized_slices, axis=2)
                                    grayscale_cam = grayscale_cam.transpose(1, 0, 2)

                            regional_intensities = calculate_regional_intensity(grayscale_cam, atlas_data_resampled)

                            if regional_intensities:
                                regional_heatmap = create_regional_heatmap(regional_intensities, atlas_data_resampled)

                                # Accumulate regional heatmap (KEEP)
                                if atlas_data is not None:
                                    method_accumulators[method_name]['cam'] += grayscale_cam.astype(np.float32)
                                    method_accumulators[method_name]['regional_cam'] += regional_heatmap.astype(np.float32)
                                
                                # Bin regional heatmap (KEEP)
                                if atlas_data is not None:
                                    bin_accumulators[method_name][bin_idx]['cam'] += grayscale_cam.astype(np.float32)
                                    bin_accumulators[method_name][bin_idx]['regional_cam'] += regional_heatmap.astype(np.float32)

                                for region_label, intensity in regional_intensities.items():
                                    if not np.isnan(intensity):
                                        regional_intensity_age_data_accumulators[method_name][region_label]['intensities'].append(intensity)
                                        regional_intensity_age_data_accumulators[method_name][region_label]['ages'].append(actual_age)
                            else:
                                logging.warning(f"Skipping regional heatmap accumulation for sample {batch_idx} due to invalid regional intensities.")

            except Exception as e:
                tb_str = traceback.format_exc()
                print(f"Error processing sample {batch_idx}: {str(e)}\nTraceback:\n{tb_str}")

    for method_name in methods.keys():
        method_dir = os.path.join(output_dir, method_name)
        if method_accumulators[method_name]['count'] > 0:
            avg_image = method_accumulators[method_name]['image'] / method_accumulators[method_name]['count']
            avg_voxelwise_cam = method_accumulators[method_name]['voxelwise_cam'] / method_accumulators[method_name]['count']  # Voxel-wise average

            # --- Atlas-Based Plots (KEEP) ---
            if atlas_data is not None:
                avg_cam = method_accumulators[method_name]['cam'] / method_accumulators[method_name]['count']
                avg_regional_cam = method_accumulators[method_name]['regional_cam'] / method_accumulators[method_name]['count']
                generate_average_heatmaps(avg_image, avg_regional_cam, method_dir, prefix="atlas_")  # Use "atlas_" prefix
                plot_average_summary(avg_image, avg_regional_cam, method_dir, prefix="atlas_")
                generate_average_overlay_heatmaps(avg_image, avg_regional_cam, method_dir, prefix="atlas_")
                generate_average_overlay_heatmaps_grid(avg_image, avg_regional_cam, method_dir, prefix="atlas_")

            # --- Voxel-wise Plots (ADD) ---
            generate_average_heatmaps_voxelwise(avg_image, avg_voxelwise_cam, method_dir, prefix="voxelwise_")  # Use "voxelwise_" prefix
            plot_average_summary_voxelwise(avg_image, avg_voxelwise_cam, method_dir, prefix="voxelwise_")
            generate_average_overlay_heatmaps(avg_image, avg_voxelwise_cam, method_dir, prefix="voxelwise_") # Keep the same
            generate_average_overlay_heatmaps_grid(avg_image, avg_voxelwise_cam, method_dir, prefix="voxelwise_") # Keep the same

            for bin_idx in range(10):
                if bin_accumulators[method_name][bin_idx]['count'] > 0:
                    bin_avg_image = bin_accumulators[method_name][bin_idx]['image'] / bin_accumulators[method_name][bin_idx]['count']
                    bin_avg_voxelwise_cam = bin_accumulators[method_name][bin_idx]['voxelwise_cam'] / bin_accumulators[method_name][bin_idx]['count']  # Voxel-wise bin average

                    # --- Atlas-Based Bin Plots (KEEP) ---
                    if atlas_data is not None:
                        bin_avg_cam = bin_accumulators[method_name][bin_idx]['cam'] / bin_accumulators[method_name][bin_idx]['count']
                        bin_avg_regional_cam = bin_accumulators[method_name][bin_idx]['regional_cam'] / bin_accumulators[method_name][bin_idx]['count']
                        generate_average_heatmaps(bin_avg_image, bin_avg_regional_cam, method_dir, prefix=f"atlas_age_bin_{bin_idx+1}_")
                        plot_average_summary(bin_avg_image, bin_avg_regional_cam, method_dir, prefix=f"atlas_age_bin_{bin_idx+1}_")
                        generate_average_overlay_heatmaps(bin_avg_image, bin_avg_regional_cam, method_dir, prefix=f"atlas_age_bin_{bin_idx+1}_") #Keep
                        generate_average_overlay_heatmaps_grid(bin_avg_image, bin_avg_regional_cam, method_dir, prefix=f"atlas_age_bin_{bin_idx+1}_") #Keep

                    # --- Voxel-wise Bin Plots (ADD) ---
                    generate_average_heatmaps_voxelwise(bin_avg_image, bin_avg_voxelwise_cam, method_dir, prefix=f"voxelwise_age_bin_{bin_idx+1}_")
                    plot_average_summary_voxelwise(bin_avg_image, bin_avg_voxelwise_cam, method_dir, prefix=f"voxelwise_age_bin_{bin_idx+1}_")
                    generate_average_overlay_heatmaps(bin_avg_image, bin_avg_voxelwise_cam, method_dir, prefix=f"voxelwise_age_bin_{bin_idx+1}_") # Keep the same
                    generate_average_overlay_heatmaps_grid(bin_avg_image, bin_avg_voxelwise_cam, method_dir, prefix=f"voxelwise_age_bin_{bin_idx+1}_") # Keep the same

            # --- Atlas-Based Regional Analysis (KEEP) ---
            if atlas_data is not None:
                if regional_intensity_accumulators[method_name]:
                    plot_regional_intensity_distributions(regional_intensity_accumulators[method_name], method_dir, method_name)
                    plot_average_regional_intensity_barchart(regional_intensity_accumulators[method_name], method_dir, method_name)
                    calculate_and_plot_regional_correlation(regional_intensity_age_data_accumulators[method_name], method_dir, method_name)
                    plot_regional_intensity_heatmap(regional_intensity_accumulators[method_name], method_dir, method_name, prefix="atlas_") #atlas
                    plot_regional_correlation_heatmap(regional_intensity_age_data_accumulators[method_name], method_dir, method_name, prefix="atlas_") # atlas
                else:
                    logging.warning(f"Skipping regional analysis plots for method '{method_name}' because no regional intensities were calculated.")

                # --- Atlas-Based Bin-wise Regional Analysis (KEEP) ---
                regional_intensity_data_bins = {}
                for bin_idx in range(10):
                    if bin_accumulators[method_name][bin_idx]['count'] > 0:
                        bin_regional_intensities = []
                        for batch_idx, batch in enumerate(loader):
                            # --- KEEP THIS CODE (Atlas Resizing, etc.) ---
                            if grayscale_cam is not None:  # Check if grayscale_cam exists
                                if atlas_data_resampled is None: # Only resample atlas once.
                                    current_atlas_shape = atlas_data.shape
                                    zoom_factors = [ts / as_ for ts, as_ in zip(target_atlas_shape, current_atlas_shape)] # Calculate zoom factors
                                    atlas_data_resampled = scipy.ndimage.zoom(atlas_data, zoom_factors, order=0) # order=0 for nearest neighbor (labels)

                                # Resize Logic from above goes here
                                if grayscale_cam.shape != atlas_data_resampled.shape:
                                    if grayscale_cam.ndim == 2:
                                        # If grayscale_cam is 2D, resize and then add a singleton dimension
                                        grayscale_cam = cv2.resize(grayscale_cam, (atlas_data_resampled.shape[2], atlas_data_resampled.shape[1]), interpolation=cv2.INTER_LINEAR) # Target size reversed!
                                        grayscale_cam = np.expand_dims(grayscale_cam, axis=0)  # Add the depth dimension back
                                    elif grayscale_cam.ndim == 3:
                                        # Resize each depth slice individually
                                        resized_slices = []
                                        for d in range(grayscale_cam.shape[2]):
                                            slice_2d = grayscale_cam[:, :, d]
                                            resized_slice = cv2.resize(
                                                slice_2d,
                                                (atlas_data_resampled.shape[2], atlas_data_resampled.shape[1]),
                                                interpolation=cv2.INTER_LINEAR
                                            )
                                            resized_slices.append(resized_slice)
                                        # Stack slices back into 3D volume
                                        grayscale_cam = np.stack(resized_slices, axis=2)
                                        grayscale_cam = grayscale_cam.transpose(1,0,2)
                                # ---
                                actual_age = batch["age"].item()
                                curr_bin_idx = np.digitize(actual_age, age_bins) - 1
                                curr_bin_idx = max(0, min(curr_bin_idx, 9))
                                if curr_bin_idx == bin_idx:
                                    regional_intensities = calculate_regional_intensity(grayscale_cam, atlas_data_resampled)
                                    if regional_intensities:
                                        bin_regional_intensities.append(regional_intensities)
                        regional_intensity_data_bins[f"{bin_idx*10}-{(bin_idx+1)*10}"] = bin_regional_intensities

                        # --- Difference Heatmap (Compare to first bin) ---
                        if bin_idx > 0: # Compare each bin with the first bin
                            plot_regional_intensity_difference_heatmap(
                                regional_intensity_data_bins[list(regional_intensity_data_bins.keys())[0]],  # First bin data
                                regional_intensity_data_bins[f"{bin_idx*10}-{(bin_idx+1)*10}"],  # Current bin
                                method_dir,
                                method_name,
                                "0-10",
                                f"{bin_idx*10}-{(bin_idx+1)*10}",
                                prefix=f"atlas_age_bin_diff_" #atlas prefix
                            )
                plot_regional_intensity_change_across_bins(regional_intensity_data_bins, method_dir, method_name, prefix="atlas_age_bins_") # Atlas prefix
        # --- (rest of the cleanup: del, torch.cuda.empty_cache(), gc.collect()) ---
        del wrapped_model, image, demographics, grayscale_cam, cam_object
        torch.cuda.empty_cache()
        gc.collect()
            

def process_single_model(csv_path, model_path, test_data_dir, base_output_dir, device,methods_to_run=['all'], atlas_path=None):
    """Process a single model for XAI visualization"""
    """Loads a model based on its filename using load_model_with_params."""
    model_filename = os.path.basename(model_path)
    
    match = re.search(r'_(.+?)_layer', model_filename)
    if match:
        model_name = match.group(1)
    else:
        model_name = 'unknown' 
    model = torch.load(model_path, map_location=device, weights_only = False)
    logging.info("Loaded model: %s of type %s", model_filename, model_name)

    test_data_name = os.path.basename(os.path.normpath(test_data_dir))
    model_output_dir = os.path.join(base_output_dir, model_name, model_filename.replace('.pth', ''))
    os.makedirs(model_output_dir, exist_ok=True)

    dataset = BrainAgeDataset(csv_path, test_data_dir)
    dataset = [sample for sample in dataset if sample is not None]

    num_demographics = 6
    # Initialize the appropriate model

    # Enable gradients
    for param in model.parameters():
        param.requires_grad = True

    generate_xai_visualizations(model, dataset, model_output_dir, device,methods_to_run, atlas_path)
    return model_output_dir

def main():
    parser = argparse.ArgumentParser(description="XAI Visualization for Brain Age Models")
    parser.add_argument('--models_dir', type=str, default='cvasl/deepresearch/saved_models_test',
                        help="Directory containing the saved model .pth files")
    parser.add_argument("--test_csv", type=str, default="cvasl/deepresearch/trainingdata/test/mock_data.csv", help="Path to the training CSV file")
    parser.add_argument('--test_data_dir', type=str,
                        default='cvasl/deepresearch/trainingdata/test/images/',
                        help="Directory containing the test data (CSV and image folder)")
    parser.add_argument('--output_dir', type=str, default='cvasl/deepresearch/xai',
                        help="Base output directory for visualizations")
    parser.add_argument('--method', type=str, default='gradcam',
                        help="Comma-separated list of XAI methods (gradcam, layercam, etc.) or 'all'")
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'], help="Device to use for computation")
    parser.add_argument('--atlas_path', type=str, default='cvasl/deepresearch/Harvard-Oxford_cortical_and_subcortical_structural_atlases/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz', # <--- ADD ATLAS PATH ARGUMENT
                        help="Path to the brain atlas NIfTI file")
    args = parser.parse_args()

    # Process methods argument
    methods_to_run = ['all'] if args.method == 'all' else args.method.split(',')

    # Handle device selection
    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    model_files = [f for f in os.listdir(args.models_dir) if f.endswith('.pth')]

    for model_file in tqdm(model_files, desc="Processing models"):
        model_path = os.path.join(args.models_dir, model_file)
        try:
            output_dir = process_single_model(
                args.test_csv, model_path, args.test_data_dir, args.output_dir, device, methods_to_run, args.atlas_path

            )
            logging.info(f"Successfully processed model {model_file}. Results saved in {output_dir}")
        except Exception as e:
            tb_str = traceback.format_exc()
            logging.error(f"Error processing model {model_file}: {str(e)}\nTraceback:\n{tb_str}")
            continue

if __name__ == "__main__":
    main()