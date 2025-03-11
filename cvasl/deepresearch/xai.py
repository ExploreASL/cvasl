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
import matplotlib
matplotlib.use('Agg')

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
from scipy.ndimage import zoom
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set a seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

region_names = {
    1: 'Left Cerebral White Matter',
    2: 'Left Cerebral Cortex',
    3: 'Left Lateral Ventricle',
    4: 'Left Thalamus',
 
}


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
        'layercam': LayerCAM,
    }

    # Filter methods if specific ones are requested
    if 'all' not in methods_to_run:
        all_methods = {k: v for k, v in all_methods.items() if k in methods_to_run}

    for method_name in all_methods.keys():
        method_dir = os.path.join(base_output_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)

    return all_methods

def get_last_conv_layer(model):
    """
    Recursively finds the last convolutional layer before a pooling or flattening operation.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        nn.Module: The last convolutional layer, or None if no convolutional layers are found.
    """
    last_conv_layer = None

    def recursive_search(module):
        nonlocal last_conv_layer
        
        #check if we have a container
        if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            modules = module.children()
        else:
            modules = [module]

        for submodule in modules:

            if isinstance(submodule, (nn.Conv2d, nn.Conv3d)):
                last_conv_layer = submodule
            elif isinstance(submodule, (nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool2d, nn.AvgPool3d,
                                       nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, nn.Flatten)):
                return  # Stop searching after pooling or flattening
            elif isinstance(submodule, nn.Linear):
                return # Stop searching: Linear layer after conv layers usually indicates flattening

            recursive_search(submodule)


    recursive_search(model)


    return last_conv_layer

def get_transformer_layer(model):
    """
    Gets a suitable layer within the last Transformer block for Grad-CAM.

    Args:
        model (nn.Module): The PyTorch model (assumed to be a VisionTransformer3D).

    Returns:
        nn.Module: The LayerNorm layer (norm2) within the last Transformer block.
                   Returns None if no Transformer blocks are found.
    """
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        last_block = model.blocks[-1]
        if hasattr(last_block, 'norm2') and isinstance(last_block.norm2, nn.LayerNorm):
            return last_block.norm2
        else: return None #Error, last block does not contains the norm2 LayerNorm
    
def get_target_layers(wrappedmodel):
    """Get the target layers for visualization based on model type."""
    model = wrappedmodel.model
    model_name = model.__class__.__name__  # Access the wrapped model

    if model_name == 'Large3DCNN':
        return get_last_conv_layer(model)  # Last Conv3d layer
    elif model_name == 'DenseNet3D':
        return get_last_conv_layer(model) # Transition layer before last avg pool
    elif model_name == 'EfficientNet3D':
        return get_last_conv_layer(model)  # Head convolution before avg pool
    elif model_name == 'Improved3DCNN':
        # Access the last layer in the sequential conv_layers
        return get_last_conv_layer(model)
    elif model_name == 'ResNet3D':
        get_last_conv_layer(model)  # Last conv layer in last ResNet block of layer3
    elif model_name == 'ResNeXt3D':
        return get_last_conv_layer(model)  # Last conv layer in last ResNeXt block of layer3
    elif model_name == 'VisionTransformer3D':
        # Target the final convolutional layer within the HybridEmbed3D module if it's used.
        if model.use_hybrid_embed:
            target_layer = get_last_conv_layer(model)
            if target_layer is not None:
                return [target_layer]
            else:
                return None # Handle the (unlikely) case where hybrid embed has no conv layers
        else:
            target_layer = get_transformer_layer(model)
            return [target_layer] if target_layer else None
    else:
        return None  # Default or unknown model type

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

def compute_centroid(atlas_slice, label):
    y, x = np.where(atlas_slice == label)
    if len(y) == 0:
        return None
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    return centroid_x, centroid_y        

def normalize_cam(cam, target_size=None):
    """Memory-efficient normalization and resizing"""
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7) # Small value for numerical stability

    if target_size is not None: # Resize if target_size is provided
        #cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR) # Use cv2.resize
        cam_resized = cv2.resize(cam, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        return cam_resized.astype(np.float16)  # Use float16 for memory efficiency
    return cam.astype(np.float16)


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

def get_atlas_labels(atlas_path):
    """
    Extracts unique region labels from the atlas.  Handles both integer and
    float atlases, and excludes the background (label 0).

    Args:
        atlas_path (str): Path to the NIfTI atlas file.

    Returns:
        list: A list of unique atlas region labels (excluding 0).  Returns an
              empty list if the atlas cannot be loaded.
    """
    atlas_data, _ = load_atlas(atlas_path)
    if atlas_data is None:
        return []

    # Flatten the atlas data and get unique values
    unique_labels = np.unique(atlas_data)

    # Remove 0 (background) and return the rest
    return [label for label in unique_labels if label != 0]



def resize_cam(cam, target_size):
    """
    Resizes a 3D CAM or image to the target size using trilinear interpolation.

    Args:
        cam (np.ndarray): The 3D CAM or image.
        target_size (tuple): The target size (D, H, W).

    Returns:
        np.ndarray: The resized CAM.  Returns the original CAM if resizing fails.
    """
    try:
        # Convert cam to float32 for cv2.resize
        cam_float32 = cam.astype(np.float32)
        
        # Resize using OpenCV (trilinear interpolation for 3D)
        resized_cam = cv2.resize(cam_float32, dsize=(target_size[2], target_size[1]), interpolation=cv2.INTER_LINEAR) # Resize H and W
        # Add depth dimension and use np.repeat to duplicate the resized image across depth
        resized_cam = np.repeat(resized_cam[np.newaxis, :, :], target_size[0], axis=0)

        return resized_cam

    except Exception as e:
        tb_str = traceback.format_exc()
        logging.error(f"Error resizing CAM: {e}\nTraceback:\n{tb_str}")
        return cam  # Return original in case of failure


def compute_region_based_heatmap(cam, atlas_data):
    """
    Computes a region-based heatmap where each voxel is assigned the average CAM value of its atlas region.
    
    Args:
        cam (np.ndarray): The 3D CAM heatmap (normalized), shape (D, H, W).
        atlas_data (np.ndarray): The 3D atlas data with integer region labels, shape (D, H, W).
    
    Returns:
        np.ndarray: The region-based heatmap, same shape as cam, with values constant within each region.
    """
    if cam.shape != atlas_data.shape:
        logging.error(f"CAM shape {cam.shape} does not match atlas shape {atlas_data.shape}")
        raise ValueError("CAM and atlas must have the same spatial dimensions")
    
    unique_labels = np.unique(atlas_data)
    region_heatmap = np.zeros_like(cam, dtype=np.float32)
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        mask = (atlas_data == label)
        if np.sum(mask) > 0:  # Ensure region has voxels
            avg_cam = np.mean(cam[mask])
            region_heatmap[mask] = avg_cam
    
    return region_heatmap


def generate_xai_visualizations_atlas(model, dataset, output_dir, device='cuda', methods_to_run=['all'], atlas_path=None):
    methods = create_visualization_dirs(output_dir, methods_to_run)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    atlas_data, _ = load_atlas(atlas_path)
    if atlas_data is None:
        logging.error("Failed to load atlas. Skipping atlas-based visualizations.")
        return

    view_axes = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    view_names = ['sagittal', 'coronal', 'axial']

    for method_name, cam_class in methods.items():
        method_output_dir = os.path.join(output_dir, method_name)
        all_slices_data = None

        for idx, sample in enumerate(tqdm(loader, desc=f"Processing images for {method_name} (atlas)")):
            if sample is None:
                logging.warning(f"Skipping None sample at index {idx}")
                continue

            image, demographics, brain_age = sample['image'], sample['demographics'], sample['age']
            image = image.to(device)
            demographics = demographics.to(device)
            wrapped_model = BrainAgeWrapper(model, demographics)
            target_layers = get_target_layers(wrapped_model)
            cam = cam_class(model=wrapped_model, target_layers=target_layers)

            try:
                grayscale_cam = cam(input_tensor=image.unsqueeze(0))
                grayscale_cam = grayscale_cam[0, :]
                heatmap = normalize_cam(grayscale_cam)
                img_np = image.cpu().numpy().squeeze()

                if idx == 0:
                    if atlas_data.shape != img_np.shape:
                        logging.info(f"Resizing atlas from {atlas_data.shape} to match image shape {img_np.shape}")
                        zoom_factors = [t / s for t, s in zip(img_np.shape, atlas_data.shape)]
                        resized_atlas = zoom(atlas_data, zoom_factors, order=0, mode='nearest')
                    else:
                        resized_atlas = atlas_data

                    all_slices_data = {view: {slice_idx: {'slices': [], 'heatmaps': []} for slice_idx in range(img_np.shape[view_axes[view]])} for view in view_names}

                region_heatmap = compute_region_based_heatmap(heatmap, resized_atlas)

                for view_name in view_names:
                    view_axis = view_axes[view_name]
                    for slice_idx in range(img_np.shape[view_axis]):
                        slice_data = np.take(img_np, indices=slice_idx, axis=view_axis)
                        heatmap_slice_data = np.take(region_heatmap, indices=slice_idx, axis=view_axis)
                        heatmap_resized_third = normalize_cam(heatmap_slice_data, slice_data.shape)
                        all_slices_data[view_name][slice_idx]['slices'].append(slice_data)
                        all_slices_data[view_name][slice_idx]['heatmaps'].append(heatmap_resized_third)

            except Exception as e:
                logging.error(f"Error processing image with {method_name} (atlas): {e}")
                continue

        for view in view_names:
            for slice_idx in all_slices_data[view]:
                slices_list = all_slices_data[view][slice_idx]['slices']
                heatmaps_list = all_slices_data[view][slice_idx]['heatmaps']
                if slices_list:
                    all_slices_data[view][slice_idx]['avg_slice'] = np.mean(np.stack(slices_list, axis=0), axis=0)
                    all_slices_data[view][slice_idx]['avg_heatmap'] = np.mean(np.stack(heatmaps_list, axis=0), axis=0)
                else:
                    all_slices_data[view][slice_idx]['avg_slice'] = np.zeros((64,64))
                    all_slices_data[view][slice_idx]['avg_heatmap'] = np.zeros((64,64))

        for view_name in view_names:
            view_data = [(slice_idx, all_slices_data[view_name][slice_idx]['avg_slice'], all_slices_data[view_name][slice_idx]['avg_heatmap']) for slice_idx in sorted(all_slices_data[view_name].keys())]
            n_slices = len(view_data)
            if n_slices == 0:
                continue
            n_cols = 12
            n_rows = ceil(n_slices / n_cols)

            fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), dpi=600)
            fig3.suptitle(f"All Slice Heatmaps (Atlas) - {method_name.capitalize()} - {view_name.capitalize()}")

            if n_rows == 1:
                axes3 = axes3[np.newaxis, :]

            for idx, (slice_idx, avg_slice, avg_heatmap) in enumerate(view_data):
                row_idx = idx // n_cols
                col_idx = idx % n_cols
                ax = axes3[row_idx, col_idx] if axes3.ndim > 1 else axes3[col_idx]
                ax.imshow(avg_slice, cmap='gray')
                im = ax.imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
                ax.set_title(f"Slice {slice_idx}")
                ax.axis('off')

            if n_slices % n_cols != 0:
                for j in range(n_slices % n_cols, n_cols):
                    (axes3[n_rows - 1, j] if axes3.ndim > 1 else axes3[j]).axis('off')
            cbar_ax = fig3.add_axes([0.125, 0.05, 0.775, 0.02]) 
            fig3.colorbar(mappable=im, cax=cbar_ax, # Use the last 'im' for simplicity.  Better to be explicit (see below).
             #ax=axes3.ravel().tolist(),  # Pass ALL axes.
             orientation='horizontal',
             label='Normalized CAM',
              shrink=0.6,  # Optional: Adjust size (e.g., make it slightly smaller)
            #  aspect=40,   # Optional: Adjust aspect ratio (make it wider)
            #  pad=-0.5     # Optional: Add some padding between colorbar and subplots.
             )
            plt.tight_layout()  # Adjust layout for title

            
            plt.savefig(os.path.join(method_output_dir, f"all_slices_heatmaps_{view_name}_atlas.png"), dpi=600)
            plt.close(fig3)

def generate_xai_visualizations_atlas_binned(model, dataset, output_dir, device='cuda', methods_to_run=['all'], atlas_path=None, age_bin_width=10):
    methods = create_visualization_dirs(output_dir, methods_to_run)
    model.eval()
    atlas_data, _ = load_atlas(atlas_path)
    if atlas_data is None:
        logging.error("Failed to load atlas. Skipping atlas-based visualizations.")
        return

    max_age = max([sample['age'] for sample in dataset if sample is not None]).item()
    bin_edges = np.arange(0, max_age + age_bin_width, age_bin_width)
    if bin_edges[-1] < max_age:
        bin_edges = np.append(bin_edges, max_age)
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)]

    view_axes = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    view_names = ['sagittal', 'coronal', 'axial']

    for method_name, cam_class in methods.items():
        method_output_dir = os.path.join(output_dir, method_name)
        for bin_idx in range(len(bin_edges) - 1):
            bin_label = bin_labels[bin_idx]
            bin_lower = bin_edges[bin_idx]
            bin_upper = bin_edges[bin_idx + 1]
            bin_output_dir = os.path.join(method_output_dir, f"bin_{bin_label}")
            os.makedirs(bin_output_dir, exist_ok=True)

            binned_dataset = [sample for sample in dataset if sample is not None and bin_lower <= sample['age'] < bin_upper]
            if not binned_dataset:
                logging.warning(f"No data for age bin {bin_label}. Skipping.")
                continue

            loader = DataLoader(binned_dataset, batch_size=1, shuffle=False)
            all_slices_data = None

            for idx, sample in enumerate(tqdm(loader, desc=f"Processing images for {method_name}, bin {bin_label} (atlas)")):
                if sample is None:
                    logging.warning(f"Skipping None sample at index {idx}")
                    continue

                image, demographics, brain_age = sample['image'], sample['demographics'], sample['age']
                image = image.to(device)
                demographics = demographics.to(device)
                wrapped_model = BrainAgeWrapper(model, demographics)
                target_layers = get_target_layers(wrapped_model)
                cam = cam_class(model=wrapped_model, target_layers=target_layers)

                try:
                    grayscale_cam = cam(input_tensor=image.unsqueeze(0))
                    grayscale_cam = grayscale_cam[0, :]
                    heatmap = normalize_cam(grayscale_cam)
                    img_np = image.cpu().numpy().squeeze()

                    if idx == 0:
                        if atlas_data.shape != img_np.shape:
                            logging.info(f"Resizing atlas from {atlas_data.shape} to match image shape {img_np.shape}")
                            zoom_factors = [t / s for t, s in zip(img_np.shape, atlas_data.shape)]
                            resized_atlas = zoom(atlas_data, zoom_factors, order=0, mode='nearest')
                        else:
                            resized_atlas = atlas_data

                        all_slices_data = {view: {slice_idx: {'slices': [], 'heatmaps': []} for slice_idx in range(img_np.shape[view_axes[view]])} for view in view_names}

                    region_heatmap = compute_region_based_heatmap(heatmap, resized_atlas)

                    for view_name in view_names:
                        view_axis = view_axes[view_name]
                        for slice_idx in range(img_np.shape[view_axis]):
                            slice_data = np.take(img_np, indices=slice_idx, axis=view_axis)
                            heatmap_slice_data = np.take(region_heatmap, indices=slice_idx, axis=view_axis)
                            heatmap_resized_third = normalize_cam(heatmap_slice_data, slice_data.shape)
                            all_slices_data[view_name][slice_idx]['slices'].append(slice_data)
                            all_slices_data[view_name][slice_idx]['heatmaps'].append(heatmap_resized_third)

                except Exception as e:
                    logging.error(f"Error processing image with {method_name}, bin {bin_label} (atlas): {e}")
                    continue

            for view in view_names:
                for slice_idx in all_slices_data[view]:
                    slices_list = all_slices_data[view][slice_idx]['slices']
                    heatmaps_list = all_slices_data[view][slice_idx]['heatmaps']
                    if slices_list:
                        all_slices_data[view][slice_idx]['avg_slice'] = np.mean(np.stack(slices_list, axis=0), axis=0)
                        all_slices_data[view][slice_idx]['avg_heatmap'] = np.mean(np.stack(heatmaps_list, axis=0), axis=0)
                    else:
                        all_slices_data[view][slice_idx]['avg_slice'] = np.zeros((64,64))
                        all_slices_data[view][slice_idx]['avg_heatmap'] = np.zeros((64,64))

            for view_name in view_names:
                view_data = [(slice_idx, all_slices_data[view_name][slice_idx]['avg_slice'], all_slices_data[view_name][slice_idx]['avg_heatmap']) for slice_idx in sorted(all_slices_data[view_name].keys())]
                n_slices = len(view_data)
                if n_slices == 0:
                    continue
                n_cols = 12
                n_rows = ceil(n_slices / n_cols)

                fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), dpi=600)
                fig3.suptitle(f"All Slice Heatmaps (Atlas) - {method_name.capitalize()} - {view_name.capitalize()} - Bin: {bin_label}")

                if n_rows == 1:
                    axes3 = axes3[np.newaxis, :]

                for idx, (slice_idx, avg_slice, avg_heatmap) in enumerate(view_data):
                    row_idx = idx // n_cols
                    col_idx = idx % n_cols
                    ax = axes3[row_idx, col_idx] if axes3.ndim > 1 else axes3[col_idx]
                    ax.imshow(avg_slice, cmap='gray')
                    im = ax.imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
                    ax.set_title(f"Slice {slice_idx}")
                    ax.axis('off')

                if n_slices % n_cols != 0:
                    for j in range(n_slices % n_cols, n_cols):
                        (axes3[n_rows - 1, j] if axes3.ndim > 1 else axes3[j]).axis('off')
                cbar_ax = fig3.add_axes([0.125, 0.05, 0.775, 0.02]) 
                fig3.colorbar(mappable=im, cax=cbar_ax, # Use the last 'im' for simplicity.  Better to be explicit (see below).
                #ax=axes3.ravel().tolist(),  # Pass ALL axes.
                orientation='horizontal',
                label='Normalized CAM',
                shrink=0.6,  # Optional: Adjust size (e.g., make it slightly smaller)
                #  aspect=40,   # Optional: Adjust aspect ratio (make it wider)
                #  pad=-0.5     # Optional: Add some padding between colorbar and subplots.
                )
                plt.tight_layout()  # Adjust layout for title

                plt.savefig(os.path.join(bin_output_dir, f"all_slices_heatmaps_{view_name}_atlas.png"), dpi=600)
                plt.close(fig3)


def generate_xai_visualizations(model, dataset, output_dir, device='cuda', methods_to_run=['all'], atlas_path=None, age_bin_width = 10):
    """Main visualization function."""
    methods = create_visualization_dirs(output_dir, methods_to_run)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    atlas_data, _ = load_atlas(atlas_path)  # Keep atlas loading

    view_axes = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    view_names = ['sagittal', 'coronal', 'axial']

    for method_name, cam_class in methods.items():
        method_output_dir = os.path.join(output_dir, method_name)
        avg_middle_slices = {view: [] for view in view_names}
        avg_middle_heatmaps = {view: [] for view in view_names}
        avg_all_slices_view = {view: [] for view in view_names}
        avg_all_heatmaps_view = {view: [] for view in view_names}
        avg_every_third_slices_heatmaps = [] # List of tuples (view_name, slice_index, slice_data, heatmap_data)

        for idx, sample in enumerate(tqdm(loader, desc=f"Processing images for {method_name}")):
            if sample is None: # Handle None samples
                logging.warning(f"Skipping None sample at index {idx}")
                continue

            image, demographics, brain_age = sample['image'], sample['demographics'], sample['age'] # filename is a list
            image = image.to(device)
            demographics = demographics.to(device)
            wrapped_model = BrainAgeWrapper(model, demographics) # Wrap the model
            # targets = [ClassifierOutputTarget(0)] # For regression, target the output node directly.
            target_layers = get_target_layers(wrapped_model)
            

            cam = cam_class(model=wrapped_model, target_layers=target_layers)


            try:
                grayscale_cam = cam(input_tensor=image.unsqueeze(0))
                                 
                grayscale_cam = grayscale_cam[0, :] # Get rid of batch dimension
                heatmap = normalize_cam(grayscale_cam)
                img_np = image.cpu().numpy().squeeze() # Remove batch and channel dims


                for view_name in view_names:
                    view_axis = view_axes[view_name]
                    slice_index = img_np.shape[view_axis] // 2
                    original_slice = np.take(img_np, indices=slice_index, axis=view_axis)
                    heatmap_slice = np.take(heatmap, indices=slice_index, axis=view_axis)
                    heatmap_resized = normalize_cam(heatmap_slice, original_slice.shape)

                    avg_middle_slices[view_name].append(original_slice)
                    avg_middle_heatmaps[view_name].append(heatmap_resized)

                    all_slices_view = np.sum(img_np, axis=view_axis)
                    all_heatmaps_view = np.sum(heatmap, axis=view_axis)

                    avg_all_slices_view[view_name].append(all_slices_view)
                    avg_all_heatmaps_view[view_name].append(all_heatmaps_view)

                    for slice_idx in range(0, img_np.shape[view_axis]):
                        slice_data = np.take(img_np, indices=slice_idx, axis=view_axis)
                        heatmap_slice_data = np.take(heatmap, indices=slice_idx, axis=view_axis)
                        heatmap_resized_third = normalize_cam(heatmap_slice_data, slice_data.shape)
                        avg_every_third_slices_heatmaps.append((view_name, slice_idx, slice_data, heatmap_resized_third))


            except Exception as e:
                logging.error(f"Error processing image with {method_name}: {e}")
                tb_str = traceback.format_exc()
                logging.error(f"Traceback:\n{tb_str}")
                continue
        gc.collect() # Garbage collect after each image


        # --- Create Averaged Visualizations ---

        # Figure 1: Average middle slice
        fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10), dpi=900)
        fig1.suptitle(f"Average Middle Slice Heatmaps - {method_name.capitalize()}")
        for i, view_name in enumerate(view_names):
            #avg_slice = np.mean(np.stack(avg_middle_slices[view_name], axis=0), axis=0) if avg_middle_slices[view_name] else np.zeros_like(original_slice)
            if avg_middle_slices[view_name]:
                avg_slice = np.mean(np.stack(avg_middle_slices[view_name], axis=0), axis=0)
            else:
                avg_slice = np.zeros_like((64,64))
            avg_heatmap = np.mean(np.stack(avg_middle_heatmaps[view_name], axis=0), axis=0) if avg_middle_heatmaps[view_name] else np.zeros_like(original_slice)

            axes1[0, i].imshow(avg_slice, cmap='gray')
            axes1[0, i].set_title(f"{view_name.capitalize()} - Avg Slice")
            axes1[0, i].axis('off')

            axes1[1, i].imshow(avg_slice, cmap='gray')
            heatmap_im = axes1[1, i].imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
            #fig1.colorbar(heatmap_im, ax=axes1[1, i])
            axes1[1, i].set_title(f"{view_name.capitalize()} - Avg Heatmap")
            axes1[1, i].axis('off')
        fig1.colorbar(mappable=axes1[1, 2].images[1], ax=axes1[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
        plt.savefig(os.path.join(method_output_dir, "avg_middle_slice_heatmaps.png"))
        plt.close(fig1)


        # Figure 2: Average all slices combined
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10), dpi=900)
        fig2.suptitle(f"Average All Slices Combined Heatmaps - {method_name.capitalize()}")
        for i, view_name in enumerate(view_names):
            avg_slice_view = np.mean(np.stack(avg_all_slices_view[view_name], axis=0), axis=0) if avg_all_slices_view[view_name] else np.zeros_like(all_slices_view)
            avg_heatmap_view = np.mean(np.stack(avg_all_heatmaps_view[view_name], axis=0), axis=0) if avg_all_heatmaps_view[view_name] else np.zeros_like(all_heatmaps_view)

            axes2[0, i].imshow(avg_slice_view, cmap='gray')
            axes2[0, i].set_title(f"{view_name.capitalize()} - Avg Combined Slices")
            axes2[0, i].axis('off')

            axes2[1, i].imshow(avg_slice_view, cmap='gray')
            axes2[1, i].imshow(avg_heatmap_view, cmap='jet', alpha=0.5, interpolation='none')
            axes2[1, i].set_title(f"{view_name.capitalize()} - Avg Combined Heatmaps")
            axes2[1, i].axis('off')
        fig2.colorbar(mappable=axes2[1, 2].images[1], ax=axes2[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
        plt.savefig(os.path.join(method_output_dir, "avg_all_slices_heatmaps.png"))
        plt.close(fig2)


        # Figure 3: All slices (modified for separate images per view, 10 columns)
        for view_name in view_names:
            # Collect all slices for the current view
            view_data = []
            for v_name, slice_idx, slice_data, heatmap_data in avg_every_third_slices_heatmaps:
                if v_name == view_name:
                    view_data.append((slice_idx, slice_data, heatmap_data))
            
            # Sort by slice index to ensure correct order
            view_data.sort(key=lambda x: x[0])

            # Get unique slice indices. Since we are accumulating *every third* slice,
            # we need to reconstruct the *full* slice list.
            unique_slice_indices = sorted(list(set([x[0] for x in view_data])))

            # Reconstruct full slice list (assuming steps of 1 between original slices)
            full_slice_indices = []
            if unique_slice_indices: # Prevent error if list is empty
              min_slice = min(unique_slice_indices)
              max_slice = max(unique_slice_indices)
              full_slice_indices = list(range(min_slice, max_slice+1))


            all_slices_data = []
            for slice_idx in full_slice_indices:

                slice_found = False
                for v_idx, v_slice, v_heatmap in view_data:
                    if v_slice.ndim !=2 :
                        v_slice = np.zeros((100,100))
                        v_heatmap = np.zeros((100,100))
                    if v_idx == slice_idx:
                      all_slices_data.append((slice_idx, v_slice, v_heatmap))
                      slice_found = True
                      break #inner

                if not slice_found: # Fill missing with empty
                    all_slices_data.append((slice_idx, np.zeros_like(view_data[0][1] if view_data else np.zeros((10,10))), np.zeros_like(view_data[0][2] if view_data else np.zeros((10,10)) )))

            view_data = all_slices_data
            n_slices = len(view_data)
            n_cols = 12
            n_rows = ceil(n_slices / n_cols)

            if n_slices == 0:
                continue

            fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), dpi=900)  # Increased DPI and figsize
            fig3.suptitle(f"All Slice Heatmaps - {method_name.capitalize()} - {view_name.capitalize()}")

            if n_rows == 1:
                axes3 = axes3[np.newaxis, :]

            for idx, (slice_idx, avg_slice, avg_heatmap) in enumerate(view_data):
                row_idx = idx // n_cols
                col_idx = idx % n_cols

                if axes3.ndim == 1:  # Handle single-row case
                    ax = axes3[col_idx]
                else:
                    ax = axes3[row_idx, col_idx]


                ax.imshow(avg_slice, cmap='gray')
                im = ax.imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
                ax.set_title(f"Slice {slice_idx}")
                ax.axis('off')

            # Turn off unused axes
            if n_slices % n_cols != 0:
                for j in range(n_slices % n_cols, n_cols):
                    if axes3.ndim == 1:
                        axes3[j].axis('off')
                    else:
                        axes3[n_rows - 1, j].axis('off')

            #add colorbar to bottom of the plot
            #fig3.colorbar(mappable=axes3[1, 2].images[1], ax=axes3[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
            #fig3.colorbar(mappable=axes3[1, 2].images[1], ax=axes3[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
            #plt.tight_layout(rect=[0, 0.02, 1, 0.95]) 
            cbar_ax = fig3.add_axes([0.125, 0.05, 0.775, 0.02]) 
            fig3.colorbar(mappable=im, cax=cbar_ax, # Use the last 'im' for simplicity.  Better to be explicit (see below).
             #ax=axes3.ravel().tolist(),  # Pass ALL axes.
             orientation='horizontal',
             label='Normalized CAM',
              shrink=0.6,  # Optional: Adjust size (e.g., make it slightly smaller)
            #  aspect=40,   # Optional: Adjust aspect ratio (make it wider)
            #  pad=-0.5     # Optional: Add some padding between colorbar and subplots.
             )
            plt.tight_layout()  # Adjust layout for title
            #plt.tight_layout(rect=[0, 0.15, 1, 0.95])
            
            plt.savefig(os.path.join(method_output_dir, f"all_slices_heatmaps_{view_name}.png"), dpi=600) # Save with increased DPI
            plt.close(fig3)

            
def generate_xai_visualizations_binned(model, dataset, output_dir, device='cuda', methods_to_run=['all'], atlas_path=None, age_bin_width=10):
    """
    Generates XAI visualizations, binning the data by age ranges and creating separate plots for each bin.

    Args:
        model: The trained PyTorch model.
        dataset: The dataset (must be BrainAgeDataset or compatible).
        output_dir: Base output directory.
        device:  'cuda' or 'cpu'.
        methods_to_run: List of XAI methods (or 'all').
        atlas_path: Path to the brain atlas NIfTI file.
        age_bin_width (int or list):  The number of age bins (int) or the bin edges (list).
                                  If an integer is provided, it creates equally spaced bins.
    """

    methods = create_visualization_dirs(output_dir, methods_to_run)
    model.eval()

    # --- Binning Setup ---
    if isinstance(age_bin_width, int):
        # Create bins of specified width
        #min_age = min([sample['age'] for sample in dataset if sample is not None]).item()
        max_age = max([sample['age'] for sample in dataset if sample is not None]).item()
        bin_edges = np.arange(0, max_age + age_bin_width, age_bin_width)
        # Ensure the last bin includes the maximum age
        if bin_edges[-1] < max_age:
            bin_edges = np.append(bin_edges, max_age)


    else:
        raise ValueError("age_bin_width must be an integer (width of bins).")


    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)]


    view_axes = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    view_names = ['sagittal', 'coronal', 'axial']


    for method_name, cam_class in methods.items():
        method_output_dir = os.path.join(output_dir, method_name)

        for bin_idx in range(len(bin_edges) - 1):
            bin_label = bin_labels[bin_idx]
            bin_lower = bin_edges[bin_idx]
            bin_upper = bin_edges[bin_idx + 1]

            bin_output_dir = os.path.join(method_output_dir, f"bin_{bin_label}")
            os.makedirs(bin_output_dir, exist_ok=True)
            
            avg_middle_slices = {view: [] for view in view_names}
            avg_middle_heatmaps = {view: [] for view in view_names}
            avg_all_slices_view = {view: [] for view in view_names}
            avg_all_heatmaps_view = {view: [] for view in view_names}
            avg_every_third_slices_heatmaps = [] # List of tuples (view_name, slice_index, slice_data, heatmap_data)


            # --- Data Filtering for the Current Bin ---
            binned_dataset = [
                sample for sample in dataset
                if sample is not None and bin_lower <= sample['age'] < bin_upper
            ]

            if not binned_dataset:
                logging.warning(f"No data found for age bin {bin_label}. Skipping.")
                continue  # Skip to the next bin

            loader = DataLoader(binned_dataset, batch_size=1, shuffle=False)

            for idx, sample in enumerate(tqdm(loader, desc=f"Processing images for {method_name}, bin {bin_label}")):
                if sample is None: # Handle None samples
                    logging.warning(f"Skipping None sample at index {idx}")
                    continue
                
                image, demographics, brain_age = sample['image'], sample['demographics'], sample['age']
                image = image.to(device)
                demographics = demographics.to(device)

                wrapped_model = BrainAgeWrapper(model, demographics)
                target_layers = get_target_layers(wrapped_model)
                cam = cam_class(model=wrapped_model, target_layers=target_layers)

                try:
                    grayscale_cam = cam(input_tensor=image.unsqueeze(0))
                    grayscale_cam = grayscale_cam[0, :]
                    heatmap = normalize_cam(grayscale_cam)
                    img_np = image.cpu().numpy().squeeze()
                    
                    for view_name in view_names:
                        view_axis = view_axes[view_name]
                        slice_index = img_np.shape[view_axis] // 2
                        original_slice = np.take(img_np, indices=slice_index, axis=view_axis)
                        heatmap_slice = np.take(heatmap, indices=slice_index, axis=view_axis)
                        heatmap_resized = normalize_cam(heatmap_slice, original_slice.shape)

                        avg_middle_slices[view_name].append(original_slice)
                        avg_middle_heatmaps[view_name].append(heatmap_resized)

                        all_slices_view = np.sum(img_np, axis=view_axis)
                        all_heatmaps_view = np.sum(heatmap, axis=view_axis)

                        avg_all_slices_view[view_name].append(all_slices_view)
                        avg_all_heatmaps_view[view_name].append(all_heatmaps_view)

                        for slice_idx in range(0, img_np.shape[view_axis]):
                            slice_data = np.take(img_np, indices=slice_idx, axis=view_axis)
                            heatmap_slice_data = np.take(heatmap, indices=slice_idx, axis=view_axis)
                            heatmap_resized_third = normalize_cam(heatmap_slice_data, slice_data.shape)
                            avg_every_third_slices_heatmaps.append((view_name, slice_idx, slice_data, heatmap_resized_third))
                
                except Exception as e:
                    logging.error(f"Error processing image with {method_name}, bin {bin_label}: {e}")
                    tb_str = traceback.format_exc()
                    logging.error(f"Traceback:\n{tb_str}")
                    continue
            gc.collect()



            # --- Create Averaged Visualizations (Binned) ---
            
            # Figure 1: Average middle slice
            fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10), dpi=900)
            fig1.suptitle(f"Average Middle Slice Heatmaps - {method_name.capitalize()} - Bin: {bin_label}")
            for i, view_name in enumerate(view_names):
                if avg_middle_slices[view_name]:
                    avg_slice = np.mean(np.stack(avg_middle_slices[view_name], axis=0), axis=0)
                else:  # Handle empty bin
                    avg_slice = np.zeros((64, 64))
                    
                avg_heatmap = np.mean(np.stack(avg_middle_heatmaps[view_name], axis=0), axis=0) if avg_middle_heatmaps[view_name] else np.zeros_like(avg_slice)
                
                axes1[0, i].imshow(avg_slice, cmap='gray')
                axes1[0, i].set_title(f"{view_name.capitalize()} - Avg Slice")
                axes1[0, i].axis('off')
                
                axes1[1, i].imshow(avg_slice, cmap='gray')
                heatmap_im = axes1[1, i].imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
                #fig1.colorbar(heatmap_im, ax=axes1[1, i])
                axes1[1, i].set_title(f"{view_name.capitalize()} - Avg Heatmap")
                axes1[1, i].axis('off')
                #add colorbar to axes[1,i]
                
                
            fig1.colorbar(mappable=axes1[1, 2].images[1], ax=axes1[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
            plt.savefig(os.path.join(bin_output_dir, "avg_middle_slice_heatmaps.png"))
            plt.close(fig1)
            
            # Figure 2: Average all slices combined
            fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10), dpi=900)
            fig2.suptitle(f"Average All Slices Combined Heatmaps - {method_name.capitalize()} - Bin: {bin_label}")
            for i, view_name in enumerate(view_names):
                
                avg_slice_view = np.mean(np.stack(avg_all_slices_view[view_name], axis=0), axis=0) if avg_all_slices_view[view_name] else np.zeros_like(avg_all_slices_view[view_name][0] if avg_all_slices_view[view_name] else (64,64))
                avg_heatmap_view = np.mean(np.stack(avg_all_heatmaps_view[view_name], axis=0), axis=0) if avg_all_heatmaps_view[view_name] else np.zeros_like(avg_slice_view)
                
                axes2[0, i].imshow(avg_slice_view, cmap='gray')
                axes2[0, i].set_title(f"{view_name.capitalize()} - Avg Combined Slices")
                axes2[0, i].axis('off')
                
                axes2[1, i].imshow(avg_slice_view, cmap='gray')
                axes2[1, i].imshow(avg_heatmap_view, cmap='jet', alpha=0.5, interpolation='none')
                axes2[1, i].set_title(f"{view_name.capitalize()} - Avg Combined Heatmaps")
                axes2[1, i].axis('off')
            
            fig2.colorbar(mappable=axes2[1, 2].images[1], ax=axes2[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
            plt.savefig(os.path.join(bin_output_dir, "avg_all_slices_heatmaps.png"))
            plt.close(fig2)

            # Figure 3: All slices (modified for separate images per view, 10 columns)
            for view_name in view_names:
                # Collect all slices for the current view
                view_data = []
                for v_name, slice_idx, slice_data, heatmap_data in avg_every_third_slices_heatmaps:
                    if v_name == view_name:
                        view_data.append((slice_idx, slice_data, heatmap_data))
                
                # Sort by slice index to ensure correct order
                view_data.sort(key=lambda x: x[0])

                # Get unique slice indices. Since we are accumulating *every third* slice,
                # we need to reconstruct the *full* slice list.
                unique_slice_indices = sorted(list(set([x[0] for x in view_data])))

                # Reconstruct full slice list (assuming steps of 1 between original slices)
                full_slice_indices = []
                if unique_slice_indices: # Prevent error if list is empty
                  min_slice = min(unique_slice_indices)
                  max_slice = max(unique_slice_indices)
                  full_slice_indices = list(range(min_slice, max_slice+1))


                all_slices_data = []
                for slice_idx in full_slice_indices:

                    slice_found = False
                    for v_idx, v_slice, v_heatmap in view_data:
                        if v_slice.ndim !=2 :
                            v_slice = np.zeros((100,100))
                            v_heatmap = np.zeros((100,100))
                        if v_idx == slice_idx:
                          all_slices_data.append((slice_idx, v_slice, v_heatmap))
                          slice_found = True
                          break #inner

                    if not slice_found: # Fill missing with empty
                        all_slices_data.append((slice_idx, np.zeros_like(view_data[0][1] if view_data else np.zeros((10,10))), np.zeros_like(view_data[0][2] if view_data else np.zeros((10,10)) )))

                view_data = all_slices_data
                n_slices = len(view_data)
                n_cols = 12
                n_rows = ceil(n_slices / n_cols)

                if n_slices == 0:
                    continue

                fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), dpi=900)  # Increased DPI and figsize
                fig3.suptitle(f"All Slice Heatmaps - {method_name.capitalize()} - {view_name.capitalize()} - Bin: {bin_label}")

                if n_rows == 1:
                    axes3 = axes3[np.newaxis, :]

                for idx, (slice_idx, avg_slice, avg_heatmap) in enumerate(view_data):
                    row_idx = idx // n_cols
                    col_idx = idx % n_cols

                    if axes3.ndim == 1:  # Handle single-row case
                        ax = axes3[col_idx]
                    else:
                        ax = axes3[row_idx, col_idx]


                    ax.imshow(avg_slice, cmap='gray')
                    im = ax.imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
                    ax.set_title(f"Slice {slice_idx}")
                    ax.axis('off')

                # Turn off unused axes
                if n_slices % n_cols != 0:
                    for j in range(n_slices % n_cols, n_cols):
                        if axes3.ndim == 1:
                            axes3[j].axis('off')
                        else:
                            axes3[n_rows - 1, j].axis('off')

                
                cbar_ax = fig3.add_axes([0.125, 0.05, 0.775, 0.02]) 
                fig3.colorbar(mappable=im, cax=cbar_ax, # Use the last 'im' for simplicity.  Better to be explicit (see below).
                #ax=axes3.ravel().tolist(),  # Pass ALL axes.
                orientation='horizontal',
                label='Normalized CAM',
                shrink=0.6,  # Optional: Adjust size (e.g., make it slightly smaller)
                #  aspect=40,   # Optional: Adjust aspect ratio (make it wider)
                #  pad=-0.5     # Optional: Add some padding between colorbar and subplots.
                )
                plt.tight_layout()  # Adjust layout for title

                
                plt.savefig(os.path.join(bin_output_dir, f"all_slices_heatmaps_{view_name}.png"), dpi=600) # Save with increased DPI
                plt.close(fig3)
                
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
    
    generate_xai_visualizations_atlas(model, dataset, model_output_dir, device, methods_to_run, atlas_path)
    generate_xai_visualizations_atlas_binned(model, dataset, model_output_dir, device, methods_to_run, atlas_path, age_bin_width=10)    
    generate_xai_visualizations(model, dataset, model_output_dir, device,methods_to_run, atlas_path)
    generate_xai_visualizations_binned(model, dataset, model_output_dir, device,methods_to_run, atlas_path, age_bin_width=10)
    
    
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
    parser.add_argument('--method', type=str, default='all',
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