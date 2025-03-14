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
import gc
import matplotlib.pyplot as plt
import logging
import csv
from tqdm import tqdm
import re
import gc
from typing import List, Tuple, Optional
from functools import partial
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
from models.resnext3d import SEBlock3D

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
    # Add all regions up to 91 as per the HarvardOxford subcortical atlas
    # Refer to FSL's atlas documentation for the full list
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

def get_target_layers(wrappedmodel):
    """Get the target layers for visualization based on model type."""
    model = wrappedmodel.model
    model_name = model.__class__.__name__  # Access the wrapped model

    if model_name == 'Large3DCNN':
        return [model.conv_layers[-1]]  # Last Conv3d layer
    elif model_name == 'DenseNet3D':
        return [model.trans2[1]]  # Transition layer before last avg pool
    elif model_name == 'EfficientNet3D':
        return [model.conv_head]  # Head convolution before avg pool
    elif model_name == 'Improved3DCNN':
        # Access the last layer in the sequential conv_layers
        if isinstance(model.conv_layers[-1], nn.MaxPool3d):
            return [model.conv_layers[-5]]
        elif isinstance(model.conv_layers[-2], (nn.ReLU, SEBlock3D, nn.Identity)): #Explicitly verify
            return [model.conv_layers[-3]] # -2 will be BN, so go to -3
        else:
            raise ValueError("Unexpected layer structure in Improved3DCNN.conv_layers")        
        
    elif model_name == 'ResNet3D':
        return [model.layer3[-1].conv2]  # Last conv layer in last ResNet block of layer3
    elif model_name == 'ResNeXt3D':
        return [model.layer3[-1].conv3]  # Last conv layer in last ResNeXt block of layer3
    elif model_name == 'VisionTransformer3D':
        # Target the final convolutional layer within the HybridEmbed3D module if it's used.
        if model.use_hybrid_embed:
            return [model.embed.proj[-1]]  # Last layer of the HybridEmbed3D's projection
        # If not using hybrid embedding, target the final layer of the last transformer block.
        else:
            return [model.blocks[-1].norm2] #select the LayerNorm before the MLP block.

    else:
        return None  # Default or unknown model type


def normalize_cam(cam, target_size=None):
    """Memory-efficient normalization and resizing"""
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7) # Small value for numerical stability

    if target_size is not None: # Resize if target_size is provided
        #cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR) # Use cv2.resize
        cam_resized = cv2.resize(cam, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        return cam_resized.astype(np.float16)  # Use float16 for memory efficiency
    return cam.astype(np.float16)

def generate_xai_visualizations_binned(model, dataset, output_dir, device='cuda', methods_to_run=['all'], atlas_path=None, age_bin_width=10):
    methods = create_visualization_dirs(output_dir, methods_to_run)
    model.eval()

    # Binning setup
    max_age = max([sample['age'] for sample in dataset if sample is not None]).item()
    bin_edges = np.arange(0, max_age + age_bin_width, age_bin_width)
    if bin_edges[-1] < max_age:
        bin_edges = np.append(bin_edges, max_age)
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)]

    view_axes = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    view_names = ['sagittal', 'coronal', 'axial']

    # Get image dimensions
    first_sample = dataset[0]
    D, H, W = first_sample['image'].squeeze().shape

    for method_name, cam_class in methods.items():
        method_output_dir = os.path.join(output_dir, method_name)
        gc.collect()
        torch.cuda.empty_cache()
        for bin_idx in range(len(bin_edges) - 1):
            bin_label = bin_labels[bin_idx]
            bin_lower = bin_edges[bin_idx]
            bin_upper = bin_edges[bin_idx + 1]
            bin_output_dir = os.path.join(method_output_dir, f"bin_{bin_label}")
            os.makedirs(bin_output_dir, exist_ok=True)

            # Initialize sum arrays
            sum_middle_slices = {view: np.zeros((H, W) if view == 'sagittal' else (D, W) if view == 'coronal' else (D, H), dtype=np.float32) for view in view_names}
            sum_middle_heatmaps = {view: np.zeros((H, W) if view == 'sagittal' else (D, W) if view == 'coronal' else (D, H), dtype=np.float32) for view in view_names}
            sum_all_slices_view = {view: np.zeros((H, W) if view == 'sagittal' else (D, W) if view == 'coronal' else (D, H), dtype=np.float32) for view in view_names}
            sum_all_heatmaps_view = {view: np.zeros((H, W) if view == 'sagittal' else (D, W) if view == 'coronal' else (D, H), dtype=np.float32) for view in view_names}
            sum_slices_per_view = {
                'sagittal': np.zeros((D, H, W), dtype=np.float32),
                'coronal': np.zeros((H, D, W), dtype=np.float32),
                'axial': np.zeros((W, D, H), dtype=np.float32)
            }
            sum_heatmaps_per_view = {
                'sagittal': np.zeros((D, H, W), dtype=np.float32),
                'coronal': np.zeros((H, D, W), dtype=np.float32),
                'axial': np.zeros((W, D, H), dtype=np.float32)
            }
            count = 0

            # Filter dataset for this bin
            binned_dataset = [sample for sample in dataset if sample is not None and bin_lower <= sample['age'] < bin_upper]
            if not binned_dataset:
                logging.warning(f"No data found for age bin {bin_label}. Skipping.")
                continue
            loader = DataLoader(binned_dataset, batch_size=1, shuffle=False)

            # Accumulate sums
            for sample in tqdm(loader, desc=f"Processing images for {method_name}, bin {bin_label}"):
                if sample is None:
                    continue
                image, demographics, brain_age = sample['image'], sample['demographics'], sample['age']
                image = image.to(device)
                demographics = demographics.to(device)
                wrapped_model = BrainAgeWrapper(model, demographics)
                target_layers = get_target_layers(wrapped_model)
                cam = cam_class(model=wrapped_model, target_layers=target_layers)

                try:
                    grayscale_cam = cam(input_tensor=image.unsqueeze(0))[0, :]
                    heatmap = grayscale_cam
                    img_np = image.cpu().numpy().squeeze()

                    for view_name in view_names:
                        view_axis = view_axes[view_name]
                        slice_index = img_np.shape[view_axis] // 2
                        original_slice = np.take(img_np, indices=slice_index, axis=view_axis)
                        heatmap_slice = np.take(heatmap, indices=slice_index, axis=view_axis)
                        sum_middle_slices[view_name] += original_slice
                        sum_middle_heatmaps[view_name] += heatmap_slice

                        all_slices_view = np.sum(img_np, axis=view_axis)
                        all_heatmaps_view = np.sum(heatmap, axis=view_axis)
                        sum_all_slices_view[view_name] += all_slices_view
                        sum_all_heatmaps_view[view_name] += all_heatmaps_view

                        for slice_idx in range(img_np.shape[view_axis]):
                            slice_data = np.take(img_np, indices=slice_idx, axis=view_axis)
                            heatmap_slice_data = np.take(heatmap, indices=slice_idx, axis=view_axis)
                            if view_name == 'sagittal':
                                sum_slices_per_view['sagittal'][slice_idx] += slice_data
                                sum_heatmaps_per_view['sagittal'][slice_idx] += heatmap_slice_data
                            elif view_name == 'coronal':
                                sum_slices_per_view['coronal'][slice_idx] += slice_data
                                sum_heatmaps_per_view['coronal'][slice_idx] += heatmap_slice_data
                            elif view_name == 'axial':
                                sum_slices_per_view['axial'][slice_idx] += slice_data
                                sum_heatmaps_per_view['axial'][slice_idx] += heatmap_slice_data
                    count += 1
                except Exception as e:
                    logging.error(f"Error in bin {bin_label}: {e}")
                    continue
                del image, grayscale_cam, heatmap, img_np
                gc.collect()
            gc.collect()
            torch.cuda.empty_cache()
            # Compute averages
            if count == 0:
                logging.warning(f"No valid samples in bin {bin_label}")
                continue
            avg_middle_slices = {view: sum_middle_slices[view] / count for view in view_names}
            avg_middle_heatmaps = {view: sum_middle_heatmaps[view] / count for view in view_names}
            avg_all_slices_view = {view: sum_all_slices_view[view] / count for view in view_names}
            avg_all_heatmaps_view = {view: sum_all_heatmaps_view[view] / count for view in view_names}
            avg_slices_per_view = {view: sum_slices_per_view[view] / count for view in view_names}
            avg_heatmaps_per_view = {view: sum_heatmaps_per_view[view] / count for view in view_names}

            # Plotting (similar to non-binned version, just with bin_output_dir)
            # Figure 1
            fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10), dpi=900)
            fig1.suptitle(f"Average Middle Slice Heatmaps - {method_name.capitalize()} - Bin: {bin_label}")
            for i, view_name in enumerate(view_names):
                avg_slice = avg_middle_slices[view_name]
                avg_heatmap = normalize_cam(avg_middle_heatmaps[view_name], avg_slice.shape)
                axes1[0, i].imshow(avg_slice, cmap='gray')
                axes1[0, i].set_title(f"{view_name.capitalize()} - Avg Slice")
                axes1[0, i].axis('off')
                axes1[1, i].imshow(avg_slice, cmap='gray')
                heatmap_im = axes1[1, i].imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
                axes1[1, i].set_title(f"{view_name.capitalize()} - Avg Heatmap")
                axes1[1, i].axis('off')
            fig1.colorbar(mappable=heatmap_im, ax=axes1[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
            plt.savefig(os.path.join(bin_output_dir, "avg_middle_slice_heatmaps.png"))
            plt.close(fig1)

            # Figure 2
            fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10), dpi=900)
            fig2.suptitle(f"Average All Slices Combined Heatmaps - {method_name.capitalize()} - Bin: {bin_label}")
            for i, view_name in enumerate(view_names):
                avg_slice_view = avg_all_slices_view[view_name]
                avg_heatmap_view = normalize_cam(avg_all_heatmaps_view[view_name], avg_slice_view.shape)
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

            # Figure 3
            for view_name in view_names:
                n_slices = D if view_name == 'sagittal' else H if view_name == 'coronal' else W
                n_cols = 12
                n_rows = ceil(n_slices / n_cols)
                fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), dpi=900)
                fig3.suptitle(f"All Slice Heatmaps - {method_name.capitalize()} - {view_name.capitalize()} - Bin: {bin_label}")
                if n_rows == 1:
                    axes3 = axes3[np.newaxis, :]

                for slice_idx in range(n_slices):
                    row_idx = slice_idx // n_cols
                    col_idx = slice_idx % n_cols
                    ax = axes3[row_idx, col_idx]
                    avg_slice = avg_slices_per_view[view_name][slice_idx]
                    avg_heatmap = normalize_cam(avg_heatmaps_per_view[view_name][slice_idx], avg_slice.shape)
                    ax.imshow(avg_slice, cmap='gray')
                    im = ax.imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
                    ax.set_title(f"Slice {slice_idx}")
                    ax.axis('off')

                if n_slices % n_cols != 0:
                    for j in range(n_slices % n_cols, n_cols):
                        axes3[n_rows - 1, j].axis('off')
                cbar_ax = fig3.add_axes([0.125, 0.05, 0.775, 0.02])
                fig3.colorbar(mappable=im, cax=cbar_ax, orientation='horizontal', label='Normalized CAM', shrink=0.6)
                plt.tight_layout()
                plt.savefig(os.path.join(bin_output_dir, f"all_slices_heatmaps_{view_name}.png"), dpi=600)
                plt.close(fig3)

def generate_xai_visualizations(model, dataset, output_dir, device='cuda', methods_to_run=['all'], atlas_path=None, age_bin_width=10):
    methods = create_visualization_dirs(output_dir, methods_to_run)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    view_axes = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    view_names = ['sagittal', 'coronal', 'axial']

    # Get image dimensions from the first image
    first_sample = next(iter(loader))
    first_image = first_sample['image'].numpy().squeeze()
    D, H, W = first_image.shape

    for method_name, cam_class in methods.items():
        method_output_dir = os.path.join(output_dir, method_name)
        gc.collect()
        torch.cuda.empty_cache()
        # Initialize sum arrays (use float32 for precision; float16 possible but riskier)
        sum_middle_slices = {
            'sagittal': np.zeros((H, W), dtype=np.float32),
            'coronal': np.zeros((D, W), dtype=np.float32),
            'axial': np.zeros((D, H), dtype=np.float32)
        }
        sum_middle_heatmaps = {
            'sagittal': np.zeros((H, W), dtype=np.float32),
            'coronal': np.zeros((D, W), dtype=np.float32),
            'axial': np.zeros((D, H), dtype=np.float32)
        }
        sum_all_slices_view = {
            'sagittal': np.zeros((H, W), dtype=np.float32),
            'coronal': np.zeros((D, W), dtype=np.float32),
            'axial': np.zeros((D, H), dtype=np.float32)
        }
        sum_all_heatmaps_view = {
            'sagittal': np.zeros((H, W), dtype=np.float32),
            'coronal': np.zeros((D, W), dtype=np.float32),
            'axial': np.zeros((D, H), dtype=np.float32)
        }
        sum_slices_per_view = {
            'sagittal': np.zeros((D, H, W), dtype=np.float32),
            'coronal': np.zeros((H, D, W), dtype=np.float32),
            'axial': np.zeros((W, D, H), dtype=np.float32)
        }
        sum_heatmaps_per_view = {
            'sagittal': np.zeros((D, H, W), dtype=np.float32),
            'coronal': np.zeros((H, D, W), dtype=np.float32),
            'axial': np.zeros((W, D, H), dtype=np.float32)
        }
        count = 0

        # Process images and accumulate sums
        for sample in tqdm(loader, desc=f"Processing images for {method_name}"):
            if sample is None:
                logging.warning("Skipping None sample")
                continue

            image, demographics, brain_age = sample['image'], sample['demographics'], sample['age']
            image = image.to(device)
            demographics = demographics.to(device)
            wrapped_model = BrainAgeWrapper(model, demographics)
            target_layers = get_target_layers(wrapped_model)

            cam = cam_class(model=wrapped_model, target_layers=target_layers)
            try:
                grayscale_cam = cam(input_tensor=image.unsqueeze(0))[0, :]
                heatmap = grayscale_cam
                img_np = image.cpu().numpy().squeeze()

                for view_name in view_names:
                    view_axis = view_axes[view_name]
                    slice_index = img_np.shape[view_axis] // 2
                    original_slice = np.take(img_np, indices=slice_index, axis=view_axis)
                    heatmap_slice = np.take(heatmap, indices=slice_index, axis=view_axis)
                    sum_middle_slices[view_name] += original_slice
                    sum_middle_heatmaps[view_name] += heatmap_slice

                    all_slices_view = np.sum(img_np, axis=view_axis)
                    all_heatmaps_view = np.sum(heatmap, axis=view_axis)
                    sum_all_slices_view[view_name] += all_slices_view
                    sum_all_heatmaps_view[view_name] += all_heatmaps_view

                    for slice_idx in range(img_np.shape[view_axis]):
                        slice_data = np.take(img_np, indices=slice_idx, axis=view_axis)
                        heatmap_slice_data = np.take(heatmap, indices=slice_idx, axis=view_axis)
                        if view_name == 'sagittal':
                            sum_slices_per_view['sagittal'][slice_idx] += slice_data
                            sum_heatmaps_per_view['sagittal'][slice_idx] += heatmap_slice_data
                        elif view_name == 'coronal':
                            sum_slices_per_view['coronal'][slice_idx] += slice_data
                            sum_heatmaps_per_view['coronal'][slice_idx] += heatmap_slice_data
                        elif view_name == 'axial':
                            sum_slices_per_view['axial'][slice_idx] += slice_data
                            sum_heatmaps_per_view['axial'][slice_idx] += heatmap_slice_data
                count += 1
            except Exception as e:
                logging.error(f"Error processing image with {method_name}: {e}")
                continue
            del image, grayscale_cam, heatmap, img_np
            gc.collect()

        gc.collect()
        torch.cuda.empty_cache()
        # Compute averages
        if count == 0:
            logging.warning(f"No valid samples processed for {method_name}")
            continue
        avg_middle_slices = {view: sum_middle_slices[view] / count for view in view_names}
        avg_middle_heatmaps = {view: sum_middle_heatmaps[view] / count for view in view_names}
        avg_all_slices_view = {view: sum_all_slices_view[view] / count for view in view_names}
        avg_all_heatmaps_view = {view: sum_all_heatmaps_view[view] / count for view in view_names}
        avg_slices_per_view = {view: sum_slices_per_view[view] / count for view in view_names}
        avg_heatmaps_per_view = {view: sum_heatmaps_per_view[view] / count for view in view_names}

        # Plotting (unchanged except using averages directly)
        # Figure 1: Average middle slice
        fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10), dpi=900)
        fig1.suptitle(f"Average Middle Slice Heatmaps - {method_name.capitalize()}")
        for i, view_name in enumerate(view_names):
            avg_slice = avg_middle_slices[view_name]
            avg_heatmap = normalize_cam(avg_middle_heatmaps[view_name], avg_slice.shape)
            axes1[0, i].imshow(avg_slice, cmap='gray')
            axes1[0, i].set_title(f"{view_name.capitalize()} - Avg Slice")
            axes1[0, i].axis('off')
            axes1[1, i].imshow(avg_slice, cmap='gray')
            heatmap_im = axes1[1, i].imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
            axes1[1, i].set_title(f"{view_name.capitalize()} - Avg Heatmap")
            axes1[1, i].axis('off')
        fig1.colorbar(mappable=heatmap_im, ax=axes1[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
        plt.savefig(os.path.join(method_output_dir, "avg_middle_slice_heatmaps.png"))
        plt.close(fig1)

        # Figure 2: Average all slices combined
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10), dpi=900)
        fig2.suptitle(f"Average All Slices Combined Heatmaps - {method_name.capitalize()}")
        for i, view_name in enumerate(view_names):
            avg_slice_view = avg_all_slices_view[view_name]
            avg_heatmap_view = normalize_cam(avg_all_heatmaps_view[view_name], avg_slice_view.shape)
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

        # Figure 3: All slices
        for view_name in view_names:
            if view_name == 'sagittal':
                n_slices = D
            elif view_name == 'coronal':
                n_slices = H
            elif view_name == 'axial':
                n_slices = W
            n_cols = 12
            n_rows = ceil(n_slices / n_cols)
            fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), dpi=900)
            fig3.suptitle(f"All Slice Heatmaps - {method_name.capitalize()} - {view_name.capitalize()}")
            if n_rows == 1:
                axes3 = axes3[np.newaxis, :]

            for slice_idx in range(n_slices):
                row_idx = slice_idx // n_cols
                col_idx = slice_idx % n_cols
                ax = axes3[row_idx, col_idx]
                avg_slice = avg_slices_per_view[view_name][slice_idx]
                avg_heatmap = normalize_cam(avg_heatmaps_per_view[view_name][slice_idx], avg_slice.shape)
                ax.imshow(avg_slice, cmap='gray')
                im = ax.imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
                ax.set_title(f"Slice {slice_idx}")
                ax.axis('off')

            if n_slices % n_cols != 0:
                for j in range(n_slices % n_cols, n_cols):
                    axes3[n_rows - 1, j].axis('off')
            cbar_ax = fig3.add_axes([0.125, 0.05, 0.775, 0.02])
            fig3.colorbar(mappable=im, cax=cbar_ax, orientation='horizontal', label='Normalized CAM', shrink=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(method_output_dir, f"all_slices_heatmaps_{view_name}.png"), dpi=600)
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
    
    generate_xai_visualizations(model, dataset, model_output_dir, device,methods_to_run, atlas_path)
    gc.collect()
    torch.cuda.empty_cache()
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
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            tb_str = traceback.format_exc()
            logging.error(f"Error processing model {model_file}: {str(e)}\nTraceback:\n{tb_str}")
            continue

if __name__ == "__main__":
    main()