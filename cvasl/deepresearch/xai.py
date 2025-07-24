import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, HiResCAM, LayerCAM
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import argparse
import gc
import logging
import re
from math import ceil

import matplotlib.pyplot as plt
from tqdm import tqdm

torch.backends.cudnn.benchmark = True  # Optimizes CUDA operations
torch.cuda.empty_cache()
import traceback

import cv2
from matplotlib.colors import ListedColormap
from pytorch_grad_cam import GradCAM, HiResCAM, LayerCAM

from .data import BrainAgeDataset
from .models.resnext3d import SEBlock3D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class AgeFilteredDataset:
    """Dataset wrapper that filters samples based on age range."""
    
    def __init__(self, base_dataset, min_age, max_age):
        """
        Args:
            base_dataset: Original dataset
            min_age: Minimum age (inclusive)
            max_age: Maximum age (inclusive)
        """
        self.base_dataset = base_dataset
        self.min_age = min_age
        self.max_age = max_age
        
        # Create indices of samples within the age range
        self.valid_indices = []
        for i in range(len(base_dataset)):
            sample = base_dataset[i]
            if sample is not None and min_age <= sample['age'].item() <= max_age:
                self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")
        return self.base_dataset[self.valid_indices[idx]]

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


def normalize_cam(cam, target_size=None, preserve_zeros=True):
    """Memory-efficient normalization and resizing with option to preserve zero values"""
    cam = np.maximum(cam, 0)
    
    if preserve_zeros:
        # Only normalize non-zero values to preserve masked areas
        non_zero_mask = cam > 0
        if np.any(non_zero_mask):
            non_zero_values = cam[non_zero_mask]
            min_val = np.min(non_zero_values)
            max_val = np.max(non_zero_values)
            if max_val > min_val:
                cam[non_zero_mask] = (non_zero_values - min_val) / (max_val - min_val)
            # Zero values remain zero
    else:
        # Original normalization
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7)

    if target_size is not None:
        cam_resized = cv2.resize(cam, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        return cam_resized.astype(np.float16)
    return cam.astype(np.float16)

def generate_age_binned_xai_visualizations(model, dataset, output_dir, device='cuda', methods_to_run=['all'], age_bin_width=10):
    """
    Generate XAI visualizations for different age bins.
    
    Args:
        model: The trained model
        dataset: The dataset containing samples with 'age' field
        output_dir: Base directory for output visualizations
        device: Device to run the model on ('cuda' or 'cpu')
        methods_to_run: List of XAI methods to run
        age_bin_width: Width of age bins in years
    """
    # Calculate max age and create bin edges
    max_age = max([sample['age'] for sample in dataset if sample is not None]).item()
    bin_edges = np.arange(0, max_age + age_bin_width, age_bin_width)
    if bin_edges[-1] < max_age:
        bin_edges = np.append(bin_edges, max_age)
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)]
    
    # Create a directory for each age bin
    for i, bin_label in enumerate(bin_labels):
        bin_dir = os.path.join(output_dir, f"age_bin_{bin_label}")
        os.makedirs(bin_dir, exist_ok=True)
        
        # Filter dataset for current age bin
        bin_min, bin_max = bin_edges[i], bin_edges[i+1]
        bin_dataset = AgeFilteredDataset(dataset, bin_min, bin_max)
        
        # Skip empty bins
        if len(bin_dataset) == 0:
            logging.warning(f"No samples in age bin {bin_label}, skipping...")
            continue
        
        logging.info(f"Processing age bin {bin_label} with {len(bin_dataset)} samples")
        
        # Generate visualizations for this bin
        generate_xai_visualizations(
            model=model,
            dataset=bin_dataset,
            output_dir=bin_dir,
            device=device,
            methods_to_run=methods_to_run,
            output_prefix=f"age_bin_{bin_label}_"
        )

def generate_xai_visualizations(model, dataset, output_dir, device='cuda', methods_to_run=['all'], atlas_path=None, age_bin_width=10, output_prefix="", individual_patients=False):
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
        
        # Create subdirectories for the two approaches
        raw_dir = os.path.join(method_output_dir, "raw_accumulation")
        norm_dir = os.path.join(method_output_dir, "normalized_accumulation")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(norm_dir, exist_ok=True)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Initialize arrays for BOTH approaches
        # 1. Raw accumulation (original approach)
        sum_middle_slices_raw = {
            'sagittal': np.zeros((H, W), dtype=np.float32),
            'coronal': np.zeros((D, W), dtype=np.float32),
            'axial': np.zeros((D, H), dtype=np.float32)
        }
        sum_middle_heatmaps_raw = {
            'sagittal': np.zeros((H, W), dtype=np.float32),
            'coronal': np.zeros((D, W), dtype=np.float32),
            'axial': np.zeros((D, H), dtype=np.float32)
        }
        sum_all_slices_view_raw = {
            'sagittal': np.zeros((H, W), dtype=np.float32),
            'coronal': np.zeros((D, W), dtype=np.float32),
            'axial': np.zeros((D, H), dtype=np.float32)
        }
        sum_all_heatmaps_view_raw = {
            'sagittal': np.zeros((H, W), dtype=np.float32),
            'coronal': np.zeros((D, W), dtype=np.float32),
            'axial': np.zeros((D, H), dtype=np.float32)
        }
        sum_slices_per_view_raw = {
            'sagittal': np.zeros((D, H, W), dtype=np.float32),
            'coronal': np.zeros((H, D, W), dtype=np.float32),
            'axial': np.zeros((W, D, H), dtype=np.float32)
        }
        sum_heatmaps_per_view_raw = {
            'sagittal': np.zeros((D, H, W), dtype=np.float32),
            'coronal': np.zeros((H, D, W), dtype=np.float32),
            'axial': np.zeros((W, D, H), dtype=np.float32)
        }
        
        # 2. Normalized accumulation (new approach)
        sum_middle_slices_norm = {
            'sagittal': np.zeros((H, W), dtype=np.float32),
            'coronal': np.zeros((D, W), dtype=np.float32),
            'axial': np.zeros((D, H), dtype=np.float32)
        }
        sum_middle_heatmaps_norm = {
            'sagittal': np.zeros((H, W), dtype=np.float32),
            'coronal': np.zeros((D, W), dtype=np.float32),
            'axial': np.zeros((D, H), dtype=np.float32)
        }
        sum_all_slices_view_norm = {
            'sagittal': np.zeros((H, W), dtype=np.float32),
            'coronal': np.zeros((D, W), dtype=np.float32),
            'axial': np.zeros((D, H), dtype=np.float32)
        }
        sum_all_heatmaps_view_norm = {
            'sagittal': np.zeros((H, W), dtype=np.float32),
            'coronal': np.zeros((D, W), dtype=np.float32),
            'axial': np.zeros((D, H), dtype=np.float32)
        }
        sum_slices_per_view_norm = {
            'sagittal': np.zeros((D, H, W), dtype=np.float32),
            'coronal': np.zeros((H, D, W), dtype=np.float32),
            'axial': np.zeros((W, D, H), dtype=np.float32)
        }
        sum_heatmaps_per_view_norm = {
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
                # Get raw GradCAM heatmap
                grayscale_cam = cam(input_tensor=image.unsqueeze(0))[0, :]

                img_np = image.cpu().numpy().squeeze()
                
                # Create a boolean mask from the input image itself.
                brain_mask = img_np != 0
                
                # Re-apply the mask to the heatmap to eliminate upsampling artifacts.
                grayscale_cam[~brain_mask] = 0

                raw_heatmap = grayscale_cam
                
                # Get normalized GradCAM heatmap (for equal contribution)
                # First normalize the entire 3D heatmap
                normalized_heatmap = normalize_cam(grayscale_cam)
                
                img_np = image.cpu().numpy().squeeze()

                for view_name in view_names:
                    view_axis = view_axes[view_name]
                    slice_index = img_np.shape[view_axis] // 2
                    
                    # Get original image slice and both types of heatmap slices
                    original_slice = np.take(img_np, indices=slice_index, axis=view_axis)
                    raw_heatmap_slice = np.take(raw_heatmap, indices=slice_index, axis=view_axis)
                    norm_heatmap_slice = np.take(normalized_heatmap, indices=slice_index, axis=view_axis)
                    
                    # APPROACH 1: Raw accumulation
                    sum_middle_slices_raw[view_name] += original_slice
                    sum_middle_heatmaps_raw[view_name] += raw_heatmap_slice
                    
                    # APPROACH 2: Normalized accumulation
                    sum_middle_slices_norm[view_name] += original_slice  # Same original slices
                    sum_middle_heatmaps_norm[view_name] += norm_heatmap_slice  # But normalized heatmaps
                    
                    # All slices view - raw accumulation
                    all_slices_view = np.sum(img_np, axis=view_axis)
                    all_raw_heatmaps_view = np.sum(raw_heatmap, axis=view_axis)
                    sum_all_slices_view_raw[view_name] += all_slices_view
                    sum_all_heatmaps_view_raw[view_name] += all_raw_heatmaps_view
                    
                    # All slices view - normalized accumulation
                    all_norm_heatmaps_view = np.sum(normalized_heatmap, axis=view_axis)
                    sum_all_slices_view_norm[view_name] += all_slices_view
                    sum_all_heatmaps_view_norm[view_name] += all_norm_heatmaps_view
                    
                    # Process each slice - for both approaches
                    for slice_idx in range(img_np.shape[view_axis]):
                        slice_data = np.take(img_np, indices=slice_idx, axis=view_axis)
                        raw_heatmap_slice_data = np.take(raw_heatmap, indices=slice_idx, axis=view_axis)
                        norm_heatmap_slice_data = np.take(normalized_heatmap, indices=slice_idx, axis=view_axis)
                        
                        if view_name == 'sagittal':
                            sum_slices_per_view_raw['sagittal'][slice_idx] += slice_data
                            sum_heatmaps_per_view_raw['sagittal'][slice_idx] += raw_heatmap_slice_data
                            sum_slices_per_view_norm['sagittal'][slice_idx] += slice_data
                            sum_heatmaps_per_view_norm['sagittal'][slice_idx] += norm_heatmap_slice_data
                        elif view_name == 'coronal':
                            sum_slices_per_view_raw['coronal'][slice_idx] += slice_data
                            sum_heatmaps_per_view_raw['coronal'][slice_idx] += raw_heatmap_slice_data
                            sum_slices_per_view_norm['coronal'][slice_idx] += slice_data
                            sum_heatmaps_per_view_norm['coronal'][slice_idx] += norm_heatmap_slice_data
                        elif view_name == 'axial':
                            sum_slices_per_view_raw['axial'][slice_idx] += slice_data
                            sum_heatmaps_per_view_raw['axial'][slice_idx] += raw_heatmap_slice_data
                            sum_slices_per_view_norm['axial'][slice_idx] += slice_data
                            sum_heatmaps_per_view_norm['axial'][slice_idx] += norm_heatmap_slice_data
                count += 1
            except Exception as e:
                logging.error(f"Error processing image with {method_name}: {e}")
                continue
            del image, grayscale_cam, raw_heatmap, normalized_heatmap, img_np
            gc.collect()

        gc.collect()
        torch.cuda.empty_cache()
        
        # Compute averages
        if count == 0:
            logging.warning(f"No valid samples processed for {method_name}")
            continue
            
        # Compute averages for both approaches
        # 1. Raw accumulation approach
        avg_middle_slices_raw = {view: sum_middle_slices_raw[view] / count for view in view_names}
        avg_middle_heatmaps_raw = {view: sum_middle_heatmaps_raw[view] / count for view in view_names}
        avg_all_slices_view_raw = {view: sum_all_slices_view_raw[view] / count for view in view_names}
        avg_all_heatmaps_view_raw = {view: sum_all_heatmaps_view_raw[view] / count for view in view_names}
        avg_slices_per_view_raw = {view: sum_slices_per_view_raw[view] / count for view in view_names}
        avg_heatmaps_per_view_raw = {view: sum_heatmaps_per_view_raw[view] / count for view in view_names}
        
        # 2. Normalized accumulation approach
        avg_middle_slices_norm = {view: sum_middle_slices_norm[view] / count for view in view_names}
        avg_middle_heatmaps_norm = {view: sum_middle_heatmaps_norm[view] / count for view in view_names}
        avg_all_slices_view_norm = {view: sum_all_slices_view_norm[view] / count for view in view_names}
        avg_all_heatmaps_view_norm = {view: sum_all_heatmaps_view_norm[view] / count for view in view_names}
        avg_slices_per_view_norm = {view: sum_slices_per_view_norm[view] / count for view in view_names}
        avg_heatmaps_per_view_norm = {view: sum_heatmaps_per_view_norm[view] / count for view in view_names}

        # Plot for the RAW ACCUMULATION approach
        # Figure 1: Average middle slice - Raw accumulation
        fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10), dpi=900)
        fig1.suptitle(f"Average Middle Slice Heatmaps (Raw Accumulation) - {method_name.capitalize()}")
        for i, view_name in enumerate(view_names):
            avg_slice = avg_middle_slices_raw[view_name]
            avg_heatmap = normalize_cam(avg_middle_heatmaps_raw[view_name], avg_slice.shape)
            axes1[0, i].imshow(avg_slice, cmap='gray')
            axes1[0, i].set_title(f"{view_name.capitalize()} - Avg Slice")
            axes1[0, i].axis('off')
            axes1[1, i].imshow(avg_slice, cmap='gray')
            heatmap_im = axes1[1, i].imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
            axes1[1, i].set_title(f"{view_name.capitalize()} - Avg Heatmap")
            axes1[1, i].axis('off')
        fig1.colorbar(mappable=heatmap_im, ax=axes1[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
        plt.savefig(os.path.join(raw_dir, f"{output_prefix}avg_middle_slice_heatmaps.png"))
        plt.close(fig1)

        # Figure 2: Average all slices combined - Raw accumulation
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10), dpi=900)
        fig2.suptitle(f"Average All Slices Combined Heatmaps (Raw Accumulation) - {method_name.capitalize()}")
        for i, view_name in enumerate(view_names):
            avg_slice_view = avg_all_slices_view_raw[view_name]
            avg_heatmap_view = normalize_cam(avg_all_heatmaps_view_raw[view_name], avg_slice_view.shape)
            axes2[0, i].imshow(avg_slice_view, cmap='gray')
            axes2[0, i].set_title(f"{view_name.capitalize()} - Avg Combined Slices")
            axes2[0, i].axis('off')
            axes2[1, i].imshow(avg_slice_view, cmap='gray')
            axes2[1, i].imshow(avg_heatmap_view, cmap='jet', alpha=0.5, interpolation='none')
            axes2[1, i].set_title(f"{view_name.capitalize()} - Avg Combined Heatmaps")
            axes2[1, i].axis('off')
        fig2.colorbar(mappable=axes2[1, 2].images[1], ax=axes2[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
        plt.savefig(os.path.join(raw_dir, f"{output_prefix}avg_all_slices_heatmaps.png"))
        plt.close(fig2)

        # Figure 3: All slices - Raw accumulation
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
            fig3.suptitle(f"All Slice Heatmaps (Raw Accumulation) - {method_name.capitalize()} - {view_name.capitalize()}")
            if n_rows == 1:
                axes3 = axes3[np.newaxis, :]

            for slice_idx in range(n_slices):
                row_idx = slice_idx // n_cols
                col_idx = slice_idx % n_cols
                ax = axes3[row_idx, col_idx]
                avg_slice = avg_slices_per_view_raw[view_name][slice_idx]
                avg_heatmap = normalize_cam(avg_heatmaps_per_view_raw[view_name][slice_idx], avg_slice.shape)
                ax.imshow(avg_slice, cmap='gray')
                im = ax.imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
                ax.set_title(f"Slice {slice_idx}")
                ax.axis('off')

            if n_slices % n_cols != 0:
                for j in range(n_slices % n_cols, n_cols):
                    axes3[n_rows - 1, j].axis('off')
            cbar_ax = fig3.add_axes([0.125, 0.05, 0.775, 0.02])
            fig3.colorbar(mappable=im, cax=cbar_ax, orientation='horizontal', label='Normalized CAM', shrink=0.6)
            #plt.tight_layout()
            plt.savefig(os.path.join(raw_dir, f"{output_prefix}all_slices_heatmaps_{view_name}.png"), dpi=600)
            plt.close(fig3)
            
        # Now plot for the NORMALIZED ACCUMULATION approach
        # Figure 1: Average middle slice - Normalized accumulation
        fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10), dpi=900)
        fig1.suptitle(f"Average Middle Slice Heatmaps (Normalized Accumulation) - {method_name.capitalize()}")
        for i, view_name in enumerate(view_names):
            avg_slice = avg_middle_slices_norm[view_name]
            # Note: The heatmaps are already normalized before accumulation, but we re-normalize for visualization
            avg_heatmap = normalize_cam(avg_middle_heatmaps_norm[view_name], avg_slice.shape)
            axes1[0, i].imshow(avg_slice, cmap='gray')
            axes1[0, i].set_title(f"{view_name.capitalize()} - Avg Slice")
            axes1[0, i].axis('off')
            axes1[1, i].imshow(avg_slice, cmap='gray')
            heatmap_im = axes1[1, i].imshow(avg_heatmap, cmap='jet', alpha=0.5, interpolation='none')
            axes1[1, i].set_title(f"{view_name.capitalize()} - Avg Heatmap")
            axes1[1, i].axis('off')
        fig1.colorbar(mappable=heatmap_im, ax=axes1[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
        plt.savefig(os.path.join(norm_dir, f"{output_prefix}avg_middle_slice_heatmaps.png"))
        plt.close(fig1)

        # Figure 2: Average all slices combined - Normalized accumulation
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10), dpi=900)
        fig2.suptitle(f"Average All Slices Combined Heatmaps (Normalized Accumulation) - {method_name.capitalize()}")
        for i, view_name in enumerate(view_names):
            avg_slice_view = avg_all_slices_view_norm[view_name]
            avg_heatmap_view = normalize_cam(avg_all_heatmaps_view_norm[view_name], avg_slice_view.shape)
            axes2[0, i].imshow(avg_slice_view, cmap='gray')
            axes2[0, i].set_title(f"{view_name.capitalize()} - Avg Combined Slices")
            axes2[0, i].axis('off')
            axes2[1, i].imshow(avg_slice_view, cmap='gray')
            axes2[1, i].imshow(avg_heatmap_view, cmap='jet', alpha=0.5, interpolation='none')
            axes2[1, i].set_title(f"{view_name.capitalize()} - Avg Combined Heatmaps")
            axes2[1, i].axis('off')
        fig2.colorbar(mappable=axes2[1, 2].images[1], ax=axes2[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
        plt.savefig(os.path.join(norm_dir, f"{output_prefix}avg_all_slices_heatmaps.png"))
        plt.close(fig2)

        # Figure 3: All slices - Normalized accumulation
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
            fig3.suptitle(f"All Slice Heatmaps (Normalized Accumulation) - {method_name.capitalize()} - {view_name.capitalize()}")
            if n_rows == 1:
                axes3 = axes3[np.newaxis, :]

            for slice_idx in range(n_slices):
                row_idx = slice_idx // n_cols
                col_idx = slice_idx % n_cols
                ax = axes3[row_idx, col_idx]
                avg_slice = avg_slices_per_view_norm[view_name][slice_idx]
                avg_heatmap = normalize_cam(avg_heatmaps_per_view_norm[view_name][slice_idx], avg_slice.shape)
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
            plt.savefig(os.path.join(norm_dir, f"{output_prefix}all_slices_heatmaps_{view_name}.png"), dpi=600)
            plt.close(fig3)

    if individual_patients:
        logging.info("Generating individual patient visualizations...")
        generate_individual_patient_visualizations(model, dataset, output_dir, device, methods_to_run)            

def plot_heatmap_with_masked_colormap(ax, original_slice, heatmap_slice, alpha=0.5):
    """Plot heatmap with proper masking of zero values"""
    # Create a custom colormap where the lowest value is fully transparent
    cmap = plt.cm.jet
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)  # First value transparent
    my_cmap = ListedColormap(my_cmap)
    
    # Apply a strict mask to heatmap - ensure zeros are truly zero
    masked_heatmap = np.copy(heatmap_slice)
    mask = masked_heatmap < 0.01  # Consider anything below 0.01 as background
    masked_heatmap[mask] = np.nan  # NaN values will be transparent
    
    # Plot
    ax.imshow(original_slice, cmap='gray')
    im = ax.imshow(masked_heatmap, cmap=my_cmap, alpha=alpha, interpolation='none')
    return im

def generate_individual_patient_visualizations(model, dataset, output_dir, device='cuda', methods_to_run=['all']):
    """
    Generate XAI visualizations for individual patients.
    
    Args:
        model: The trained model
        dataset: The dataset containing samples
        output_dir: Base directory for output visualizations
        device: Device to run the model on ('cuda' or 'cpu')
        methods_to_run: List of XAI methods to run
    """
    methods = create_visualization_dirs(output_dir, methods_to_run)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    view_axes = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    view_names = ['sagittal', 'coronal', 'axial']

    for method_name, cam_class in methods.items():
        method_output_dir = os.path.join(output_dir, method_name)
        individual_dir = os.path.join(method_output_dir, "individuals")
        os.makedirs(individual_dir, exist_ok=True)
        
        gc.collect()
        torch.cuda.empty_cache()

        # Process each patient individually
        for patient_idx, sample in enumerate(tqdm(loader, desc=f"Processing individual patients for {method_name}")):
            if sample is None:
                logging.warning(f"Skipping None sample for patient {patient_idx}")
                continue

            try:
                image, demographics, brain_age = sample['image'], sample['demographics'], sample['age']
                # Get patient ID from the sample if available, otherwise use index
                patient_id = sample.get('patient_id', f"patient_{patient_idx:04d}")
                if isinstance(patient_id, torch.Tensor):
                    patient_id = patient_id.item() if patient_id.numel() == 1 else str(patient_id)
                patient_id = str(patient_id).replace('/', '_').replace('\\', '_')  # Sanitize for filename
                
                image = image.to(device)
                demographics = demographics.to(device)
                wrapped_model = BrainAgeWrapper(model, demographics)
                target_layers = get_target_layers(wrapped_model)

                cam = cam_class(model=wrapped_model, target_layers=target_layers)
                
                # Get GradCAM heatmap
                grayscale_cam = cam(input_tensor=image.unsqueeze(0))[0, :]

                img_np = image.cpu().numpy().squeeze()

                # Create a boolean mask from the input image itself.
                brain_mask = img_np != 0
                
                # Re-apply the mask to the heatmap to eliminate upsampling artifacts.
                grayscale_cam[~brain_mask] = 0

                normalized_heatmap = normalize_cam(grayscale_cam)
                
                D, H, W = img_np.shape


                # Figure 1: Middle slice heatmaps for this patient
                fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
                fig1.suptitle(f"Patient {patient_id} - Age {brain_age.item():.1f} - Middle Slice Heatmaps - {method_name.capitalize()}")
                
                for i, view_name in enumerate(view_names):
                    view_axis = view_axes[view_name]
                    slice_index = img_np.shape[view_axis] // 2
                    
                    # Get slices
                    original_slice = np.take(img_np, indices=slice_index, axis=view_axis)
                    heatmap_slice = np.take(normalized_heatmap, indices=slice_index, axis=view_axis)
                    heatmap_slice = normalize_cam(heatmap_slice, original_slice.shape)
                    
                    # Plot original slice
                    axes1[0, i].imshow(original_slice, cmap='gray')
                    axes1[0, i].set_title(f"{view_name.capitalize()} - Original")
                    axes1[0, i].axis('off')
                    
                    # Plot with heatmap overlay
                    axes1[1, i].imshow(original_slice, cmap='gray')
                    heatmap_im = axes1[1, i].imshow(heatmap_slice, cmap='jet', alpha=0.5, interpolation='none')
                    axes1[1, i].set_title(f"{view_name.capitalize()} - With Heatmap")
                    axes1[1, i].axis('off')
                
                fig1.colorbar(mappable=heatmap_im, ax=axes1[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
                plt.tight_layout()
                plt.savefig(os.path.join(individual_dir, f"{patient_id}_middle_slice_heatmaps.png"), dpi=300, bbox_inches='tight')
                plt.close(fig1)

                # Figure 2: All slices combined for this patient
                fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
                fig2.suptitle(f"Patient {patient_id} - Age {brain_age.item():.1f} - All Slices Combined - {method_name.capitalize()}")
                
                for i, view_name in enumerate(view_names):
                    view_axis = view_axes[view_name]
                    
                    # Sum all slices
                    all_slices_view = np.sum(img_np, axis=view_axis)
                    all_heatmaps_view = np.sum(normalized_heatmap, axis=view_axis)
                    all_heatmaps_view = normalize_cam(all_heatmaps_view, all_slices_view.shape)
                    
                    # Plot combined slices
                    axes2[0, i].imshow(all_slices_view, cmap='gray')
                    axes2[0, i].set_title(f"{view_name.capitalize()} - Combined Slices")
                    axes2[0, i].axis('off')
                    
                    # Plot with heatmap overlay
                    axes2[1, i].imshow(all_slices_view, cmap='gray')
                    axes2[1, i].imshow(all_heatmaps_view, cmap='jet', alpha=0.5, interpolation='none')
                    axes2[1, i].set_title(f"{view_name.capitalize()} - Combined Heatmaps")
                    axes2[1, i].axis('off')
                
                fig2.colorbar(mappable=axes2[1, 2].images[1], ax=axes2[1, :].ravel().tolist(), orientation='horizontal', label='Normalized CAM')
                plt.tight_layout()
                plt.savefig(os.path.join(individual_dir, f"{patient_id}_all_slices_combined_heatmaps.png"), dpi=300, bbox_inches='tight')
                plt.close(fig2)

                # Figure 3: Selected individual slices (every 8th slice to avoid too many images)
                for view_name in view_names:
                    view_axis = view_axes[view_name]
                    if view_name == 'sagittal':
                        n_total_slices = D
                    elif view_name == 'coronal':
                        n_total_slices = H
                    elif view_name == 'axial':
                        n_total_slices = W
                    
                    # Select every 8th slice or at least 6 slices
                    step = max(1, n_total_slices // 6)
                    selected_indices = range(0, n_total_slices, step)
                    n_selected = len(selected_indices)
                    
                    if n_selected > 0:
                        n_cols = min(3, n_selected)
                        n_rows = ceil(n_selected / n_cols)
                        
                        fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), dpi=300)
                        fig3.suptitle(f"Patient {patient_id} - Selected {view_name.capitalize()} Slices - {method_name.capitalize()}")
                        
                        if n_rows == 1 and n_cols == 1:
                            axes3 = np.array([[axes3]])
                        elif n_rows == 1:
                            axes3 = axes3[np.newaxis, :]
                        elif n_cols == 1:
                            axes3 = axes3[:, np.newaxis]
                        
                        for idx, slice_idx in enumerate(selected_indices):
                            row_idx = idx // n_cols
                            col_idx = idx % n_cols
                            ax = axes3[row_idx, col_idx]
                            
                            slice_data = np.take(img_np, indices=slice_idx, axis=view_axis)
                            heatmap_slice_data = np.take(normalized_heatmap, indices=slice_idx, axis=view_axis)
                            heatmap_slice_data = normalize_cam(heatmap_slice_data, slice_data.shape)
                            
                            ax.imshow(slice_data, cmap='gray')
                            im = ax.imshow(heatmap_slice_data, cmap='jet', alpha=0.5, interpolation='none')
                            ax.set_title(f"Slice {slice_idx}")
                            ax.axis('off')
                        
                        # Hide unused subplots
                        for idx in range(n_selected, n_rows * n_cols):
                            row_idx = idx // n_cols
                            col_idx = idx % n_cols
                            axes3[row_idx, col_idx].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(individual_dir, f"{patient_id}_selected_slices_{view_name}.png"), dpi=300, bbox_inches='tight')
                        plt.close(fig3)

            except Exception as e:
                logging.error(f"Error processing patient {patient_id} with {method_name}: {e}")
                continue
            
            # Clean up memory after each patient
            del image, grayscale_cam, normalized_heatmap, img_np
            gc.collect()

        gc.collect()
        torch.cuda.empty_cache()

def verify_brain_masking(dataset, output_dir=None, num_samples=50):
    """
    Verifies brain masking by plotting the middle slice in three orthogonal views 
    for images in the dataset and analyzes masking statistics across all samples.
    
    Args:
        dataset: BrainAgeDataset or list of samples
        output_dir: Directory to save the verification plots (defaults to './mask_verification')
        num_samples: Number of samples to check (default 50)
    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    
    if output_dir is None:
        output_dir = './mask_verification'
    os.makedirs(output_dir, exist_ok=True)
    
    view_names = ['sagittal', 'coronal', 'axial']
    view_axes = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    
    # Create a custom colormap with true black for zeros
    custom_gray = plt.cm.gray(np.arange(256))
    custom_gray[0] = [0, 0, 0, 1]  # Pure black for zero values
    custom_cmap = ListedColormap(custom_gray)
    
    # Determine if we're dealing with a dataset or a list
    if hasattr(dataset, '__getitem__') and not isinstance(dataset, list):
        # It's a dataset object
        sample_getter = lambda i: dataset[i]
        n_samples = min(num_samples, len(dataset))
    else:
        # It's a list of samples
        sample_getter = lambda i: dataset[i]
        n_samples = min(num_samples, len(dataset))
    
    # Statistics collection
    zero_percentages = []
    valid_samples = 0
    sample_stats = []
    
    for i in range(n_samples):
        sample = sample_getter(i)
        if sample is None:
            logging.info(f"Sample {i} is None, skipping")
            continue
            
        image = sample['image'].numpy() if torch.is_tensor(sample['image']) else sample['image']
        age = sample['age'].item() if torch.is_tensor(sample['age']) else sample['age']
        patient_id = sample.get('patient_id', f"patient_{i}")
        
        if isinstance(patient_id, torch.Tensor):
            patient_id = patient_id.item() if patient_id.numel() == 1 else str(patient_id)
        
        # Calculate masking statistics
        total_voxels = image.size
        zero_voxels = np.sum(image == 0)
        nonzero_voxels = total_voxels - zero_voxels
        percent_zero = (zero_voxels / total_voxels) * 100
        percent_nonzero = 100 - percent_zero
        
        # Store statistics
        zero_percentages.append(percent_zero)
        sample_stats.append({
            'patient_id': patient_id,
            'age': age,
            'total_voxels': total_voxels,
            'zero_voxels': zero_voxels,
            'percent_zero': percent_zero
        })
        valid_samples += 1
        
        # Only create detailed visualizations for first 5 samples to avoid too many files
        if i < 5:
            # Create figure with 4 rows and 3 columns
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))
            fig.suptitle(f"Patient {patient_id} - Age {age:.1f} - Mask Verification", fontsize=16)
            
            # Add text with statistics
            stats_text = (
                f"Total voxels: {total_voxels:,}\n"
                f"Zero voxels: {zero_voxels:,} ({percent_zero:.2f}%)\n"
                f"Non-zero voxels: {nonzero_voxels:,} ({percent_nonzero:.2f}%)"
            )
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            
            for j, view_name in enumerate(view_names):
                view_axis = view_axes[view_name]
                slice_index = image.shape[view_axis] // 2
                
                # Get middle slice
                original_slice = np.take(image, indices=slice_index, axis=view_axis)
                
                # Create binary mask (1 where brain is present, 0 elsewhere)
                binary_mask = original_slice != 0
                
                # Row 1: Original image with default colormap (shows how matplotlib renders by default)
                axes[0, j].imshow(original_slice, cmap='gray')
                axes[0, j].set_title(f"{view_name.capitalize()} - Default gray colormap")
                axes[0, j].axis('off')
                
                # Row 2: Original image with custom colormap (pure black for zeros)
                axes[1, j].imshow(original_slice, cmap=custom_cmap, vmin=0)
                axes[1, j].set_title(f"{view_name.capitalize()} - Custom colormap (black zeros)")
                axes[1, j].axis('off')
                
                # Row 3: Binary mask (white = brain, black = background)
                axes[2, j].imshow(binary_mask, cmap='binary')
                axes[2, j].set_title(f"{view_name.capitalize()} - Binary mask")
                axes[2, j].axis('off')
                
                # Row 4: Histogram of pixel values
                if np.any(binary_mask):
                    # Log-scale histogram of non-zero values
                    nonzero_values = original_slice[binary_mask]
                    axes[3, j].hist(nonzero_values.flatten(), bins=50, log=True)
                    axes[3, j].axvline(x=0, color='r', linestyle='--', alpha=0.7)
                    axes[3, j].set_title(f"{view_name.capitalize()} - Histogram (log scale)")
                else:
                    axes[3, j].text(0.5, 0.5, "No non-zero values", ha='center')
                    axes[3, j].set_title(f"{view_name.capitalize()} - No brain voxels")
                    axes[3, j].axis('off')
            
            plt.tight_layout(rect=[0, 0.04, 1, 0.96])
            plt.savefig(os.path.join(output_dir, f"mask_verification_patient_{patient_id}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Create a mask boundary visualization
            fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
            fig2.suptitle(f"Patient {patient_id} - Mask Boundary Visualization", fontsize=16)
            
            for j, view_name in enumerate(view_names):
                view_axis = view_axes[view_name]
                slice_index = image.shape[view_axis] // 2
                
                # Get original slice
                original_slice = np.take(image, indices=slice_index, axis=view_axis)
                brain_mask = original_slice != 0
                
                # Use masked array for proper visualization
                masked_data = np.ma.masked_where(~brain_mask, original_slice)
                
                # Plot with proper masking
                axes2[j].imshow(original_slice, cmap='gray', alpha=0.7)
                
                # Highlight the boundary
                from scipy import ndimage
                boundary = ndimage.binary_dilation(brain_mask, iterations=1) ^ brain_mask
                y, x = np.where(boundary)
                if len(y) > 0:
                    axes2[j].scatter(x, y, s=1, c='red', alpha=0.8)
                    
                axes2[j].set_title(f"{view_name.capitalize()} - Mask Boundary")
                axes2[j].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"mask_boundary_{patient_id}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig2)
    
    # Create histogram of zero percentages
    if zero_percentages:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Histogram
        ax1.hist(zero_percentages, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.axvline(np.mean(zero_percentages), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(zero_percentages):.2f}%')
        ax1.axvline(np.median(zero_percentages), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(zero_percentages):.2f}%')
        ax1.set_xlabel('Percentage of Zero Voxels')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title(f'Distribution of Zero Voxel Percentages (n={valid_samples} samples)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(zero_percentages, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_xlabel('Percentage of Zero Voxels')
        ax2.set_title('Box Plot of Zero Voxel Percentages')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'zero_voxels_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Calculate statistics
        mean_zero = np.mean(zero_percentages)
        median_zero = np.median(zero_percentages)
        std_zero = np.std(zero_percentages)
        min_zero = np.min(zero_percentages)
        max_zero = np.max(zero_percentages)
        q25_zero = np.percentile(zero_percentages, 25)
        q75_zero = np.percentile(zero_percentages, 75)
        
        # Analysis and recommendations
        print("\n" + "="*80)
        print("BRAIN MASKING ANALYSIS RESULTS")
        print("="*80)
        print(f"Analyzed {valid_samples} samples from the dataset")
        print()
        print("ZERO VOXEL STATISTICS:")
        print(f"  Mean:     {mean_zero:.2f}%")
        print(f"  Median:   {median_zero:.2f}%")
        print(f"  Std Dev:  {std_zero:.2f}%")
        print(f"  Min:      {min_zero:.2f}%")
        print(f"  Max:      {max_zero:.2f}%")
        print(f"  Q25:      {q25_zero:.2f}%")
        print(f"  Q75:      {q75_zero:.2f}%")
        print()
        
        # Masking quality assessment
        print("MASKING QUALITY ASSESSMENT:")
        print("-" * 40)
        
        # Expected range for properly masked brain images
        expected_min = 65  # Conservative lower bound
        expected_max = 90  # Liberal upper bound
        
        masking_quality = "UNKNOWN"
        recommendations = []
        
        if mean_zero < expected_min:
            masking_quality = "LIKELY INSUFFICIENT"
            recommendations.append("• Masking appears too liberal - background regions may be included")
            recommendations.append("• Consider using a more restrictive brain extraction algorithm")
            recommendations.append("• Check if skull stripping was properly applied")
        elif mean_zero > expected_max:
            masking_quality = "LIKELY TOO AGGRESSIVE"
            recommendations.append("• Masking appears too conservative - brain tissue may be excluded")
            recommendations.append("• Consider using a more liberal brain extraction algorithm")
            recommendations.append("• Check for over-erosion of brain boundaries")
        else:
            if std_zero < 5:
                masking_quality = "GOOD - CONSISTENT"
                recommendations.append("• Masking appears appropriate and consistent across samples")
                recommendations.append("• Zero voxel percentages are within expected range")
            elif std_zero > 10:
                masking_quality = "INCONSISTENT"
                recommendations.append("• High variability in masking quality across samples")
                recommendations.append("• Some samples may be poorly masked")
                recommendations.append("• Consider reviewing individual cases with extreme values")
            else:
                masking_quality = "GOOD - MODERATE VARIATION"
                recommendations.append("• Masking appears generally appropriate")
                recommendations.append("• Some natural variation in brain sizes accounts for the spread")
        
        # Additional checks
        outliers_low = np.sum(np.array(zero_percentages) < (mean_zero - 2*std_zero))
        outliers_high = np.sum(np.array(zero_percentages) > (mean_zero + 2*std_zero))
        
        print(f"Overall Assessment: {masking_quality}")
        print()
        
        if outliers_low > 0 or outliers_high > 0:
            print(f"OUTLIERS DETECTED:")
            print(f"  {outliers_low} samples with unusually low zero percentages (< {mean_zero - 2*std_zero:.2f}%)")
            print(f"  {outliers_high} samples with unusually high zero percentages (> {mean_zero + 2*std_zero:.2f}%)")
            recommendations.append("• Review outlier samples for masking artifacts")
            print()
        
        print("RECOMMENDATIONS:")
        for rec in recommendations:
            print(rec)
        
        # Save detailed statistics to CSV
        stats_df = pd.DataFrame(sample_stats)
        stats_df.to_csv(os.path.join(output_dir, 'masking_statistics.csv'), index=False)
        
        print()
        print(f"Detailed results saved to: {output_dir}")
        print(f"- Individual sample visualizations (first 5 samples)")
        print(f"- Zero voxel distribution histogram")
        print(f"- Detailed statistics CSV file")
        print("="*80)
        
        return {
            'mean_zero_percent': mean_zero,
            'median_zero_percent': median_zero,
            'std_zero_percent': std_zero,
            'quality_assessment': masking_quality,
            'valid_samples': valid_samples,
            'zero_percentages': zero_percentages
        }
    else:
        print("No valid samples found for analysis")
        return None
    

def process_single_model(csv_path, model_path, test_data_dir, base_output_dir, device, methods_to_run=['all'], atlas_path=None, indices_path=None, individual_patients=False):
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

    dataset = BrainAgeDataset(csv_path, test_data_dir, mask_path=test_data_dir, indices=indices_path)
    dataset = [sample for sample in dataset if sample is not None]
    
    # Add this line to check masking right after data loading
    mask_verification_dir = os.path.join(model_output_dir, 'mask_verification')
    verify_brain_masking(dataset, output_dir=mask_verification_dir)
    
    num_demographics = 6
    # Initialize the appropriate model

    # Enable gradients
    for param in model.parameters():
        param.requires_grad = True
    
    generate_xai_visualizations(model, dataset, model_output_dir, device, methods_to_run, atlas_path, individual_patients=individual_patients)
    gc.collect()
    torch.cuda.empty_cache()
    generate_age_binned_xai_visualizations(model, dataset, model_output_dir, device, methods_to_run, age_bin_width=10)    
    
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
    parser.add_argument('--atlas_path', type=str, default='cvasl/deepresearch/Harvard-Oxford_cortical_and_subcortical_structural_atlases/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz',
                        help="Path to the brain atlas NIfTI file")
    parser.add_argument("--indices_path", type=str, default="None", help="Path to files containing indices for test split for each dataset")
    parser.add_argument('--individual_patients', action='store_true', 
                        help="Generate individual patient visualizations in addition to aggregate plots")    
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
                args.test_csv, model_path, args.test_data_dir, args.output_dir, device, methods_to_run, args.atlas_path, args.indices_path, args.individual_patients
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