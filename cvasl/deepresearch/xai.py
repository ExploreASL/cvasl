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
from models.cnn import Large3DCNN
from models.densenet3d import DenseNet3D
from models.efficientnet3d import EfficientNet3D
from models.improvedcnn3d import Improved3DCNN
from models.resnet3d import ResNet3D
from models.resnext3d import ResNeXt3D
from data import BrainAgeDataset

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
        # 'hirescam': HiResCAM,
        # 'gradcam++': GradCAMPlusPlus,
        # 'xgradcam': XGradCAM,
        # 'eigencam': EigenGradCAM,
        # 'layercam': LayerCAM
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

def normalize_cam(cam):
    """Normalize CAM output to range [0, 1]"""
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7)
    return cam.astype(np.float16)  # Reduce memory footprint

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

    # Axial view
    axes[0, 0].imshow(image_norm[slice_indices['axial'], :, :], cmap='gray')
    axes[1, 0].imshow(image_norm[slice_indices['axial'], :, :], cmap='gray')
    axes[1, 0].imshow(cam[slice_indices['axial'], :, :], cmap='jet', alpha=alpha)
    axes[0, 0].set_title('Axial - Original')
    axes[1, 0].set_title('Axial - Overlay')

    # Coronal view
    axes[0, 1].imshow(image_norm[:, slice_indices['coronal'], :], cmap='gray')
    axes[1, 1].imshow(image_norm[:, slice_indices['coronal'], :], cmap='gray')
    axes[1, 1].imshow(cam[:, slice_indices['coronal'], :], cmap='jet', alpha=alpha)
    axes[0, 1].set_title('Coronal - Original')
    axes[1, 1].set_title('Coronal - Overlay')

    # Sagittal view
    axes[0, 2].imshow(image_norm[:, :, slice_indices['sagittal']], cmap='gray')
    axes[1, 2].imshow(image_norm[:, :, slice_indices['sagittal']], cmap='gray')
    axes[1, 2].imshow(cam[:, :, slice_indices['sagittal']], cmap='jet', alpha=alpha)
    axes[0, 2].set_title('Sagittal - Original')
    axes[1, 2].set_title('Sagittal - Overlay')

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
    slice_indices = np.linspace(0, num_slices - 1, num=min(num_slices_per_view, num_slices), dtype=int) # Adjust num_slices if image has fewer

    rows = ceil(np.sqrt(len(slice_indices)))
    cols = ceil(len(slice_indices) / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), dpi=300) # Adjusted figsize and DPI

    image_norm = (image - image.min()) / (image.max() - image.min())
    cam_norm = normalize_cam(cam) # Use normalize_cam function

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
                img_slice = image_norm[slice_idx, :, :]
                cam_slice = cam_norm[slice_idx, :, :]

            ax.imshow(img_slice, cmap='gray')
            ax.imshow(cam_slice, cmap='jet', alpha=0.5)
            ax.set_title(f'Slice {slice_idx}', fontsize='small') # Smaller title fontsize
        ax.axis('off')

    if actual_age is not None and predicted_age is not None:
        fig.suptitle(f'{plane.capitalize()} View - Act: {actual_age:.1f}, Pred: {predicted_age:.1f}', y=0.99, fontsize=12) # Adjusted suptitle y and fontsize

    plt.tight_layout(h_pad=0.1, w_pad=0.1) # Tight layout with padding adjustments
    return fig


def generate_average_heatmaps(avg_cam, output_dir, prefix=""):
    """Generate average heatmaps for all planes using plot_all_slices"""
    # Create dummy image for plotting - not really used in plot_all_slices, but needed for function signature
    dummy_image = np.zeros(avg_cam.shape)

    # Grid plots for each plane using the new plot_all_slices function
    for plane in ['axial', 'coronal', 'sagittal']:
        fig = plot_all_slices(dummy_image, avg_cam, plane=plane) # Call plot_all_slices here
        fig.savefig(os.path.join(output_dir, f"{prefix}average_{plane}_slices_grid.png"), # Changed filename
                   bbox_inches='tight', dpi=300)
        plt.close(fig)


def plot_all_slices_side_by_side(image, cam, plane='axial', num_slices=25,
                                actual_age=None, predicted_age=None):
    """Plot 25 slices with original and heatmap side-by-side (5 rows x 10 columns) - REMOVED"""
    pass # Removed plotting of individual slices


def plot_average_summary(avg_image, avg_cam,age_range=None):
    """Create 2x3 grid showing average image and heatmap across planes"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    planes = ['sagittal', 'axial', 'coronal']
    if age_range:
        fig.suptitle(age_range, y=1.02, fontsize=14, weight='bold')
    # Normalize averages
    avg_image_norm = (avg_image - avg_image.min()) / (avg_image.max() - avg_image.min())
    avg_cam_norm = (avg_cam - avg_cam.min()) / (avg_cam.max() - avg_cam.min() + 1e-7)

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
        axes[1, col].imshow(img_slice, cmap='gray')
        axes[1, col].imshow(cam_slice, cmap='jet', alpha=0.5)
        axes[1, col].set_title(f'Average {plane.capitalize()} Heatmap')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    return fig


def generate_xai_visualizations(model, dataset, output_dir, device='cuda', methods_to_run=['all']):
    """Main visualization function with memory optimizations, only averages plotted"""
    methods = create_visualization_dirs(output_dir, methods_to_run)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize accumulators for averages
    image_accumulator = None
    cam_accumulator = None
    total_samples = 0
    age_bins = np.linspace(0, 100, 11) # 11 to get 10 bins, 0-10, 10-20, ..., 90-100
    bin_accumulators = {i: {'image': None, 'cam': None, 'count': 0} for i in range(10)} # Initialize for bins 0-9 (bin 1 to 10)


    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Processing images")):
            try:
                # Existing processing code...
                image = batch["image"].unsqueeze(1).to(device)
                demographics = batch["demographics"].to(device)
                actual_age = batch["age"].item()

                wrapped_model = BrainAgeWrapper(model, demographics)
                predicted_age = wrapped_model(image).detach().item()

                with torch.enable_grad():
                    target_layers = get_target_layers(wrapped_model)
                    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
                    grayscale_cam = cam(input_tensor=image)[0]#.cpu().numpy()

                orig_image = image.detach().cpu().squeeze().numpy()
                grayscale_cam = normalize_cam(grayscale_cam)

                # Update accumulators
                if image_accumulator is None:
                    image_accumulator = np.zeros_like(orig_image, dtype=np.float32)
                    cam_accumulator = np.zeros_like(grayscale_cam, dtype=np.float32)
                image_accumulator += orig_image.astype(np.float32)
                cam_accumulator += grayscale_cam.astype(np.float32)
                total_samples += 1

                # Update age bins
                bin_idx = np.digitize(actual_age, age_bins) - 1
                bin_idx = max(0, min(bin_idx, 9)) # Ensure bin_idx is within 0-9
                if bin_accumulators[bin_idx]['image'] is None:
                    bin_accumulators[bin_idx]['image'] = np.zeros_like(orig_image, dtype=np.float32)
                    bin_accumulators[bin_idx]['cam'] = np.zeros_like(grayscale_cam, dtype=np.float32)
                bin_accumulators[bin_idx]['image'] += orig_image.astype(np.float32)
                bin_accumulators[bin_idx]['cam'] += grayscale_cam.astype(np.float32)
                bin_accumulators[bin_idx]['count'] += 1


            except Exception as e:
                print(f"Error processing sample {batch_idx}: {str(e)}")

            # Memory cleanup
            del wrapped_model, image, demographics, grayscale_cam
            torch.cuda.empty_cache()
            gc.collect()

    # Generate final averages
    gradcam_dir = os.path.join(output_dir, 'gradcam')

    if total_samples > 0:
        avg_image = image_accumulator / total_samples
        avg_cam = cam_accumulator / total_samples

        # Dataset-wide averages - using new grid plot for planes
        generate_average_heatmaps(avg_cam, gradcam_dir, prefix="dataset_") # Modified to use generate_average_heatmaps which now does grid plots


        # Age-binned averages
        for bin_idx in range(10): # Loop through all 10 bins (0-9)
            if bin_accumulators[bin_idx]['count'] > 0:
                # Calculate age range string
                lower = age_bins[bin_idx]
                upper = age_bins[bin_idx+1]
                age_range = f"Age Range: {lower:.1f}-{upper:.1f} years"

                bin_avg_image = bin_accumulators[bin_idx]['image'] / bin_accumulators[bin_idx]['count']
                bin_avg_cam = bin_accumulators[bin_idx]['cam'] / bin_accumulators[bin_idx]['count']

                # Age-binned averages - using new grid plot for planes
                generate_average_heatmaps(bin_avg_cam, gradcam_dir, prefix=f"age_bin_{bin_idx+1}_") # Modified to use generate_average_heatmaps


def normalize_cam(cam):
    """Memory-efficient normalization"""
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7)
    return cam.astype(np.float16)


def process_single_model(csv_path, model_path, test_data_dir, base_output_dir, device):
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

    generate_xai_visualizations(model, dataset, model_output_dir, device)
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
                args.test_csv, model_path, args.test_data_dir, args.output_dir, device

            )
            logging.info(f"Successfully processed model {model_file}. Results saved in {output_dir}")
        except Exception as e:
            logging.error(f"Error processing model {model_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main()