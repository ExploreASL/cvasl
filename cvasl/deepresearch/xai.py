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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set a seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class BrainAgeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        logging.info("Initializing BrainAgeDataset...")
        self.data_dir = data_dir
        self.transform = transform
        self.csv_file = None
        self.image_dir = None

        # Find the csv file and image directory within the root folder
        for item in os.listdir(data_dir):
            if item.endswith(".csv"):
                self.csv_file = os.path.join(data_dir,item)
            elif os.path.isdir(os.path.join(data_dir,item)):
                self.image_dir = os.path.join(data_dir, item)

        if self.csv_file is None:
          raise FileNotFoundError("No csv file found inside the input folder")
        if self.image_dir is None:
          raise FileNotFoundError("No image folder found inside the input folder")


        self.data_df = pd.read_csv(self.csv_file)
        logging.info(f"CSV file loaded: {self.csv_file}")


        # Create a mapping dictionary from participant IDs to filenames
        self.id_to_filename = {}
        recognized_files_count = 0
        skipped_files_count = 0
        all_files_in_dir = set(os.listdir(self.image_dir))

        for participant_id in self.data_df['participant_id'].values:
            original_filename_base = f"{participant_id}"
            transformed_filename_base = None

            parts = participant_id.rsplit('_', 1)
            if len(parts) == 2:
                id_part, suffix = parts
                if len(id_part) > 2 and id_part[-1].isdigit() and id_part[-2].isdigit(): # check if last 2 chars are digits
                    transformed_id_part = id_part[:-2] # remove last two digits
                    transformed_filename_base = f"{transformed_id_part}_{suffix}"


            found_match = False
            for filename in all_files_in_dir:
                if original_filename_base in filename:
                    self.id_to_filename[participant_id] = filename
                    recognized_files_count += 1
                    found_match = True
                    break # Assuming one to one mapping, break after finding the first match

            if not found_match and transformed_filename_base:
                for filename in all_files_in_dir:
                     if transformed_filename_base in filename:
                        self.id_to_filename[participant_id] = filename
                        recognized_files_count += 1
                        found_match = True
                        break

            if not found_match:
                skipped_files_count += 1
                logging.warning(f"No image file found for participant ID: {participant_id}")


        logging.info(f"Number of files in image directory: {len(all_files_in_dir)}")
        logging.info(f"Number of recognized image files: {recognized_files_count}")
        logging.info(f"Number of skipped participant IDs (no matching image files): {skipped_files_count}")
        logging.info(f"Number of participant IDs with filenames mapped: {len(self.id_to_filename)}")


        logging.info(f"Found {len(self.id_to_filename)} matching image files.")
        # Preprocessing the data_df
        self.data_df = self.preprocess_data(self.data_df)
        logging.info("Preprocessing of the dataframe done")

    def preprocess_data(self, df):
            # Select required columns
            logging.info("Selecting and preprocessing relevant columns")
            df = df[["participant_id","Age","Sex","Site","LD","PLD","Labelling","Readout"]].copy()
            # Convert categorical features
            categorical_cols = ["Sex", "Site", "Labelling", "Readout"]
            for col in categorical_cols:
                logging.info(f"Encoding categorical column: {col}")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            # Convert numerical features to float
            numerical_cols = ["Age", "LD", "PLD"]
            for col in numerical_cols:
                logging.info(f"Converting column to float: {col}")
                df[col] = df[col].astype(float)
            return df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        patient_id = self.data_df.iloc[idx]["participant_id"]
        logging.debug(f"Getting item at index {idx} for patient ID: {patient_id}")

        if patient_id in self.id_to_filename:
          image_name = self.id_to_filename[patient_id]
          image_path = os.path.join(self.image_dir, image_name)
          logging.debug(f"Loading and preprocessing image: {image_path}")
          # Load and pre-process image
          try:
              image = self.load_and_preprocess(image_path)
          except Exception as e:
              logging.error(f"Error loading/preprocessing image {image_path}: {e}")
              return None #skip this sample
        else:
          logging.warning(f"Skipping patient ID: {patient_id} as image file was not found")
          return None #skip this sample

        # Get the data row
        data_row = self.data_df.iloc[idx]
        # Extract data and labels
        age = data_row["Age"]
        demographics = data_row[["Sex", "Site", "LD", "PLD", "Labelling", "Readout"]].values.astype(float) #get demographic data as numpy array
        sample = {"image": image, "age": torch.tensor(age, dtype=torch.float32), "demographics": torch.tensor(demographics, dtype=torch.float32)}
        logging.debug(f"Returning sample for patient: {patient_id}")
        return sample

    def load_and_preprocess(self, image_path):
        """
        Loads, preprocesses, and handles NaN values in the NIfTI image.
        """
        logging.debug(f"Loading image data from: {image_path}")
        img = nib.load(image_path)
        data = img.get_fdata()
        logging.debug(f"Image data loaded with shape: {data.shape}")
        data = np.squeeze(data)  # Remove the last dimension which is 1
        logging.debug(f"Image data squeezed to shape: {data.shape}")
        # Handle NaNs: Replace with mean of non-NaN voxels
        mask = ~np.isnan(data)
        mean_val = np.mean(data[mask]) if np.any(mask) else 0 #check if mask is non-empty. if not, then the value is zero
        logging.debug(f"Replacing NaNs with mean value: {mean_val}")
        data[~mask] = mean_val
        # Intensity normalization (standard scaling)
        mean = np.mean(data)
        std = np.std(data)
        logging.debug(f"Mean: {mean}, Std: {std}")
        if std > 0:
          data = (data - mean) / std # Avoid division by zero
        else:
          data = data - mean
        logging.debug(f"Returning preprocessed image data with shape: {data.shape}")
        return data.astype(np.float32) #ensure that data type is float32


class Small3DCNN(nn.Module):
    def __init__(self, num_demographics):
        super(Small3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 30 * 36 * 30, 64) #manually computed based on the final shape after max pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64 + num_demographics, 1)  #combined the 64 features with demographic information
        self.dropout = nn.Dropout(0.2) #Adding dropout

    def forward(self, x, demographics):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = torch.cat((x, demographics), dim=1) #concatenate image and demographics
        x = self.fc2(x)
        return x

class Small3DCNN(nn.Module):
    def __init__(self, num_demographics):
        super(Small3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 30 * 36 * 30, 64) #manually computed based on the final shape after max pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64 + num_demographics, 1)  #combined the 64 features with demographic information
        self.dropout = nn.Dropout(0.2) #Adding dropout
    def forward(self, x, demographics):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = torch.cat((x, demographics), dim=1) #concatenate image and demographics
        x = self.fc2(x)
        return x

class Large3DCNN(nn.Module):
    def __init__(self, num_demographics):
        super(Large3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 15 * 18 * 15, 128) #manually computed based on the final shape after max pooling
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128 + num_demographics, 1)  #combined the 128 features with demographic information
        self.dropout = nn.Dropout(0.2) #Adding dropout

    def forward(self, x, demographics):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = torch.cat((x, demographics), dim=1) #concatenate image and demographics
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn
import logging

class ResNet3DBlock(nn.Module):
    """
    Input shape maintained through padding=1 with stride=1
    Output shape = input shape when stride=1
    Output shape = input shape // stride when stride>1
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class DenseBlock3D(nn.Module):
    """
    Maintains spatial dimensions while increasing channel depth
    Output channels = in_channels + (growth_rate * num_layers)
    """
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels

        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm3d(current_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(current_channels, growth_rate, kernel_size=3, padding=1)
            )
            self.layers.append(layer)
            current_channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class ResNet3D(nn.Module):
    """
    For input shape (1, 120, 144, 120):
    1. Initial conv + pool: (32, 30, 36, 30)
    2. Layer1: maintains shape
    3. Layer2: (64, 15, 18, 15)
    4. Layer3: (128, 8, 9, 8)
    """
    def __init__(self, num_demographics):
        super(ResNet3D, self).__init__()

        # Initial conv: (1, H, W, D) -> (32, H/2, W/2, D/2)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        # After maxpool: (32, H/4, W/4, D/4)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Maintains spatial dimensions
        self.layer1 = self._make_layer(32, 32, 2)
        # Halves spatial dimensions
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        # Halves spatial dimensions again
        self.layer3 = self._make_layer(64, 128, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64 + num_demographics, 1)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResNet3DBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNet3DBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, demographics):
        # Input shape logging
        logging.debug(f"Input shape: {x.shape}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        logging.debug(f"After initial conv: {x.shape}")

        x = self.maxpool(x)
        logging.debug(f"After maxpool: {x.shape}")

        x = self.layer1(x)
        logging.debug(f"After layer1: {x.shape}")

        x = self.layer2(x)
        logging.debug(f"After layer2: {x.shape}")

        x = self.layer3(x)
        logging.debug(f"After layer3: {x.shape}")

        x = self.avgpool(x)
        logging.debug(f"After avgpool: {x.shape}")

        x = torch.flatten(x, 1)
        logging.debug(f"After flatten: {x.shape}")

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logging.debug(f"After FC1: {x.shape}")

        x = torch.cat((x, demographics), dim=1)
        logging.debug(f"After concatenation with demographics: {x.shape}")

        x = self.fc2(x)
        return x

class DenseNet3D(nn.Module):
    """
    For input shape (1, 120, 144, 120):
    1. Initial conv + pool: (32, 30, 36, 30)
    2. Dense1 + trans1: (64, 15, 18, 15)
    3. Dense2 + trans2: (128, 8, 9, 8)
    """
    def __init__(self, num_demographics, growth_rate=16):
        super(DenseNet3D, self).__init__()

        # Initial convolution: (1, H, W, D) -> (32, H/2, W/2, D/2)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        # After maxpool: (32, H/4, W/4, D/4)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # First dense block: 32 -> 32 + (4 * growth_rate) channels
        self.dense1 = DenseBlock3D(32, growth_rate, num_layers=4)
        # Transition reduces channels and spatial dimensions by 2
        self.trans1 = nn.Sequential(
            nn.BatchNorm3d(32 + 4 * growth_rate),
            nn.Conv3d(32 + 4 * growth_rate, 64, kernel_size=1),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )

        # Second dense block: 64 -> 64 + (4 * growth_rate) channels
        self.dense2 = DenseBlock3D(64, growth_rate, num_layers=4)
        # Final transition
        self.trans2 = nn.Sequential(
            nn.BatchNorm3d(64 + 4 * growth_rate),
            nn.Conv3d(64 + 4 * growth_rate, 128, kernel_size=1),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64 + num_demographics, 1)

    def forward(self, x, demographics):
        # Input shape logging
        logging.debug(f"Input shape: {x.shape}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        logging.debug(f"After initial conv: {x.shape}")

        x = self.maxpool(x)
        logging.debug(f"After maxpool: {x.shape}")

        x = self.dense1(x)
        logging.debug(f"After dense1: {x.shape}")

        x = self.trans1(x)
        logging.debug(f"After trans1: {x.shape}")

        x = self.dense2(x)
        logging.debug(f"After dense2: {x.shape}")

        x = self.trans2(x)
        logging.debug(f"After trans2: {x.shape}")

        x = self.avgpool(x)
        logging.debug(f"After avgpool: {x.shape}")

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logging.debug(f"After FC1: {x.shape}")

        x = torch.cat((x, demographics), dim=1)
        logging.debug(f"After concatenation with demographics: {x.shape}")

        x = self.fc2(x)
        return x


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
    """Get the target layers for visualization"""
    if isinstance(model.model, Large3DCNN):
        return [model.model.conv3]
    elif isinstance(model.model, Small3DCNN):
        return [model.model.conv2]
    elif isinstance(model.model, ResNet3D):
        return [model.model.layer3[-1].conv2]
    elif isinstance(model.model, DenseNet3D):
        return [model.model.trans2[1]]
    else:
        raise ValueError(f"Unsupported model type: {type(model.model)}")

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

                # REMOVED individual sample plotting

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

def find_csv_file(directory):
    """Find the single CSV file in the given directory"""
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {directory}")
    if len(csv_files) > 1:
        raise ValueError(f"Multiple CSV files found in {directory}. Expected only one.")
    return os.path.join(directory, csv_files[0])

def process_single_model(model_path, test_data_dir, base_output_dir, device):
    """Process a single model for XAI visualization"""
    model_filename = os.path.basename(model_path)
    # Extract model type from the filename (format: best_model-{type}_...)
    if 'model-' in model_filename:
        model_type = model_filename.split('model-')[1].split('_')[0]
    else:
        model_type = 'large'  # Default if pattern not found

    model_name = os.path.splitext(model_filename)[0]
    test_data_name = os.path.basename(os.path.normpath(test_data_dir))
    model_output_dir = os.path.join(base_output_dir, model_name, test_data_name)
    os.makedirs(model_output_dir, exist_ok=True)

    dataset = BrainAgeDataset(test_data_dir)
    dataset = [sample for sample in dataset if sample is not None]

    num_demographics = 6
    # Initialize the appropriate model
    if model_type == 'small':
        model = Small3DCNN(num_demographics).to(device)
    elif model_type == 'large':
        model = Large3DCNN(num_demographics).to(device)
    elif model_type == 'resnet':
        model = ResNet3D(num_demographics).to(device)
    elif model_type == 'densenet':
        model = DenseNet3D(num_demographics).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    # Enable gradients
    for param in model.parameters():
        param.requires_grad = True

    generate_xai_visualizations(model, dataset, model_output_dir, device)
    return model_output_dir

def main():
    parser = argparse.ArgumentParser(description="XAI Visualization for Brain Age Models")
    parser.add_argument('--models_dir', type=str, default='./trainedmodels_unmasked',
                        help="Directory containing the saved model .pth files")
    parser.add_argument('--test_data_dir', type=str,
                        default='/home/radv/samiri/my-scratch/trainingdata/unmasked/',
                        help="Directory containing the test data (CSV and image folder)")
    parser.add_argument('--output_dir', type=str, default='xai/unmasked',
                        help="Base output directory for visualizations")
    parser.add_argument('--method', type=str, default='gradcam',
                        help="Comma-separated list of XAI methods (gradcam, layercam, etc.) or 'all'")
    parser.add_argument('--device', type=str, default='cuda',
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
                model_path, args.test_data_dir, args.output_dir, device

            )
            logging.info(f"Successfully processed model {model_file}. Results saved in {output_dir}")
        except Exception as e:
            logging.error(f"Error processing model {model_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main()