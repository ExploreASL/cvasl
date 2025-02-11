import os
import pandas as pd
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import logging
import wandb
import argparse

# Set up logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set a seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class BrainAgeDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        logging.info("Initializing BrainAgeDataset...")
        self.data_df = pd.read_csv(csv_file)
        logging.info(f"CSV file loaded: {csv_file}")
        self.image_dir = image_dir
        self.transform = transform
        # Create a mapping dictionary from participant IDs to filenames
        self.id_to_filename = {}
        recognized_files_count = 0
        skipped_files_count = 0
        all_files_in_dir = set(os.listdir(image_dir))
        for participant_id in self.data_df["participant_id"].values:
            original_filename_base = f"{participant_id}"
            transformed_filename_base = None
            parts = participant_id.rsplit("_", 1)
            if len(parts) == 2:
                id_part, suffix = parts
                if (
                    len(id_part) > 2 and id_part[-1].isdigit(
                    ) and id_part[-2].isdigit()
                ):  # check if last 2 chars are digits
                    # remove last two digits
                    transformed_id_part = id_part[:-2]
                    transformed_filename_base = f"{transformed_id_part}_{suffix}"
            found_match = False
            for filename in all_files_in_dir:
                if original_filename_base in filename:
                    self.id_to_filename[participant_id] = filename
                    recognized_files_count += 1
                    found_match = True
                    break  # Assuming one to one mapping, break after finding the first match
            if not found_match and transformed_filename_base:
                for filename in all_files_in_dir:
                    if transformed_filename_base in filename:
                        self.id_to_filename[participant_id] = filename
                        recognized_files_count += 1
                        found_match = True
                        break
            if not found_match:
                skipped_files_count += 1
                logging.warning(
                    f"No image file found for participant ID: {participant_id}"
                )
        logging.info(
            f"Number of files in image directory: {len(all_files_in_dir)}")
        logging.info(
            f"Number of recognized image files: {recognized_files_count}")
        logging.info(
            f"Number of skipped participant IDs (no matching image files): {skipped_files_count}"
        )
        logging.info(
            f"Number of participant IDs with filenames mapped: {len(self.id_to_filename)}"
        )
        logging.info(f"Found {len(self.id_to_filename)} matching image files.")
        # Preprocessing the data_df
        self.data_df = self.preprocess_data(self.data_df)
        logging.info("Preprocessing of the dataframe done")

    def preprocess_data(self, df):
        # Select required columns
        logging.info("Selecting and preprocessing relevant columns")
        df = df[
            [
                "participant_id",
                "Age",
                "Sex",
                "Site",
                "LD",
                "PLD",
                "Labelling",
                "Readout",
            ]
        ].copy()
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
        logging.debug(
            f"Getting item at index {idx} for patient ID: {patient_id}")
        if patient_id in self.id_to_filename:
            image_name = self.id_to_filename[patient_id]
            image_path = os.path.join(self.image_dir, image_name)
            logging.debug(f"Loading and preprocessing image: {image_path}")
            # Load and pre-process image
            try:
                image = self.load_and_preprocess(image_path)
            except Exception as e:
                logging.error(
                    f"Error loading/preprocessing image {image_path}: {e}")
                return None  # skip this sample
        else:
            logging.warning(
                f"Skipping patient ID: {patient_id} as image file was not found"
            )
            return None  # skip this sample
        # Get the data row
        data_row = self.data_df.iloc[idx]
        # Extract data and labels
        age = data_row["Age"]
        demographics = data_row[
            ["Sex", "Site", "LD", "PLD", "Labelling", "Readout"]
        ].values.astype(
            float
        )  # get demographic data as numpy array
        sample = {
            "image": image,
            "age": torch.tensor(age, dtype=torch.float32),
            "demographics": torch.tensor(demographics, dtype=torch.float32),
        }
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
        mean_val = (
            np.mean(data[mask]) if np.any(mask) else 0
        )  # check if mask is non-empty. if not, then the value is zero
        logging.debug(f"Replacing NaNs with mean value: {mean_val}")
        data[~mask] = mean_val
        # Intensity normalization (standard scaling)
        mean = np.mean(data)
        std = np.std(data)
        logging.debug(f"Mean: {mean}, Std: {std}")
        if std > 0:
            data = (data - mean) / std  # Avoid division by zero
        else:
            data = data - mean
        logging.debug(
            f"Returning preprocessed image data with shape: {data.shape}")
        return data.astype(np.float32)  # ensure that data type is float32

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
        self.fc1 = nn.Linear(
            16 * 30 * 36 * 30, 64
        )  # manually computed based on the final shape after max pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(
            64 + num_demographics, 1
        )  # combined the 64 features with demographic information
        self.dropout = nn.Dropout(0.2)  # Adding dropout

    def forward(self, x, demographics):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        # concatenate image and demographics
        x = torch.cat((x, demographics), dim=1)
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
        self.fc1 = nn.Linear(
            16 * 30 * 36 * 30, 64
        )  # manually computed based on the final shape after max pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(
            64 + num_demographics, 1
        )  # combined the 64 features with demographic information
        self.dropout = nn.Dropout(0.2)  # Adding dropout

    def forward(self, x, demographics):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        # concatenate image and demographics
        x = torch.cat((x, demographics), dim=1)
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

        self.fc1 = nn.Linear(
            64 * 15 * 18 * 15, 128
        )  # manually computed based on the final shape after max pooling

        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(
            128 + num_demographics, 1
        )  # combined the 128 features with demographic information
        self.dropout = nn.Dropout(0.2)  # Adding dropout

    def forward(self, x, demographics):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        # concatenate image and demographics
        x = torch.cat((x, demographics), dim=1)
        x = self.fc2(x)
        return x

class ResNet3DBlock(nn.Module):
    """
    Input shape maintained through padding=1 with stride=1
    Output shape = input shape when stride=1
    Output shape = input shape // stride when stride>1
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
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
                nn.Conv3d(current_channels, growth_rate,
                          kernel_size=3, padding=1),
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
            nn.AvgPool3d(kernel_size=2, stride=2),
        )
        # Second dense block: 64 -> 64 + (4 * growth_rate) channels
        self.dense2 = DenseBlock3D(64, growth_rate, num_layers=4)
        # Final transition
        self.trans2 = nn.Sequential(
            nn.BatchNorm3d(64 + 4 * growth_rate),
            nn.Conv3d(64 + 4 * growth_rate, 128, kernel_size=1),
            nn.AvgPool3d(kernel_size=2, stride=2),
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


def test_model_dimensions():
    """
    Utility function to test model dimensions with sample input
    """

    # Sample input dimensions for brain MRI (batch_size, channels, depth, height, width)
    x = torch.randn(1, 1, 120, 144, 120)
    demographics = torch.randn(1, 6)  # 6 demographic features
    models = {
        "ResNet3D": ResNet3D(num_demographics=6),
        "DenseNet3D": DenseNet3D(num_demographics=6),
    }
    for name, model in models.items():
        print(f"\nTesting {name} dimensions:")
        try:
            output = model(x, demographics)
            print(f"Success! Output shape: {output.shape}")
        except Exception as e:
            print(f"Error in {name}: {str(e)}")

def create_demographics_table(dataset, wandb_run):
    """Creates and logs a demographic table to wandb."""
    df = dataset.data_df.copy()
    # Group by 'Site', 'Sex' and calculate the mean and std for 'Age'
    summary_df = (
        df.groupby(["Site", "Sex"])
        .agg(
            mean_age=pd.NamedAgg(column="Age", aggfunc="mean"),
            std_age=pd.NamedAgg(column="Age", aggfunc="std"),
            count=pd.NamedAgg(column="Age", aggfunc="count"),
        )
        .reset_index()
    )
    # Create a wandb table
    demographics_table = wandb.Table(dataframe=summary_df)
    wandb_run.log({"demographics_table": demographics_table})
    logging.info("Demographics table created and logged to wandb.")

def train_model(
    csv_file,
    image_dir,
    model_type="large",
    batch_size=4,
    learning_rate=0.001,
    num_epochs=100,
    use_wandb=True,
    pretrained_model_path=None,
    use_cuda=False,
    split_strategy="stratified_group",
    test_size=0.2,
    bins=10,
    output_dir="./saved_models",
    wandb_prefix=wandb_prefix,
):

    logging.info("Starting training process...")
    os.makedirs(output_dir, exist_ok=True)
    # Create parameter string for naming
    lr_str = f"{learning_rate:.1e}".replace("+", "").replace("-", "_")
    param_str = (
        f"{wandb_prefix}_model-{model_type}_lr{lr_str}_epochs{num_epochs}_"
        f"bs{batch_size}_split-{split_strategy}_test{test_size}_bins-{bins}"
    )
    # 1. Initialize wandb
    if use_wandb:
        wandb.init(
            project="asl-brainage",
            name=param_str,
            config={
                "model_type": model_type,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "use_cuda": use_cuda,
            },
        )
        run = wandb.run
    else:
        run = None
    # 2. Create Dataset and Split Data
    dataset = BrainAgeDataset(csv_file, image_dir)
    if run:
        create_demographics_table(dataset, run)  # Create table before trainig
    dataset = [
        sample for sample in dataset if sample is not None
    ]  # remove None samples
    logging.info(
        f"Number of samples after filtering for missing data: {len(dataset)}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    logging.info(f"Using device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )  # Convert to GB
        gpu_info = (
            f"**\033[1mDetected GPU:\033[0m** {gpu_name}, Memory: {gpu_memory:.2f} GB"
        )
        logging.info(gpu_info)
    # 3. Define model, loss and optimizer
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    logging.info(f"Using split strategy: {split_strategy}")
    if split_strategy == "random":
        train_size = int((1 - test_size) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
    elif split_strategy == "stratified":
        # Create age bins for stratification
        ages = [sample["age"].item() for sample in dataset]
        age_bins = pd.qcut(ages, q=bins, labels=False)
        train_idx, test_idx = train_test_split(
            range(len(dataset)), test_size=test_size, stratify=age_bins, random_state=42
        )
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
    elif split_strategy == "group":
        # Group by site
        sites = [
            sample["demographics"][1].item() for sample in dataset
        ]  # Site is index 1
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(
            gss.split(range(len(dataset)), groups=sites))
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
    elif split_strategy == "stratified_group_sex":
        # Combine age bins and site for stratification
        ages = [sample["age"].item() for sample in dataset]
        sites = [sample["demographics"][1].item() for sample in dataset]
        sexes = [
            sample["demographics"][0].item() for sample in dataset
        ]  # Get sex for stratification
        age_bins = pd.qcut(ages, q=bins, labels=False)
        stratify_groups = [
            f"{site}_{sex}_{age_bin}"
            for site, sex, age_bin in zip(sites, sexes, age_bins)
        ]  # Include sex in stratification groups
        train_idx, test_idx = train_test_split(
            range(len(dataset)),
            test_size=test_size,
            stratify=stratify_groups,
            random_state=42,
        )
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
    elif split_strategy == "stratified_group":
        # Combine age bins and site for stratification
        ages = [sample["age"].item() for sample in dataset]
        sites = [sample["demographics"][1].item() for sample in dataset]
        age_bins = pd.qcut(ages, q=bins, labels=False)
        stratify_groups = [
            f"{site}_{age_bin}" for site, age_bin in zip(sites, age_bins)
        ]
        train_idx, test_idx = train_test_split(
            range(len(dataset)),
            test_size=test_size,
            stratify=stratify_groups,
            random_state=42,
        )
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
    else:
        raise ValueError(f"Invalid split strategy: {split_strategy}")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    logging.info("Data loaders created")
    # Set device
    # 3. Define model, loss and optimizer
    num_demographics = 6  # Number of demographic features after one-hot encoding
    if model_type == "small":
        model = Small3DCNN(num_demographics).to(device)
        model = torch.compile(model)
        logging.info(f"Using a small CNN model")
    elif model_type == "large":
        model = Large3DCNN(num_demographics).to(device)
        model = torch.compile(model)
        logging.info(f"Using a large CNN model")
    elif model_type == "resnet":
        model = ResNet3D(num_demographics).to(device)
        model = torch.compile(model)
        logging.info(f"Using a ResNet3D")
    elif model_type == "densenet":
        model = DenseNet3D(num_demographics).to(device)
        model = torch.compile(model)
        logging.info(f"Using a DenseNet model")
    else:
        raise ValueError(
            f"model_type {model_type} is not supported. Use small or large."
        )
    logging.info(f"Model created: {model}")
    if pretrained_model_path:
        model.load_state_dict(torch.load(
            pretrained_model_path, map_location=device))
        logging.info(f"Loaded pretrained model from: {pretrained_model_path}")
    criterion = nn.L1Loss()  # L1 loss
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate)  # Adam Optimizer
    logging.info(f"Loss function and optimizer set up.")
    best_model_path = os.path.join(output_dir, f"best_{param_str}.pth")
    # 4. Start training
    if not pretrained_model_path:  # Skip training if pretrained model is provided
        best_test_mae = float("inf")
        for epoch in range(num_epochs):
            logging.info(f"Starting epoch: {epoch}")
            model.train()
            train_loss = 0
            for i, batch in enumerate(train_loader):
                logging.debug(f"Processing batch: {i}")
                images = (
                    batch["image"].unsqueeze(1).to(device)
                )  # add a channel dimension
                ages = batch["age"].unsqueeze(1).to(device)
                demographics = batch["demographics"].to(device)
                optimizer.zero_grad()  # zero out old gradients
                outputs = model(images, demographics)  # forward pass
                loss = criterion(outputs, ages)  # calculate loss
                loss.backward()  # backward pass
                optimizer.step()  # update weights
                train_loss += loss.item()
                logging.debug(f"Batch {i} processed. Loss: {loss.item()}")
            # Average loss in the epoch
            train_loss = train_loss / len(train_loader)
            logging.info(f"Epoch: {epoch}, Training Loss: {train_loss}")
            # 4. Evaluate model after training
            model.eval()
            test_mae, test_rmse, test_r2, test_pearson = evaluate_model(
                model, test_loader, device
            )
            logging.info(
                f"Epoch {epoch} Test MAE: {test_mae}, RMSE: {test_rmse}, R2: {test_r2}, Pearson Correlation: {test_pearson}"
            )
            if run:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "test_mae": test_mae,
                        "test_rmse": test_rmse,
                        "test_r2": test_r2,
                        "test_pearson": test_pearson,
                    }
                )
            if test_mae < best_test_mae:
                best_test_mae = test_mae
                # save best model
                torch.save(model.state_dict(), best_model_path)
                logging.info(
                    f"Model saved at epoch {epoch} as test MAE {test_mae} is better than previous best {best_test_mae}"
                )
        logging.info("Training completed.")
    # 5. Evaluate model and create prediction chart
    model.eval()
    test_mae, test_rmse, test_r2, test_pearson = evaluate_model(
        model, test_loader, device
    )
    logging.info(
        f"Final Test MAE: {test_mae}, RMSE: {test_rmse}, R2: {test_r2}, Pearson Correlation: {test_pearson}"
    )
    if run:
        create_prediction_chart(model, test_loader, device, run)
        wandb.log(
            {
                "final_test_mae": test_mae,
                "final_test_rmse": test_rmse,
                "final_test_r2": test_r2,
                "final_test_pearson": test_pearson,
            }
        )
    if run:
        wandb.finish()
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].unsqueeze(1).to(device)
            ages = batch["age"].unsqueeze(1).to(device)
            demographics = batch["demographics"].to(device)
            outputs = model(images, demographics)  # forward pass
            all_predictions.extend(
                outputs.cpu().numpy().flatten()
            )  # flatten the output and append to list
            all_targets.extend(
                ages.cpu().numpy().flatten()
            )  # flatten the targets and append to list
    all_predictions = np.array(all_predictions)  # convert to numpy array
    all_targets = np.array(all_targets)  # convert to numpy array
    mae = mean_absolute_error(all_targets, all_predictions)  # compute mae
    rmse = np.sqrt(mean_squared_error(
        all_targets, all_predictions))  # compute rmse

    r2 = r2_score(all_targets, all_predictions)  # compute r2
    # compute pearson correlation
    pearson, _ = pearsonr(all_targets, all_predictions)
    return mae, rmse, r2, pearson

def create_prediction_chart(model, test_loader, device, wandb_run):
    """Creates and logs a prediction vs actual chart to wandb."""
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].unsqueeze(1).to(device)
            ages = batch["age"].unsqueeze(1).to(device)
            demographics = batch["demographics"].to(device)
            outputs = model(images, demographics)
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(ages.cpu().numpy().flatten())
    # Create the chart
    plt.figure(figsize=(8, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.5)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Actual vs Predicted Age")
    plt.plot(
        [min(all_targets), max(all_targets)],
        [min(all_targets), max(all_targets)],
        color="red",
    )  # Diagonal line
    plt.grid(True)
    plt.tight_layout()
    wandb_run.log({"prediction_chart": plt})
    logging.info("Prediction chart created and logged to wandb.")

def main():
    parser = argparse.ArgumentParser(
        description="Brain Age Prediction Training Script")
    parser.add_argument(
        "--csv_file",
        type=str,
        default="/home/radv/samiri/my-scratch/trainingdata/topmri.csv",
        help="Path to the CSV file",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/home/radv/samiri/my-scratch/trainingdata/topmri/",
        help="Path to the image directory",
    )
    parser.add_argument("--wandb_prefix", type=str,
                        default="", help="wandb job prefix")
    parser.add_argument(
        "--model_type",
        type=str,
        default="large",
        choices=["small", "large", "resnet", "densenet"],
        help="Type of model to use (small or large)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.000015,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=150, help="Number of epochs for training"
    )
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Path to a pretrained model file",
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        default=False,
        help="Enable CUDA (GPU) if available",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="stratified_group_sex",
        choices=[
            "random",
            "stratified",
            "group",
            "stratified_group",
            "stratified_group_sex",
        ],
        help="Data split strategy (random/stratified/group/stratified_group)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--bins", type=int, default=10, help="Number of bins for stratification"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./saved_models",
        help="Directory to save trained models",
    )
    args = parser.parse_args()
    model = train_model(
        csv_file=args.csv_file,
        image_dir=args.image_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        use_wandb=args.use_wandb,
        pretrained_model_path=args.pretrained_model_path,
        use_cuda=args.use_cuda,
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        bins=args.bins,
        output_dir=args.output_dir,
        wandb_prefix=args.wandb_prefix,
    )

if __name__ == "__main__":
    main()
