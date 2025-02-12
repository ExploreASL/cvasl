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
import json
import torch.nn.functional as F
import platform

class BrainAgeDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        logging.info("Initializing BrainAgeDataset...")
        self.data_df = pd.read_csv(csv_file)
        logging.info(f"CSV file loaded: {csv_file}")
        self.image_dir = image_dir
        self.transform = transform
        self.id_to_filename = {}
        recognized_files_count = 0
        skipped_files_count = 0
        all_files_in_dir = set(os.listdir(image_dir))
        valid_participant_ids = []  # List to store valid participant IDs

        for participant_id in self.data_df["participant_id"].values:
            original_filename_base = f"{participant_id}"
            transformed_filename_base = None
            parts = participant_id.rsplit("_", 1)
            if len(parts) == 2:
                id_part, suffix = parts
                if (
                    len(id_part) > 2 and id_part[-1].isdigit() and id_part[-2].isdigit()
                ):
                    transformed_id_part = id_part[:-2]
                    transformed_filename_base = f"{transformed_id_part}_{suffix}"
            found_match = False
            image_path = None # Initialize image_path here

            for filename in all_files_in_dir:
                if original_filename_base in filename:
                    image_path = os.path.join(image_dir, filename) # Construct path here
                    found_match = True
                    break
            if not found_match and transformed_filename_base:
                for filename in all_files_in_dir:
                    if transformed_filename_base in filename:
                        image_path = os.path.join(image_dir, filename) # Construct path here
                        found_match = True
                        break

            if found_match and image_path: # Check if image_path is not None and found_match is True
                try:
                    # Attempt to load and preprocess to check for errors early
                    self.load_and_preprocess(image_path)
                    self.id_to_filename[participant_id] = os.path.basename(image_path) # Store just the filename
                    recognized_files_count += 1
                    valid_participant_ids.append(participant_id) # Add to valid IDs
                except Exception as e:
                    skipped_files_count += 1
                    logging.warning(
                        f"Error loading/preprocessing image for participant ID: {participant_id} at {image_path}. Skipping. Error: {e}"
                    )
            else:
                skipped_files_count += 1
                logging.warning(
                    f"No image file found for participant ID: {participant_id}"
                )

        logging.info(
            f"Number of files in image directory: {len(all_files_in_dir)}")
        logging.info(
            f"Number of recognized image files: {recognized_files_count}")
        logging.info(
            f"Number of skipped participant IDs (no matching or loadable image files): {skipped_files_count}"
        )
        logging.info(
            f"Number of participant IDs with filenames mapped: {len(self.id_to_filename)}"
        )
        logging.info(f"Found {len(self.id_to_filename)} matching image files.")

        # Filter the dataframe to keep only valid participant IDs
        self.data_df = self.data_df[self.data_df['participant_id'].isin(valid_participant_ids)].copy()

        self.data_df = self.preprocess_data(self.data_df)
        logging.info("Preprocessing of the dataframe done")

    def preprocess_data(self, df):
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
        
        categorical_cols = ["Sex", "Site", "Labelling", "Readout"]
        for col in categorical_cols:
            logging.info(f"Encoding categorical column: {col}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        
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
            try:
                image = self.load_and_preprocess(image_path)
            except Exception as e:
                logging.error(
                    f"Error loading/preprocessing image {image_path}: {e}")
                return None  # Return None if image loading fails
        else:
            logging.warning(
                f"Skipping patient ID: {patient_id} as image file was not found"
            )
            return None  # Return None if image file not found
        data_row = self.data_df.iloc[idx]
        age = data_row["Age"]
        demographics = data_row[
            ["Sex", "Site", "LD", "PLD", "Labelling", "Readout"]
        ].values.astype(
            float
        )
        sample = {
            "image": image,
            "age": torch.tensor(age, dtype=torch.float32),
            "demographics": torch.tensor(demographics, dtype=torch.float32),
            "participant_id": patient_id,  # Add participant_id here
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
        data = np.squeeze(data)
        logging.debug(f"Image data squeezed to shape: {data.shape}")
        mask = ~np.isnan(data)
        mean_val = (
            np.mean(data[mask]) if np.any(mask) else 0
        )
        logging.debug(f"Replacing NaNs with mean value: {mean_val}")
        data[~mask] = mean_val
        mean = np.mean(data)
        std = np.std(data)
        logging.debug(f"Mean: {mean}, Std: {std}")
        if std > 0:
            data = (data - mean) / std
        else:
            data = data - mean
        logging.debug(
            f"Returning preprocessed image data with shape: {data.shape}")
        return data.astype(np.float32)
