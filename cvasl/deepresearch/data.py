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
    def __init__(self, csv_file, image_dir, cat_cols=["Sex", "Site", "Labelling", "Readout", "LD", "PLD"], num_cols=[], target_col='Age', patient_id_col='participant_id', transform=None):
        """
        Initializes BrainAgeDataset.

        Args:
            csv_file (string): Path to the CSV file containing annotations.
            image_dir (string): Directory with all the NIfTI images.
            cat_cols (list, optional): List of categorical column names. 
            num_cols (list, optional): List of numerical column names.
            target_col (str, optional): Name of the target column (e.g., 'Age'). Defaults to 'Age'.
            patient_id_col (str, optional): Name of the patient ID column. Defaults to 'participant_id'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        logging.info("Initializing BrainAgeDataset...")
        self.data_df = pd.read_csv(csv_file)
        logging.info(f"CSV file loaded: {csv_file}")
        self.image_dir = image_dir
        self.transform = transform
        self.id_to_filename = {}
        self.target_col = target_col
        self.patient_id_col = patient_id_col
        self.cat_cols = cat_cols 
        self.num_cols = num_cols

        recognized_files_count = 0
        skipped_files_count = 0
        all_files_in_dir = set(os.listdir(image_dir))
        valid_participant_ids = []
        valid_image_paths = []

        sample_image_shape = None

        for participant_id in self.data_df[self.patient_id_col].values:
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
            image_path = None

            for filename in all_files_in_dir:
                if original_filename_base in filename:
                    image_path = os.path.join(image_dir, filename)
                    found_match = True
                    break
            if not found_match and transformed_filename_base:
                for filename in all_files_in_dir:
                    if transformed_filename_base in filename:
                        image_path = os.path.join(image_dir, filename)
                        found_match = True
                        break

            if found_match and image_path:
                try:

                    temp_img = self.load_and_preprocess_shape_check(image_path)
                    if sample_image_shape is None:
                        sample_image_shape = temp_img.shape
                        logging.info(f"Detected image shape: {sample_image_shape}")

                    self.id_to_filename[participant_id] = os.path.basename(image_path)
                    recognized_files_count += 1
                    valid_participant_ids.append(participant_id)
                    valid_image_paths.append(image_path)
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


        self.data_df = self.data_df[self.data_df[self.patient_id_col].isin(valid_participant_ids)].copy()

        self.data_df = self.preprocess_data(self.data_df)
        logging.info("Preprocessing of the dataframe done")

        if sample_image_shape is not None:
            self.voxel_averages = self.calculate_voxel_averages(valid_image_paths, sample_image_shape)
            logging.info("Voxel-wise averages calculated.")
        else:
            self.voxel_averages = None
            logging.warning("No valid images found to calculate voxel averages. NaN replacement will use in-image mean.")


    def load_and_preprocess_shape_check(self, image_path):
        """Loads image just to get shape, lighter version"""
        img = nib.load(image_path)
        data = img.get_fdata()
        return np.squeeze(data)


    def calculate_voxel_averages(self, image_paths, sample_image_shape):
        """
        Calculates the average value for each voxel across the dataset,
        ignoring NaN values.
        """
        logging.info("Calculating voxel-wise averages across the dataset...")
        voxel_sum = np.zeros(sample_image_shape, dtype=np.float64)
        voxel_count = np.zeros(sample_image_shape, dtype=np.int32)

        for image_path in image_paths:
            try:
                img_data = nib.load(image_path).get_fdata()
                img_data = np.squeeze(img_data)
                mask = ~np.isnan(img_data)
                voxel_sum = np.where(mask, voxel_sum + img_data, voxel_sum)
                voxel_count = np.where(mask, voxel_count + 1, voxel_count)
            except Exception as e:
                logging.error(f"Error loading image {image_path} for voxel average calculation: {e}")
                continue

        voxel_average = np.zeros(sample_image_shape, dtype=np.float32)
        mask_count_gt_0 = voxel_count > 0
        voxel_average[mask_count_gt_0] = voxel_sum[mask_count_gt_0] / voxel_count[mask_count_gt_0]
        logging.info("Voxel-wise averages calculation complete.")
        return voxel_average


    def preprocess_data(self, df):
        logging.info("Selecting and preprocessing relevant columns")
        cols_to_select = [self.patient_id_col, self.target_col] + self.cat_cols + self.num_cols
        available_cols = df.columns.tolist()
        final_cols_to_select = [col for col in cols_to_select if col in available_cols]
        df = df[final_cols_to_select].copy()

        for col in self.cat_cols:
            if col in df.columns:
                logging.info(f"Encoding categorical column: {col}")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            else:
                logging.warning(f"Categorical column '{col}' not found in DataFrame, skipping encoding.")

        for col in self.num_cols:
            if col in df.columns:
                logging.info(f"Converting column to float: {col}")
                df[col] = df[col].astype(float)
            else:
                logging.warning(f"Numerical column '{col}' not found in DataFrame, skipping conversion to float.")
        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].astype(float)
        else:
            logging.error(f"Target column '{self.target_col}' not found in DataFrame.")

        return df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        patient_id = self.data_df.iloc[idx][self.patient_id_col]
        logging.debug(
            f"Getting item at index {idx} for patient ID: {patient_id}")
        if patient_id in self.id_to_filename:
            image_name = self.id_to_filename[patient_id]
            image_path = os.path.join(self.image_dir, image_name)
            logging.debug(f"Loading and preprocessing image: {image_path}")
            try:
                image = self.load_and_preprocess(image_path)
                if image is None:
                    return None
            except Exception as e:
                logging.error(
                    f"Error loading/preprocessing image {image_path}: {e}")
                return None
        else:
            logging.warning(
                f"Skipping patient ID: {patient_id} as image file was not found"
            )
            return None
        data_row = self.data_df.iloc[idx]
        age = data_row[self.target_col]
        demo_cols = self.cat_cols + self.num_cols
        demographics_cols_present = [col for col in demo_cols if col in data_row.index]
        demographics = data_row[demographics_cols_present].values.astype(float)

        sample = {
            "image": image,
            "age": torch.tensor(age, dtype=torch.float32),
            "demographics": torch.tensor(demographics, dtype=torch.float32),
            "participant_id": patient_id,
        }
        logging.debug(f"Returning sample for patient: {patient_id}")
        return sample

    def load_and_preprocess(self, image_path):
        """
        Loads, preprocesses, and handles NaN values in the NIfTI image.
        Replaces NaNs with voxel-wise average calculated over the dataset.
        If voxel-wise averages are not available, falls back to in-image mean.
        Prints percentage of NaN/Inf values after replacement.
        """
        logging.debug(f"Loading image data from: {image_path}")
        img = nib.load(image_path)
        data = img.get_fdata()
        logging.debug(f"Image data loaded with shape: {data.shape}")
        data = np.squeeze(data)
        logging.debug(f"Image data squeezed to shape: {data.shape}")
        mask = np.isnan(data)

        if self.voxel_averages is not None:
            data = np.where(mask, self.voxel_averages, data)
        else:
            mean_val = np.nanmean(data) if np.any(mask) else 0
            logging.debug(f"Replacing NaNs with in-image mean value: {mean_val}")
            data[mask] = mean_val

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