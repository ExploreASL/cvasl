import logging
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

class BrainAgeDataset(Dataset):
    def __init__(self, csv_file, image_dir, cat_cols=["Sex", "Site", "Labelling", "Readout", "LD", "PLD"], num_cols=[], target_col='Age', patient_id_col='participant_id', indices = None, transform=None, mask_path = None):
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
        self.mask_path = mask_path
        self.data_df = pd.read_csv(csv_file)
        self.original_data_df = pd.read_csv(csv_file)
        self.indices = indices
        if (self.indices is not None) and os.path.exists(self.indices):
            #if indices is a valid file path
            indices = np.load(self.indices)
            self.data_df = self.data_df.iloc[indices]
            logging.info(f"Data filtered using indices from: {self.indices}")
        logging.info(f"CSV file loaded: {csv_file} with shape: {self.data_df.shape}")
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
                if 'qCBF' in filename and original_filename_base in filename:
                    image_path = os.path.join(image_dir, filename)
                    found_match = True
                    break
            if not found_match and transformed_filename_base:
                for filename in all_files_in_dir:
                    if 'qCBF' in filename and transformed_filename_base in filename:
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

        self.data_df = self.data_df[self.data_df[self.patient_id_col].isin(valid_participant_ids)].copy()

        self.data_df = self.preprocess_data(self.data_df)

        if sample_image_shape is not None:
            self.voxel_averages = self.calculate_voxel_averages(valid_image_paths, sample_image_shape)
            
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
        cols_to_select = [self.patient_id_col, self.target_col] + self.cat_cols + self.num_cols
        available_cols = df.columns.tolist()
        final_cols_to_select = [col for col in cols_to_select if col in available_cols]
        df = df[final_cols_to_select].copy()

        for col in self.cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            else:
                logging.warning(f"Categorical column '{col}' not found in DataFrame, skipping encoding.")

        for col in self.num_cols:
            if col in df.columns:
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
                image = self.load_and_preprocess(image_path, image_path.replace('qCBF', 'BrainMaskProcessing') if self.mask_path else None)
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

    def load_and_preprocess(self, image_path, mask_file_path=None):
        """
        Loads, preprocesses (handles NaNs), optionally applies a binary mask,
        and normalizes the NIfTI image data.
        - NaN handling uses voxel averages or in-image mean (calculated pre-mask).
        - Masking zeros out data outside the mask.
        - Normalization uses mean/std of the *masked-in* non-NaN data if mask applied,
          otherwise uses global mean/std.
        Returns the preprocessed, masked, and normalized data array.
        """
        logging.debug(f"Loading image data from: {image_path}")
        try:
            img = nib.load(image_path)
            data = img.get_fdata(dtype=np.float32)
            logging.debug(f"Image data loaded with shape: {data.shape}")
        except Exception as e:
            logging.error(f"Failed to load image file {image_path}: {e}")
            return None

        data_squeezed = np.squeeze(data) # Squeeze early for consistent shape checks
        logging.debug(f"Data squeezed to shape: {data_squeezed.shape}")

        nan_mask = np.isnan(data_squeezed)
        num_nans = np.sum(nan_mask)
        inf_mask = np.isinf(data_squeezed)

        if self.mask_path:
            data_squeezed[nan_mask] = 0
            data_squeezed[inf_mask] = 0
            logging.debug(f"Replaced NaNs and Infs with 0 in data.")
            nan_mask = np.isnan(data_squeezed)
            num_nans = np.sum(nan_mask)
            inf_mask = np.isinf(data_squeezed)

        if num_nans > 0:
            logging.debug(f"Found {num_nans} NaN values before replacement.")
            if self.voxel_averages is not None:
                # check shape compatibility **AFTER** squeezing
                if self.voxel_averages.shape == data_squeezed.shape:
                    data_squeezed = np.where(nan_mask, self.voxel_averages, data_squeezed)
                    logging.debug("Replaced NaNs using voxel averages.")
                else:
                    logging.warning(f"Voxel averages shape {self.voxel_averages.shape} incompatible with squeezed data shape {data_squeezed.shape}. Falling back to in-image mean.")
                    # Calculate mean from non-NaN values of the squeezed data
                    mean_val = np.nanmean(data_squeezed) if np.any(~nan_mask) else 0
                    logging.debug(f"Replacing NaNs with in-image mean value: {mean_val}")
                    data_squeezed[nan_mask] = mean_val # Apply mean_val where original data was NaN
            else: # No voxel averages
                mean_val = np.nanmean(data_squeezed) if np.any(~nan_mask) else 0
                logging.debug(f"Replacing NaNs with in-image mean value: {mean_val}")
                data_squeezed[nan_mask] = mean_val # Apply mean_val where original data was NaN

            # Check if NaNs still exist, e.g., if mean_val was NaN because all input was NaN
            remaining_nan_mask = np.isnan(data_squeezed)
            if np.any(remaining_nan_mask):
                logging.warning(f"{np.sum(remaining_nan_mask)} NaNs remain after replacement attempts. Setting them to 0.")
                data_squeezed[remaining_nan_mask] = 0

        processed_data = data_squeezed # Start with NaN-handled data
        mask_applied = False
        final_mask = None # Keep track of the boolean mask used

        if mask_file_path and os.path.exists(mask_file_path):
            try:
                logging.debug(f"Loading mask data from: {mask_file_path}")
                mask_img = nib.load(mask_file_path)
                # Load mask data, ensure it's boolean
                mask_data = mask_img.get_fdata().astype(bool)
                logging.debug(f"Mask data loaded with shape: {mask_data.shape}")

                # ensure mask shape matches squeezed data shape,allow broadcasting for singleton dims removal
                if processed_data.shape != mask_data.shape:
                     # Try squeezing mask too
                     mask_data_squeezed = np.squeeze(mask_data)
                     if processed_data.shape == mask_data_squeezed.shape:
                         final_mask = mask_data_squeezed # Use the squeezed boolean mask
                         logging.debug(f"Squeezed mask to shape {final_mask.shape} to match data.")
                     else:
                         raise ValueError(f"Squeezed data shape {processed_data.shape} and mask shape {mask_img.shape} (squeezed: {mask_data_squeezed.shape}) are incompatible.")
                else:
                     final_mask = mask_data # Use original boolean mask

                logging.debug(f"Applying mask.")
                # apply mask: zero out voxels outside the mask
                processed_data[~final_mask] = 0
                mask_applied = True # Flag that mask was used for normalization step
                logging.debug(f"Data masked. Non-zero elements: {np.count_nonzero(processed_data)}")

            except Exception as e:
                logging.error(f"Error loading or applying mask {mask_file_path}: {e}. Proceeding without explicit mask application for this image.")
                final_mask = None # ensure mask is not used if loading failed
        elif mask_file_path:
            logging.warning(f"Mask file path provided but not found: {mask_file_path}. Proceeding without mask.")

        #Normalization (on potentially masked, NaN-handled data)
        if mask_applied and final_mask is not None:
            # Normalize based on non-zero values within the mask if mask was applied
            # Use final_mask to select voxels for statistics calculation
            if np.any(final_mask): # check if mask is not all False
                masked_in_values = processed_data[final_mask]
                mean = np.mean(masked_in_values)
                std = np.std(masked_in_values)
                logging.debug(f"Normalization stats (masked region): Mean={mean}, Std={std}")

                if std > 0:
                    # Apply normalization only to the masked-in region
                    processed_data[final_mask] = (masked_in_values - mean) / std
                elif np.any(masked_in_values): # If std is 0 but there are values, just center
                    processed_data[final_mask] = masked_in_values - mean
                # Voxels outside mask (where final_mask is False) remain 0
            else: # empty mask
                 logging.debug("Mask is empty (all False), skipping normalization.")
                 mean, std = 0, 0
        else:
            # Original normalization if no mask was applied or mask failed
            mean = np.mean(processed_data)
            std = np.std(processed_data)
            logging.debug(f"Normalization stats (full image): Mean={mean}, Std={std}")
            if std > 0:
                processed_data = (processed_data - mean) / std
            else: # Handle flat image
                processed_data = processed_data - mean

        logging.debug(
            f"Returning final preprocessed data array with shape: {processed_data.shape}")
        return processed_data.astype(np.float32)