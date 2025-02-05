import os
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')
import pandas as pd
import numpy as np
import patsy
from sklearn.preprocessing import LabelEncoder
import cvasl.vendor.covbat.covbat as covbat
import cvasl.vendor.comscan.neurocombat as cvaslneurocombat
import cvasl.vendor.neurocombat.neurocombat as neurocombat
import cvasl.vendor.open_nested_combat.nest as nest
from neuroHarmonize import harmonizationLearn
import rpy2.robjects as robjects

class NeuroHarmonize:
    def __init__(
        self, features_to_harmonize, covariates, smooth_terms = [], site_indicator = 'site', empirical_bayes = True
    ):
        """
        Wrapper class for NeuroHarmonize.

        Arguments
        ---------
        features_to_harmonize : a list
            Features to harmonize excluding covariates and site indicator

        covariates : a list
            Contains covariates to control for during harmonization.
            All covariates must be encoded numerically (no categorical variables)

        smooth_terms (Optional) :  a list, default []
            Names of columns in covars to include as smooth, nonlinear terms.
            Can be any or all columns in covars, except site_indicator.
            If empty, ComBat is applied with a linear model of covariates.
            Otherwise, Generalized Additive Models (GAMs) are used.
            Using it will increase computation time due to search for optimal smoothing.

        site_indicator : a string
            Indicates the feature that differentiates different sites, default 'site'

        empirical_bayes : bool, default True
            Whether to use empirical Bayes estimates of site effects
        """
        if not isinstance(features_to_harmonize, list):
            raise TypeError("features_to_harmonize must be a list")
        if not isinstance(covariates, list):
            raise TypeError("covariates must be a list")
        if not all(isinstance(item, str) for item in covariates):
            raise ValueError("All covariates must be strings (column names)")
        if not isinstance(smooth_terms, list):
            raise TypeError("smooth_terms must be a list")
        if not isinstance(site_indicator, str):
            raise TypeError("site_indicator must be a string")
        if not isinstance(empirical_bayes, bool):
            raise TypeError("empirical_bayes must be a boolean")

        self.features_to_harmonize = features_to_harmonize
        self.covariates = covariates
        self.smooth_terms = smooth_terms
        self.empirical_bayes = empirical_bayes
        self.site_indicator = site_indicator

    def _prepare_data_for_harmonization(self, mri_datasets):
        """
        Prepares data for harmonization by combining datasets and extracting
        features and covariates.

        Arguments
        ---------
        mri_datasets : a list
            a list of MRIdataset objects

        Returns
        -------
        features_data : pd.DataFrame
            DataFrame containing features to harmonize
        covariates_data : pd.DataFrame
            DataFrame containing covariates and site indicator
        """
        all_data = pd.concat([dataset.data for dataset in mri_datasets])

        features_data = all_data[self.features_to_harmonize]
        covariates_data = all_data[self.covariates]
        covariates_data = covariates_data.rename(columns={self.site_indicator: "SITE"})
        return features_data, covariates_data

    def _reintegrate_harmonized_data(self, mri_datasets, harmonized_data, covariates_data, all_data):
        """
        Reintegrates harmonized data back into MRIdataset objects.

        Arguments
        ---------
        mri_datasets : a list
            a list of MRIdataset objects
        harmonized_data : np.ndarray
            Harmonized feature data
        covariates_data : pd.DataFrame
            Original covariates data (used for merging)
        all_data : pd.DataFrame
            Original combined data (used for columns not harmonized)

        Returns
        -------
        mri_datasets : a list of MRIdataset objects with harmonized data
        """
        harmonized_df = pd.DataFrame(
            harmonized_data, columns=self.features_to_harmonize
        )
        harmonized_df = pd.concat(
            [harmonized_df, covariates_data.reset_index(drop=True)], axis=1
        )
        harmonized_df = pd.concat(
            [harmonized_df, all_data[[i for i in all_data.columns if i not in harmonized_df.columns]].reset_index(drop=True)],
            axis=1,
        )

        for mri_dataset in mri_datasets:
            mri_dataset.data = harmonized_df[harmonized_df["SITE"] == mri_dataset.site_id]
            mri_dataset.data = mri_dataset.data.drop(columns=["SITE",'index'], errors='ignore') # added errors='ignore' in case 'index' is not always present
        return mri_datasets


    def harmonize(self, mri_datasets):
        """
        Performs the harmonization.

        Arguments
        ---------
        mri_datasets : a list
            a list of MRIdataset objects to harmonize

        Returns
        -------
        mri_datasets : a list of MRIdataset objects with harmonized data
        """
        if not isinstance(mri_datasets, list):
            raise TypeError("mri_datasets must be a list")
        for dataset in mri_datasets:
            if not hasattr(dataset, 'data') or not hasattr(dataset, 'site_id'):
                raise ValueError("Each item in mri_datasets must be an MRIdataset object with 'data' and 'site_id' attributes")

        features_data, covariates_data = self._prepare_data_for_harmonization(mri_datasets)

        # Combine all datasets (again, needed inside harmonize for harmonizationLearn call)
        all_data = pd.concat([dataset.data for dataset in mri_datasets]) # Re-obtaining all_data here - could pass from _prepare but feels cleaner to regenerate if needed

        # Perform harmonization
        _, harmonized_data = harmonizationLearn(
            np.array(features_data), covariates_data, smooth_terms=self.smooth_terms, eb=self.empirical_bayes
        )

        mri_datasets = self._reintegrate_harmonized_data(mri_datasets, harmonized_data, covariates_data, all_data)
        return mri_datasets
    

class ComscanNeuroCombat:
    def __init__(
        self,
        features_to_harmonize,
        discrete_covariates = None,
        continuous_covariates = None,
        site_indicator = 'site',
        empirical_bayes = True,
        parametric = True,
        mean_only = False
    ):
        """
        Wrapper class for Neuro Combat.

        Arguments
        ---------
        features_to_harmonize : list
            Features to harmonize excluding covariates and site indicator.
        discrete_covariates : list, optional
            Discrete covariates to control for during harmonization.
            Must be encoded numerically. Defaults to None.
        continuous_covariates : list, optional
            Continuous covariates to control for during harmonization.
            Must be encoded numerically. Defaults to None.
        site_indicator : str, default 'site'
            Feature that differentiates different sites.
        empirical_bayes : bool, default True
            Whether to use empirical Bayes estimates of site effects.
        parametric : bool, default True
            Whether to use parametric adjustment in ComBat.
        mean_only : bool, default False
            Whether to perform mean-only adjustment in ComBat.
        """
        self._validate_init_arguments(
            features_to_harmonize,
            discrete_covariates,
            continuous_covariates,
            site_indicator,
            empirical_bayes,
            parametric,
            mean_only,
        )

        self.features_to_harmonize = features_to_harmonize
        self.site_indicator = [site_indicator]  # NeuroCombat expects site to be a list
        self.discrete_covariates = discrete_covariates if discrete_covariates is not None else []
        self.continuous_covariates = continuous_covariates if continuous_covariates is not None else []
        self.empirical_bayes = empirical_bayes
        self.parametric = parametric
        self.mean_only = mean_only

    def _validate_init_arguments(
        self,
        features_to_harmonize,
        discrete_covariates,
        continuous_covariates,
        site_indicator,
        empirical_bayes,
        parametric,
        mean_only,
    ):
        """Validates arguments passed to the __init__ method."""
        if not isinstance(features_to_harmonize, list):
            raise TypeError("features_to_harmonize must be a list.")
        if not isinstance(site_indicator, str):
            raise TypeError("site_indicator must be a string.")
        if discrete_covariates is not None and not isinstance(discrete_covariates, list):
            raise TypeError("discrete_covariates must be a list or None.")
        if continuous_covariates is not None and not isinstance(continuous_covariates, list):
            raise TypeError("continuous_covariates must be a list or None.")
        if not isinstance(empirical_bayes, bool):
            raise TypeError("empirical_bayes must be a boolean.")
        if not isinstance(parametric, bool):
            raise TypeError("parametric must be a boolean.")
        if not isinstance(mean_only, bool):
            raise TypeError("mean_only must be a boolean.")

    def _prepare_data_for_harmonization(self, mri_datasets):
        """
        Prepares and concatenates data from MRIdataset objects for harmonization.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame containing features, site, and covariates.
        """
        if not mri_datasets:
            raise ValueError("mri_datasets list is empty.")
        data = pd.concat([dataset.data for dataset in mri_datasets], ignore_index=True)
        columns_to_select = (
            self.features_to_harmonize
            + self.site_indicator
            + self.discrete_covariates
            + self.continuous_covariates
        )
        try:
            data_to_harmonize = data[columns_to_select].copy()
        except KeyError as e:
            raise KeyError(f"Missing columns in input data: {e}")
        return data_to_harmonize, data

    def _apply_combat(self, data_to_harmonize):
        """
        Applies NeuroCombat harmonization to the prepared data.

        Arguments
        ---------
        data_to_harmonize : pd.DataFrame
            DataFrame containing features, site, and covariates.

        Returns
        -------
        np.ndarray
            Harmonized feature data as a NumPy array.
        """
        combat = cvaslneurocombat.Combat(
            features=self.features_to_harmonize,
            sites=self.site_indicator,
            discrete_covariates=self.discrete_covariates,
            continuous_covariates=self.continuous_covariates,
            empirical_bayes=self.empirical_bayes,
            parametric=self.parametric,
            mean_only=self.mean_only,
        )
        try:
            harmonized_data = combat.fit_transform(data_to_harmonize) # Use fit_transform for efficiency
        except Exception as e:
            raise RuntimeError(f"Error during NeuroCombat harmonization: {e}")
        return harmonized_data

    def _reintegrate_harmonized_data(self, mri_datasets, harmonized_data, original_data, data_to_harmonize):
        """
        Reintegrates harmonized data back into MRIdataset objects.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects.
        harmonized_data : np.ndarray
            Harmonized feature data.
        original_data : pd.DataFrame
            Original concatenated data.
        data_to_harmonize : pd.DataFrame
            Dataframe used for harmonization (needed for covariates).

        Returns
        -------
        list
            List of MRIdataset objects with harmonized data.
        """
        harmonized_df = pd.DataFrame(harmonized_data, columns=self.features_to_harmonize)
        covariates_cols = (
            self.site_indicator + self.discrete_covariates + self.continuous_covariates
        )
        harmonized_df = pd.concat(
            [harmonized_df, data_to_harmonize[covariates_cols].reset_index(drop=True)], axis=1
        )
        # Preserve original columns order as much as possible
        non_harmonized_cols = [
            col for col in original_data.columns if col not in harmonized_df.columns
        ]
        harmonized_df = pd.concat(
            [harmonized_df, original_data[non_harmonized_cols].reset_index(drop=True)], axis=1
        )
        original_order = list(original_data.columns) # Use original_data.columns for order
        harmonized_df = harmonized_df[original_order] # Reorder to original columns

        for dataset in mri_datasets:
            site_value = dataset.site_id
            adjusted_data = harmonized_df[
                harmonized_df[self.site_indicator[0]] == site_value
            ].copy() # Ensure no chained assignment issues and create copy
            dataset.data = adjusted_data.reset_index(drop=True) # Reset index after filtering
        return mri_datasets

    def harmonize(self, mri_datasets):
        """
        Performs harmonization on the provided MRI datasets using NeuroCombat.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects to harmonize.

        Returns
        -------
        list or None
            List of MRIdataset objects with harmonized data, or None if no features to harmonize.
        """
        if not isinstance(mri_datasets, list):
            raise TypeError("mri_datasets must be a list.")
        if not all(hasattr(dataset, 'data') and hasattr(dataset, 'site_id') for dataset in mri_datasets):
            raise ValueError("Each item in mri_datasets must be an MRIdataset object with 'data' and 'site_id' attributes.")

        if not self.features_to_harmonize:
            print("Warning: No features to harmonize specified. Returning None.") # Informative message
            return None

        data_to_harmonize, original_data = self._prepare_data_for_harmonization(mri_datasets)
        harmonized_data = self._apply_combat(data_to_harmonize)
        mri_datasets = self._reintegrate_harmonized_data(
            mri_datasets, harmonized_data, original_data, data_to_harmonize
        )
        return mri_datasets


class AutoCombat:
    def __init__(
        self,
        data_subset,
        features_to_harmonize,
        site_indicator,
        discrete_covariates=None,
        continuous_covariates=None,
        discrete_cluster_features = None,
        continuous_cluster_features = None,
        metric = 'distortion',
        features_reduction = None,
        feature_reduction_dimensions = 2,
        empirical_bayes = True
    ):
        """
        Wrapper class for Auto Combat.

        Arguments
        ---------
        data_subset : list
            Features of the dataset subset to be passed to autocombat for harmonization.
        features_to_harmonize : list
            Features to harmonize excluding covariates and site indicator.
        site_indicator : list or str
            Column name(s) indicating the site. Can be a single string or a list of strings.
        discrete_covariates : list, optional
            Discrete covariates to control for during harmonization.
            Must be encoded numerically. Defaults to None.
        continuous_covariates : list, optional
            Continuous covariates to control for during harmonization.
        discrete_cluster_features : list, optional
            Target site features which are categorical to one-hot encode for clustering.
            Defaults to None.
        continuous_cluster_features : list, optional
            Target site features which are continuous to scale for clustering.
        metric : str, default "distortion"
            Metric to define the optimal number of clusters.
            Options: "distortion", "silhouette", "calinski_harabasz".
        features_reduction : str, optional
            Method for reduction of the embedded space with n_components.
            Options: 'pca' or 'umap'. Defaults to None.
        feature_reduction_dimensions : int, default 2
            Dimension of the embedded space for features reduction.
        empirical_bayes : bool, default True
            Whether to use empirical Bayes estimates of site effects.
        """
        self._validate_init_arguments(
            data_subset,
            features_to_harmonize,
            site_indicator,
            discrete_covariates,
            continuous_covariates,
            discrete_cluster_features,
            continuous_cluster_features,
            metric,
            features_reduction,
            feature_reduction_dimensions,
            empirical_bayes,
        )

        self.data_subset = data_subset
        self.features_to_harmonize = features_to_harmonize
        # Ensure site_indicator is a list of strings for consistency in processing
        self.site_indicator = site_indicator if isinstance(site_indicator, list) else [site_indicator]
        self.discrete_covariates = discrete_covariates if discrete_covariates is not None else []
        self.continuous_covariates = continuous_covariates if continuous_covariates is not None else []
        self.discrete_cluster_features = discrete_cluster_features if discrete_cluster_features is not None else []
        self.continuous_cluster_features = continuous_cluster_features if continuous_cluster_features is not None else []
        self.metric = metric
        self.features_reduction = features_reduction
        self.feature_reduction_dimensions = feature_reduction_dimensions
        self.empirical_bayes = empirical_bayes

    def _validate_init_arguments(
        self,
        data_subset,
        features_to_harmonize,
        site_indicator,
        discrete_covariates,
        continuous_covariates,
        discrete_cluster_features,
        continuous_cluster_features,
        metric,
        features_reduction,
        feature_reduction_dimensions,
        empirical_bayes,
    ):
        """Validates arguments passed to the __init__ method."""
        if not isinstance(data_subset, list):
            raise TypeError("data_subset must be a list.")
        if not isinstance(features_to_harmonize, list):
            raise TypeError("features_to_harmonize must be a list.")
        if not isinstance(site_indicator, (str, list)):
            raise TypeError("site_indicator must be a string or a list of strings.")
        if isinstance(site_indicator, list) and not all(isinstance(item, str) for item in site_indicator):
            raise ValueError("If site_indicator is a list, all items must be strings.")
        if discrete_covariates is not None and not isinstance(discrete_covariates, list):
            raise TypeError("discrete_covariates must be a list or None.")
        if continuous_covariates is not None and not isinstance(continuous_covariates, list):
            raise TypeError("continuous_covariates must be a list or None.")
        if discrete_cluster_features is not None and not isinstance(discrete_cluster_features, list):
            raise TypeError("discrete_cluster_features must be a list or None.")
        if continuous_cluster_features is not None and not isinstance(continuous_cluster_features, list):
            raise TypeError("continuous_cluster_features must be a list or None.")
        if not isinstance(metric, str):
            raise TypeError("metric must be a string.")
        valid_metrics = ["distortion", "silhouette", "calinski_harabasz"]
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'.")
        if features_reduction is not None and not isinstance(features_reduction, str):
            raise TypeError("features_reduction must be a string or None.")
        valid_reductions = ["pca", "umap", None]
        if features_reduction not in valid_reductions:
            raise ValueError(f"features_reduction must be one of {valid_reductions}, got '{features_reduction}'.")
        if not isinstance(feature_reduction_dimensions, int):
            raise TypeError("feature_reduction_dimensions must be an integer.")
        if not isinstance(empirical_bayes, bool):
            raise TypeError("empirical_bayes must be a boolean.")

    def _prepare_data_for_harmonization(self, mri_datasets):
        """
        Prepares and concatenates data from MRIdataset objects for harmonization.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame containing the data subset.
        """
        if not mri_datasets:
            raise ValueError("mri_datasets list is empty.")
        data = pd.concat([dataset.data for dataset in mri_datasets], ignore_index=True)
        try:
            data_to_harmonize = data[self.data_subset].copy()
        except KeyError as e:
            raise KeyError(f"Missing columns in input data: {e}")
        return data_to_harmonize, data

    def _apply_autocomat(self, data_to_harmonize):
        """
        Applies AutoCombat harmonization to the prepared data.

        Arguments
        ---------
        data_to_harmonize : pd.DataFrame
            DataFrame containing the data subset.

        Returns
        -------
        np.ndarray
            Harmonized feature data as a NumPy array.
        """
        combat = cvaslneurocombat.AutoCombat(
            features = self.features_to_harmonize,
            metric = self.metric,
            sites_features=self.site_indicator,
            discrete_combat_covariates = self.discrete_covariates,
            continuous_combat_covariates = self.continuous_covariates,
            continuous_cluster_features=self.continuous_cluster_features,
            discrete_cluster_features=self.discrete_cluster_features,
            size_min=2, # Hardcoded, consider making it a parameter if needed
            features_reduction = self.features_reduction,
            n_components =self.feature_reduction_dimensions,
             empirical_bayes=self.empirical_bayes)
        try:
            harmonized_data = combat.fit_transform(data_to_harmonize)
        except Exception as e:
            raise RuntimeError(f"Error during AutoCombat harmonization: {e}")
        return harmonized_data

    def _reintegrate_harmonized_data(self, mri_datasets, harmonized_data, original_data, data_to_harmonize):
        """
        Reintegrates harmonized data back into MRIdataset objects.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects.
        harmonized_data : np.ndarray
            Harmonized feature data.
        original_data : pd.DataFrame
            Original concatenated data.
        data_to_harmonize : pd.DataFrame
            Dataframe used for harmonization (needed for covariates).

        Returns
        -------
        list
            List of MRIdataset objects with harmonized data.
        """
        harmonized_df = pd.DataFrame(harmonized_data, columns=self.features_to_harmonize)
        covariates_cols = self.site_indicator + self.discrete_covariates + self.continuous_covariates # site_indicator is already a list
        harmonized_df = pd.concat(
            [harmonized_df, data_to_harmonize[covariates_cols].reset_index(drop=True)],
            axis=1,
        )

        non_harmonized_cols = [
            col for col in original_data.columns if col not in harmonized_df.columns
        ]
        harmonized_df = pd.concat(
            [harmonized_df, original_data[non_harmonized_cols].reset_index(drop=True)], axis=1
        )
        original_order = list(original_data.columns)
        harmonized_df = harmonized_df[original_order]


        for dataset in mri_datasets:
            site_value = dataset.site_id # Assuming dataset.site_id is a single value
            # Assuming site_indicator is a list of columns, and we use the first one for filtering for now - THIS MIGHT NEED ADJUSTMENT BASED ON HOW SITE_ID and site_indicator are related.
            site_column_to_filter = self.site_indicator[0] if self.site_indicator else None # Use the first site indicator column for filtering
            if site_column_to_filter and site_column_to_filter in harmonized_df.columns:
                adjusted_data = harmonized_df[harmonized_df[site_column_to_filter] == dataset.site_id].copy()
                dataset.data = adjusted_data.reset_index(drop=True)
            else:
                dataset.data = harmonized_df.copy() # If no site_indicator or column not found, assign the entire harmonized data (check if this is the desired fallback)
        return mri_datasets


    def harmonize(self, mri_datasets):
        """
        Performs harmonization on the provided MRI datasets using AutoCombat.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects to harmonize.

        Returns
        -------
        list or None
            List of MRIdataset objects with harmonized data, or None if no features to harmonize.
        """
        if not isinstance(mri_datasets, list):
            raise TypeError("mri_datasets must be a list.")
        if not all(hasattr(dataset, 'data') and hasattr(dataset, 'site_id') for dataset in mri_datasets):
            raise ValueError("Each item in mri_datasets must be an MRIdataset object with 'data' and 'site_id' attributes.")

        if not self.features_to_harmonize:
            print("Warning: No features to harmonize specified. Returning None.")
            return None

        data_to_harmonize, original_data = self._prepare_data_for_harmonization(mri_datasets)
        harmonized_data = self._apply_autocomat(data_to_harmonize)
        mri_datasets = self._reintegrate_harmonized_data(
            mri_datasets, harmonized_data, original_data, data_to_harmonize
        )
        return mri_datasets

class Covbat:
    def __init__(
        self, features_to_harmonize,  covariates, site_indicator='site', patient_identifier = 'participant_id', numerical_covariates = ['age'], empirical_bayes = True
    ):
        """
        Wrapper class for Covbat.

        Arguments
        ---------
        features_to_harmonize : list
            Features to harmonize excluding covariates and site indicator.
        covariates : list
            Covariates to control for during harmonization.
            All covariates must be encoded numerically.
        site_indicator : str, default 'site'
            Feature that differentiates different sites (batches in original CovBat documentation).
        patient_identifier : str, default 'participant_id'
            Feature that differentiates different patients.
        numerical_covariates : list, default ['age']
            Numerical covariates for CovBat harmonization.
        empirical_bayes : bool, default True
            Whether to use empirical Bayes estimates of site effects.
        """
        self._validate_init_arguments(
            features_to_harmonize, covariates, site_indicator, patient_identifier, numerical_covariates, empirical_bayes
        )

        self.features_to_harmonize = [f.lower() for f in features_to_harmonize]
        self.covariates = [c.lower() for c in covariates]
        self.site_indicator = site_indicator.lower()
        self.patient_identifier = patient_identifier.lower()
        self.numerical_covariates = [nc.lower() for nc in numerical_covariates]
        self.empirical_bayes = empirical_bayes

    def _validate_init_arguments(
        self, features_to_harmonize, covariates, site_indicator, patient_identifier, numerical_covariates, empirical_bayes
    ):
        """Validates arguments passed to the __init__ method."""
        if not isinstance(features_to_harmonize, list):
            raise TypeError("features_to_harmonize must be a list.")
        if not isinstance(covariates, list):
            raise TypeError("covariates must be a list.")
        if not isinstance(site_indicator, str):
            raise TypeError("site_indicator must be a string.")
        if not isinstance(patient_identifier, str):
            raise TypeError("patient_identifier must be a string.")
        if not isinstance(numerical_covariates, list):
            raise TypeError("numerical_covariates must be a list.")
        if not isinstance(empirical_bayes, bool):
            raise TypeError("empirical_bayes must be a boolean.")
        if not features_to_harmonize:
            raise ValueError("features_to_harmonize cannot be empty.")
        if not covariates:
            raise ValueError("covariates cannot be empty.")
        if not site_indicator:
            raise ValueError("site_indicator cannot be empty.")
        if not patient_identifier:
            raise ValueError("patient_identifier cannot be empty.")


    def _prepare_data_for_harmonization(self, mri_datasets):
        """
        Prepares data from MRIdataset objects for CovBat harmonization.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame prepared for CovBat.
        list
            List of semi_features DataFrames for reintegration.
        """
        semi_features = []
        datasets_to_harmonize = []

        for dataset in mri_datasets:
            if self.patient_identifier not in dataset.data.columns:
                raise ValueError(f"Patient identifier '{self.patient_identifier}' not found in dataset for site '{dataset.site_id}'.")
            if self.site_indicator not in dataset.data.columns:
                raise ValueError(f"Site indicator '{self.site_indicator}' not found in dataset for site '{dataset.site_id}'.")
            missing_features = [f for f in self.features_to_harmonize if f not in dataset.data.columns]
            if missing_features:
                raise ValueError(f"Features to harmonize '{missing_features}' not found in dataset for site '{dataset.site_id}'.")
            missing_covariates = [c for c in self.covariates if c not in dataset.data.columns]
            if missing_covariates:
                raise ValueError(f"Covariates '{missing_covariates}' not found in dataset for site '{dataset.site_id}'.")


            current_semi_features = dataset.data.drop(
                [_c for _c in self.features_to_harmonize if _c not in [self.patient_identifier]], axis=1, errors='ignore' # errors='ignore' in case feature is not present post lowercasing
            )
            semi_features.append(current_semi_features)

            cols_to_drop = [
                c for c in dataset.data.columns
                if c not in self.features_to_harmonize + self.covariates + [self.site_indicator] + [self.patient_identifier]
            ]
            current_datasets_to_harmonize = dataset.data.drop(cols_to_drop, axis=1, errors='ignore') # errors='ignore' in case col is not present post lowercasing
            datasets_to_harmonize.append(current_datasets_to_harmonize)

        pheno_features = [self.patient_identifier] + self.covariates + [self.site_indicator]
        all_data = pd.concat(datasets_to_harmonize, ignore_index=True)

        return all_data, semi_features, pheno_features

    def _apply_covbat_harmonization(self, all_data, pheno_features):
        """
        Applies CovBat harmonization to the prepared data.

        Arguments
        ---------
        all_data : pd.DataFrame
            Concatenated DataFrame prepared for CovBat.
        pheno_features : list
            List of phenotypic features (patient_identifier, covariates, site_indicator).

        Returns
        -------
        pd.DataFrame
            Harmonized feature data as a DataFrame.
        """
        phenoALLFIVE = all_data[pheno_features].copy() # Explicit copy to avoid SettingWithCopyWarning
        phenoALLFIVE = phenoALLFIVE.set_index(self.patient_identifier)

        dat_ALLFIVE = all_data.set_index(self.patient_identifier).copy() # Explicit copy

        dat_ALLFIVE = dat_ALLFIVE.T

        try:
            mod_matrix = patsy.dmatrix(
                f"~ {' + '.join(self.covariates)}", phenoALLFIVE, return_type="dataframe"
            )
        except patsy.PatsyError as e:
            raise ValueError(f"Error creating model matrix with Patsy: {e}")

        try:
            harmonized_data = covbat.combat(
                data = dat_ALLFIVE,
                batch = phenoALLFIVE[self.site_indicator],
                model=mod_matrix,
                numerical_covariates=self.numerical_covariates,
                eb=self.empirical_bayes
            )
        except Exception as e:
            raise RuntimeError(f"Error during CovBat harmonization: {e}")

        harmonized_data = harmonized_data[
            len(self.covariates) :
        ]  # Remove estimated model parameters from the output
        feature_cols = [col for col in harmonized_data.index if col not in (self.site_indicator)]
        harmonized_data = harmonized_data.loc[feature_cols]

        harmonized_data = pd.concat(
            [dat_ALLFIVE.head(len(self.covariates) + 1), harmonized_data]
        )
        harmonized_data = harmonized_data.T
        harmonized_data = harmonized_data.reset_index()
        return harmonized_data

    def _reintegrate_harmonized_data(self, mri_datasets, harmonized_data, semi_features):
        """
        Reintegrates harmonized data back into MRIdataset objects.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects.
        harmonized_data : pd.DataFrame
            Harmonized feature data.
        semi_features : list
            List of semi_features DataFrames.

        Returns
        -------
        list
            List of MRIdataset objects with harmonized data.
        """
        for i, dataset in enumerate(mri_datasets):
            site_value = dataset.site_id
            adjusted_data = harmonized_data[harmonized_data[self.site_indicator] == site_value].copy() # copy to avoid set on copy
            adjusted_data = pd.merge(adjusted_data, semi_features[i], on=self.patient_identifier, how='left') # Explicit left merge to preserve harmonized data
            dataset.data = adjusted_data.reset_index(drop=True)
        return mri_datasets


    def harmonize(self, mri_datasets):
        """
        Performs harmonization using CovBat on the provided MRI datasets.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects to harmonize.

        Returns
        -------
        list or None
            List of MRIdataset objects with harmonized data, or None if no features to harmonize.
        """
        if not isinstance(mri_datasets, list):
            raise TypeError("mri_datasets must be a list.")
        if not all(hasattr(dataset, 'data') and hasattr(dataset, 'site_id') for dataset in mri_datasets):
            raise ValueError("Each item in mri_datasets must be an MRIdataset object with 'data' and 'site_id' attributes.")
        if not self.features_to_harmonize:
            print("Warning: No features to harmonize specified. Returning None.")
            return None

        all_data, semi_features, pheno_features = self._prepare_data_for_harmonization(mri_datasets)
        harmonized_data = self._apply_covbat_harmonization(all_data, pheno_features)
        mri_datasets = self._reintegrate_harmonized_data(mri_datasets, harmonized_data, semi_features)
        return mri_datasets

class NeuroCombat:
    def __init__(
        self,
        features_to_harmonize,
        discrete_covariates,
        continuous_covariates,
        patient_identifier = 'participant_id',
        site_indicator='site',
        empirical_bayes = True,
        mean_only = False,
        parametric = True,
    ):
        """
        Wrapper class for Neuro Combat.

        Arguments
        ---------
        features_to_harmonize : list
            Features to harmonize excluding covariates and site indicator.
        discrete_covariates : list
            Discrete covariates to control for during harmonization.
            All covariates must be encoded numerically.
        continuous_covariates : list
            Continuous covariates to control for during harmonization.
            All covariates must be encoded numerically.
        patient_identifier : str, default 'participant_id'
            Column name identifying each patient.
        site_indicator : str, default 'site'
            Column name indicating the site or batch.
        empirical_bayes : bool, default True
            Whether to use empirical Bayes estimates of site effects.
        mean_only : bool, default False
            Whether to perform mean-only adjustment.
        parametric : bool, default True
            Whether to use parametric adjustments.
        """
        self._validate_init_arguments(
            features_to_harmonize,
            discrete_covariates,
            continuous_covariates,
            patient_identifier,
            site_indicator,
            empirical_bayes,
            mean_only,
            parametric
        )

        self.discrete_covariates = [a.lower() for a in discrete_covariates]
        self.continuous_covariates = [a.lower() for a in continuous_covariates]
        self.features_to_harmonize = [a.lower() for a in features_to_harmonize]
        self.patient_identifier = patient_identifier.lower()
        self.site_indicator = site_indicator.lower()
        self.empirical_bayes = empirical_bayes
        self.mean_only = mean_only
        self.parametric = parametric

    def _validate_init_arguments(self, features_to_harmonize, discrete_covariates, continuous_covariates, patient_identifier, site_indicator, empirical_bayes, mean_only, parametric):
        """Validates arguments passed to the __init__ method."""
        if not isinstance(features_to_harmonize, list):
            raise TypeError("features_to_harmonize must be a list.")
        if not isinstance(discrete_covariates, list):
            raise TypeError("discrete_covariates must be a list.")
        if not isinstance(continuous_covariates, list):
            raise TypeError("continuous_covariates must be a list.")
        if not isinstance(patient_identifier, str):
            raise TypeError("patient_identifier must be a string.")
        if not isinstance(site_indicator, str):
            raise TypeError("site_indicator must be a string.")
        if not isinstance(empirical_bayes, bool):
            raise TypeError("empirical_bayes must be a boolean.")
        if not isinstance(mean_only, bool):
            raise TypeError("mean_only must be a boolean.")
        if not isinstance(parametric, bool):
            raise TypeError("parametric must be a boolean.")
        if not features_to_harmonize:
            raise ValueError("features_to_harmonize cannot be empty.")
        if not discrete_covariates:
            raise ValueError("discrete_covariates cannot be empty.") # or should it be allowed to be empty? - for now assume not. if empty list is valid input, remove this check.
        if not continuous_covariates:
            raise ValueError("continuous_covariates cannot be empty.") # same as above.


    def _prepare_data_for_neurocombat(self, mri_datasets):
        """Prepares data for NeuroCombat harmonization."""
        semi_features = []
        datasets_to_harmonize = []
        for dataset in mri_datasets:
            if self.patient_identifier not in dataset.data.columns:
                raise ValueError(f"Patient identifier '{self.patient_identifier}' not found in dataset for site '{dataset.site_id}'.")
            if self.site_indicator not in dataset.data.columns:
                raise ValueError(f"Site indicator '{self.site_indicator}' not found in dataset for site '{dataset.site_id}'.")
            missing_features = [f for f in self.features_to_harmonize if f not in dataset.data.columns]
            if missing_features:
                raise ValueError(f"Features to harmonize '{missing_features}' not found in dataset for site '{dataset.site_id}'.")
            missing_discrete_covariates = [c for c in self.discrete_covariates if c not in dataset.data.columns]
            if missing_discrete_covariates:
                raise ValueError(f"Discrete covariates '{missing_discrete_covariates}' not found in dataset for site '{dataset.site_id}'.")
            missing_continuous_covariates = [c for c in self.continuous_covariates if c not in dataset.data.columns]
            if missing_continuous_covariates:
                raise ValueError(f"Continuous covariates '{missing_continuous_covariates}' not found in dataset for site '{dataset.site_id}'.")


            semi_features.append(
                dataset.data.drop(
                    columns=[
                        f
                        for f in self.features_to_harmonize
                        if f in dataset.data.columns
                    ], errors='ignore' # added errors='ignore' in case feature is not present after lowercasing
                )
            )
            datasets_to_harmonize.append(dataset.data[(self.discrete_covariates + self.continuous_covariates + self.features_to_harmonize + [self.site_indicator] + [self.patient_identifier])])

        dataframes_transposed = [a.set_index(self.patient_identifier).T for a in datasets_to_harmonize]
        all_together_transposed = pd.concat(
            dataframes_transposed,
            axis=1,
            join="inner",
        )

        feature_cols = [col for col in all_together_transposed.index if col not in self.discrete_covariates + self.continuous_covariates]
        features_only_transposed = all_together_transposed.loc[feature_cols]

        feature_dict = dict(enumerate(features_only_transposed.T.columns)) # More concise feature_dict creation

        features_df = features_only_transposed.reset_index(drop=True).dropna()
        batch_df = all_together_transposed.reset_index(drop=True).dropna()
        lengths = [len(_d.columns) for _d in dataframes_transposed]

        return all_together_transposed, features_df, batch_df, feature_dict, lengths, semi_features


    def _create_covariates_dataframe(self, all_together_transposed, lengths):
        """Creates the covariates DataFrame for NeuroCombat."""
        batch_ids = []
        for i, l in enumerate(lengths):
            batch_ids.extend([i + 1] * l)

        covars = {self.site_indicator: batch_ids}
        for feature in self.discrete_covariates + self.continuous_covariates:
            feature_lower = feature.lower()
            if feature_lower in all_together_transposed.index:
                covars[feature] = all_together_transposed.loc[feature_lower, :].values.tolist()
        return pd.DataFrame(covars)

    def _apply_neurocombat(self, data, covars):
        """Applies NeuroCombat harmonization."""
        try:
            data_combat = neurocombat.neuroCombat(
                dat=data,
                covars=covars,
                batch_col=self.site_indicator,
                continuous_cols=self.continuous_covariates,
                categorical_cols=self.discrete_covariates,
                eb=self.empirical_bayes,
                mean_only=self.mean_only,
                parametric=self.parametric
            )["data"]
        except Exception as e:
            raise RuntimeError(f"Error during NeuroCombat harmonization: {e}")
        return data_combat


    def  _reintegrate_harmonized_data(self, mri_datasets, data_combat, bt, feature_dict, lengths, semi_features, ocols):
        """Reintegrates harmonized data back into MRIdataset objects."""
        neurocombat_df = pd.DataFrame(data_combat)
        topper = self._make_topper(bt, self.discrete_covariates + self.continuous_covariates)
        bottom = neurocombat_df.reset_index(drop=False)
        bottom = bottom.rename(columns={"index": "char"})
        bottom.columns = topper.columns
        back_together = pd.concat([topper, bottom]).T
        new_header = back_together.iloc[0]
        back_together = back_together[1:]
        back_together.columns = new_header

        harmonized_datasets = []
        start = 0
        for i, length in enumerate(lengths):
            end = start + length
            harmonized_data = back_together.iloc[start:end]
            harmonized_data = harmonized_data.rename(feature_dict, axis="columns")

            harmonized_data = harmonized_data.reset_index().rename(
                columns={"index": self.patient_identifier}
            ).drop([self.site_indicator], axis=1)

            harmonized_data = harmonized_data.merge(
                semi_features[i], on=self.patient_identifier, how='left' # Explicit left merge
            )
            harmonized_datasets.append(harmonized_data)
            start = end

        harmonized_data_concat = pd.concat([_d for _d in harmonized_datasets])
        for i, dataset in enumerate(mri_datasets):
            site_value = dataset.site_id
            adjusted_data = harmonized_data_concat[harmonized_data_concat[self.site_indicator] == site_value].copy() # copy to avoid set on copy
            adjusted_data = pd.merge(adjusted_data, semi_features[i].drop(self.discrete_covariates + self.continuous_covariates + ['index'],axis = 1, errors='ignore'), on=self.patient_identifier, how='left') # Explicit left merge, errors='ignore'
            for _c in ocols:
                if _c + '_y' in adjusted_data.columns and _c + '_x' in adjusted_data.columns:
                    adjusted_data.drop(columns=[_c+'_y'],axis=1, inplace=True)
                    adjusted_data.rename(columns={_c + '_x': _c}, inplace=True)

            dataset.data = adjusted_data.reset_index(drop=True) # reset index for clean datasets

        return mri_datasets


    def _make_topper(self, bt, row_labels): # Keep this method as it's relatively small and specific to DataFrame reconstruction
        topper = (
            bt.head(len(row_labels))
            .rename_axis(None, axis="columns")
            .reset_index()
        )
        topper['index'] = row_labels
        return topper


    def harmonize(self, mri_datasets):
        """
        Performs harmonization on the provided MRI datasets using NeuroCombat.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects to harmonize.

        Returns
        -------
        list
            List of MRIdataset objects with harmonized data.
        """
        if not isinstance(mri_datasets, list):
            raise TypeError("mri_datasets must be a list.")
        if not all(hasattr(dataset, 'data') and hasattr(dataset, 'site_id') for dataset in mri_datasets):
            raise ValueError("Each item in mri_datasets must be an MRIdataset object with 'data' and 'site_id' attributes.")
        if not self.features_to_harmonize:
            print("Warning: No features to harmonize specified. Returning input datasets unchanged.")
            return mri_datasets # Return input datasets unchanged if no features to harmonize

        ocols = mri_datasets[0].data.columns

        # Prepare data for NeuroCombat
        all_together, ft, bt, feature_dict, lengths, semi_features = self._prepare_data_for_neurocombat(mri_datasets)

        # Create covariates DataFrame
        covars = self._create_covariates_dataframe(all_together, lengths)

        # Convert data to numpy array for NeuroCombat
        data = ft.values

        # Harmonize data using NeuroCombat
        data_combat = self._apply_neurocombat(data, covars)

        # Reintegrate harmonized data
        mri_datasets = self._reintegrate_harmonized_data(mri_datasets, data_combat, bt, feature_dict, lengths, semi_features, ocols)
        return mri_datasets


class NeuroHarmonize:
    def __init__(
        self, features_to_harmonize, covariates, smooth_terms=[], site_indicator='site', empirical_bayes=True
    ):
        """
        Wrapper class for NeuroHarmonize.

        Arguments
        ---------
        features_to_harmonize : list
            Features to harmonize excluding covariates and site indicator.

        covariates : list
            Contains covariates to control for during harmonization.
            All covariates must be encoded numerically (no categorical variables).

        smooth_terms (Optional) : list, default []
            Names of columns in covars to include as smooth, nonlinear terms.
            Can be any or all columns in covars, except site_indicator.
            If empty, ComBat is applied with a linear model of covariates.
            Otherwise, Generalized Additive Models (GAMs) are used.
            Using it will increase computation time due to search for optimal smoothing.

        site_indicator : str, default 'site'
            Indicates the feature that differentiates different sites.

        empirical_bayes : bool, default True
            Whether to use empirical Bayes estimates of site effects.
        """
        if not isinstance(features_to_harmonize, list):
            raise TypeError("features_to_harmonize must be a list")
        if not isinstance(covariates, list):
            raise TypeError("covariates must be a list")
        if not all(isinstance(item, str) for item in covariates):
            raise ValueError("All covariates must be strings (column names)")
        if not isinstance(smooth_terms, list):
            raise TypeError("smooth_terms must be a list")
        if not isinstance(site_indicator, str):
            raise TypeError("site_indicator must be a string")
        if not isinstance(empirical_bayes, bool):
            raise TypeError("empirical_bayes must be a boolean")

        self.features_to_harmonize = features_to_harmonize
        self.covariates = covariates
        self.smooth_terms = smooth_terms
        self.empirical_bayes = empirical_bayes
        self.site_indicator = site_indicator

    def _prepare_data_for_harmonization(self, mri_datasets):
        """
        Prepares data for harmonization by combining datasets and extracting
        features and covariates.

        Arguments
        ---------
        mri_datasets : list
            A list of MRIdataset objects.

        Returns
        -------
        features_data : pd.DataFrame
            DataFrame containing features to harmonize.
        covariates_data : pd.DataFrame
            DataFrame containing covariates and site indicator.
        """
        all_data = pd.concat([dataset.data for dataset in mri_datasets])
        features_data = all_data[self.features_to_harmonize]
        covariates_data = all_data[self.covariates]
        covariates_data = covariates_data.rename(columns={self.site_indicator: "SITE"})
        return features_data, covariates_data

    def _reintegrate_harmonized_data(self, mri_datasets, harmonized_data, covariates_data, all_data):
        """
        Reintegrates harmonized data back into MRIdataset objects.

        Arguments
        ---------
        mri_datasets : list
            A list of MRIdataset objects.
        harmonized_data : np.ndarray
            Harmonized feature data.
        covariates_data : pd.DataFrame
            Original covariates data (used for merging).
        all_data : pd.DataFrame
            Original combined data (used for columns not harmonized).

        Returns
        -------
        list
            List of MRIdataset objects with harmonized data.
        """
        harmonized_df = pd.DataFrame(harmonized_data, columns=self.features_to_harmonize)
        harmonized_df = pd.concat(
            [harmonized_df, covariates_data.reset_index(drop=True)], axis=1
        )
        non_harmonized = [i for i in all_data.columns if i not in harmonized_df.columns]
        harmonized_df = pd.concat(
            [harmonized_df, all_data[non_harmonized].reset_index(drop=True)], axis=1,
        )

        for mri_dataset in mri_datasets:
            mri_dataset.data = harmonized_df[harmonized_df["SITE"] == mri_dataset.site_id]
            mri_dataset.data = mri_dataset.data.drop(columns=["SITE", "index"], errors='ignore')
        return mri_datasets

    def harmonize(self, mri_datasets):
        """
        Performs the harmonization.

        Arguments
        ---------
        mri_datasets : list
            A list of MRIdataset objects to harmonize.

        Returns
        -------
        list
            List of MRIdataset objects with harmonized data.
        """
        if not isinstance(mri_datasets, list):
            raise TypeError("mri_datasets must be a list")
        for dataset in mri_datasets:
            if not hasattr(dataset, 'data') or not hasattr(dataset, 'site_id'):
                raise ValueError("Each item in mri_datasets must be an MRIdataset object with 'data' and 'site_id' attributes")

        features_data, covariates_data = self._prepare_data_for_harmonization(mri_datasets)
        all_data = pd.concat([dataset.data for dataset in mri_datasets])
        _, harmonized_data = harmonizationLearn(
            np.array(features_data), covariates_data, smooth_terms=self.smooth_terms, eb=self.empirical_bayes
        )
        mri_datasets = self._reintegrate_harmonized_data(mri_datasets, harmonized_data, covariates_data, all_data)
        return mri_datasets


class ComscanNeuroCombat:
    def __init__(
        self,
        features_to_harmonize,
        discrete_covariates=None,
        continuous_covariates=None,
        site_indicator='site',
        empirical_bayes=True,
        parametric=True,
        mean_only=False
    ):
        """
        Wrapper class for Neuro Combat.

        Arguments
        ---------
        features_to_harmonize : list
            Features to harmonize excluding covariates and site indicator.
        discrete_covariates : list, optional
            Discrete covariates to control for during harmonization.
            Must be encoded numerically. Defaults to None.
        continuous_covariates : list, optional
            Continuous covariates to control for during harmonization.
            Must be encoded numerically. Defaults to None.
        site_indicator : str, default 'site'
            Feature that differentiates different sites.
        empirical_bayes : bool, default True
            Whether to use empirical Bayes estimates of site effects.
        parametric : bool, default True
            Whether to use parametric adjustment in ComBat.
        mean_only : bool, default False
            Whether to perform mean-only adjustment in ComBat.
        """
        self._validate_init_arguments(
            features_to_harmonize,
            discrete_covariates,
            continuous_covariates,
            site_indicator,
            empirical_bayes,
            parametric,
            mean_only,
        )

        self.features_to_harmonize = features_to_harmonize
        self.site_indicator = [site_indicator]  # NeuroCombat expects site to be a list
        self.discrete_covariates = discrete_covariates if discrete_covariates is not None else []
        self.continuous_covariates = continuous_covariates if continuous_covariates is not None else []
        self.empirical_bayes = empirical_bayes
        self.parametric = parametric
        self.mean_only = mean_only

    def _validate_init_arguments(
        self,
        features_to_harmonize,
        discrete_covariates,
        continuous_covariates,
        site_indicator,
        empirical_bayes,
        parametric,
        mean_only,
    ):
        """Validates arguments passed to the __init__ method."""
        if not isinstance(features_to_harmonize, list):
            raise TypeError("features_to_harmonize must be a list.")
        if not isinstance(site_indicator, str):
            raise TypeError("site_indicator must be a string.")
        if discrete_covariates is not None and not isinstance(discrete_covariates, list):
            raise TypeError("discrete_covariates must be a list or None.")
        if continuous_covariates is not None and not isinstance(continuous_covariates, list):
            raise TypeError("continuous_covariates must be a list or None.")
        if not isinstance(empirical_bayes, bool):
            raise TypeError("empirical_bayes must be a boolean.")
        if not isinstance(parametric, bool):
            raise TypeError("parametric must be a boolean.")
        if not isinstance(mean_only, bool):
            raise TypeError("mean_only must be a boolean.")

    def _prepare_data_for_harmonization(self, mri_datasets):
        """
        Prepares and concatenates data from MRIdataset objects for harmonization.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects.

        Returns
        -------
        tuple
            data_to_harmonize : pd.DataFrame
                Concatenated DataFrame containing features, site, and covariates.
            original_data : pd.DataFrame
                Original concatenated DataFrame.
        """
        if not mri_datasets:
            raise ValueError("mri_datasets list is empty.")
        data = pd.concat([dataset.data for dataset in mri_datasets], ignore_index=True)
        columns_to_select = (
            self.features_to_harmonize
            + self.site_indicator
            + self.discrete_covariates
            + self.continuous_covariates
        )
        try:
            data_to_harmonize = data[columns_to_select].copy()
        except KeyError as e:
            raise KeyError(f"Missing columns in input data: {e}")
        return data_to_harmonize, data

    def _apply_combat(self, data_to_harmonize):
        """
        Applies NeuroCombat harmonization to the prepared data.

        Arguments
        ---------
        data_to_harmonize : pd.DataFrame
            DataFrame containing features, site, and covariates.

        Returns
        -------
        np.ndarray
            Harmonized feature data as a NumPy array.
        """
        combat = cvaslneurocombat.Combat(
            features=self.features_to_harmonize,
            sites=self.site_indicator,
            discrete_covariates=self.discrete_covariates,
            continuous_covariates=self.continuous_covariates,
            empirical_bayes=self.empirical_bayes,
            parametric=self.parametric,
            mean_only=self.mean_only,
        )
        try:
            harmonized_data = combat.fit_transform(data_to_harmonize)
        except Exception as e:
            raise RuntimeError(f"Error during NeuroCombat harmonization: {e}")
        return harmonized_data

    def _reintegrate_harmonized_data(self, mri_datasets, harmonized_data, original_data, data_to_harmonize):
        """
        Reintegrates harmonized data back into MRIdataset objects.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects.
        harmonized_data : np.ndarray
            Harmonized feature data.
        original_data : pd.DataFrame
            Original concatenated data.
        data_to_harmonize : pd.DataFrame
            DataFrame used for harmonization (needed for covariates).

        Returns
        -------
        list
            List of MRIdataset objects with harmonized data.
        """
        harmonized_df = pd.DataFrame(harmonized_data, columns=self.features_to_harmonize)
        covariates_cols = self.site_indicator + self.discrete_covariates + self.continuous_covariates
        harmonized_df = pd.concat(
            [harmonized_df, data_to_harmonize[covariates_cols].reset_index(drop=True)], axis=1
        )
        non_harmonized_cols = [col for col in original_data.columns if col not in harmonized_df.columns]
        harmonized_df = pd.concat(
            [harmonized_df, original_data[non_harmonized_cols].reset_index(drop=True)], axis=1
        )
        original_order = list(original_data.columns)
        harmonized_df = harmonized_df[original_order]

        for dataset in mri_datasets:
            site_value = dataset.site_id
            adjusted_data = harmonized_df[harmonized_df[self.site_indicator[0]] == site_value].copy()
            dataset.data = adjusted_data.reset_index(drop=True)
        return mri_datasets

    def harmonize(self, mri_datasets):
        """
        Performs harmonization on the provided MRI datasets using NeuroCombat.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects to harmonize.

        Returns
        -------
        list or None
            List of MRIdataset objects with harmonized data, or None if no features to harmonize.
        """
        if not isinstance(mri_datasets, list):
            raise TypeError("mri_datasets must be a list.")
        if not all(hasattr(dataset, 'data') and hasattr(dataset, 'site_id') for dataset in mri_datasets):
            raise ValueError("Each item in mri_datasets must be an MRIdataset object with 'data' and 'site_id' attributes.")

        if not self.features_to_harmonize:
            print("Warning: No features to harmonize specified. Returning None.")
            return None

        data_to_harmonize, original_data = self._prepare_data_for_harmonization(mri_datasets)
        harmonized_data = self._apply_combat(data_to_harmonize)
        mri_datasets = self._reintegrate_harmonized_data(
            mri_datasets, harmonized_data, original_data, data_to_harmonize
        )
        return mri_datasets


class RELIEF:
    def __init__(
        self, features_to_harmonize, covariates, patient_identifier='participant_id', intermediate_results_path='.'
    ):
        """
        Wrapper class for RELIEF harmonization method.

        This class uses an R script to perform RELIEF harmonization.

        Arguments
        ---------
        features_to_harmonize : list
            List of features to harmonize, excluding covariates and patient identifier.
        covariates : list
            List of covariates to control for during harmonization.
            All covariates should be encoded numerically.
        patient_identifier : str, default 'participant_id'
            Column name identifying patients.
        intermediate_results_path : str, default '.'
            Path to save intermediate files for the R harmonization process.
        """
        self._validate_init_arguments(
            features_to_harmonize, covariates, patient_identifier, intermediate_results_path
        )

        self.features_to_harmonize = [f.lower() for f in features_to_harmonize]
        self.intermediate_results_path = intermediate_results_path
        self.covariates = [c.lower() for c in covariates]
        self.patient_identifier = patient_identifier.lower()

    def _validate_init_arguments(
        self, features_to_harmonize, covariates, patient_identifier, intermediate_results_path
    ):
        """Validates arguments passed to the __init__ method."""
        if not isinstance(features_to_harmonize, list):
            raise TypeError("features_to_harmonize must be a list.")
        if not isinstance(covariates, list):
            raise TypeError("covariates must be a list.")
        if not isinstance(patient_identifier, str):
            raise TypeError("patient_identifier must be a string.")
        if not isinstance(intermediate_results_path, str):
            raise TypeError("intermediate_results_path must be a string.")
        if not features_to_harmonize:
            raise ValueError("features_to_harmonize cannot be empty.")
        if not covariates:
            raise ValueError("covariates cannot be empty.")
        if not patient_identifier:
            raise ValueError("patient_identifier cannot be empty.")
        if not os.path.isdir(intermediate_results_path):
            raise ValueError(f"intermediate_results_path '{intermediate_results_path}' must be a valid directory.")

    def _prepare_data_for_harmonization(self, mri_datasets):
        """
        Prepares data for harmonization by concatenating data from MRIdataset objects.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects.

        Returns
        -------
        tuple
            Tuple containing:
                - all_togetherF: Concatenated DataFrame with features and covariates.
                - ftF: DataFrame with features only, transposed for RELIEF.
                - btF: DataFrame with covariates, transposed for RELIEF.
                - feature_dictF: Dictionary mapping numerical indices to feature names.
                - *lens: Lengths of individual datasets for splitting later.
        """
        dataframes = [
            _d.data[self.features_to_harmonize + [self.patient_identifier] + self.covariates].copy()
            for _d in mri_datasets
        ]
        dataframes = [a.set_index(self.patient_identifier).T for a in dataframes]

        all_togetherF = pd.concat(
            dataframes,
            axis=1,
            join="inner",
        )

        feature_cols = [col for col in all_togetherF.index if col not in self.covariates]
        features_only = all_togetherF.loc[feature_cols]

        dictionary_features_len = len(features_only.T.columns)
        number = 0
        made_keys = []
        made_vals = []
        for n in features_only.T.columns:
            made_keys.append(number)
            made_vals.append(n)
            number += 1
        feature_dictF = dict(map(lambda i, j: (i, j), made_keys, made_vals))

        ftF = features_only.reset_index()
        ftF = ftF.rename(columns={"index": "A"})
        ftF = ftF.drop(['A'], axis=1)
        ftF = ftF.dropna()

        btF = all_togetherF.reset_index()
        btF = btF.rename(columns={"index": "A"})
        btF = btF.drop(['A'], axis=1)
        btF = btF.dropna()

        lens = [len(_d.columns) for _d in dataframes]

        return all_togetherF, ftF, btF, feature_dictF, *lens

    def _make_topper(self, bt, row_labels):
        """
        Creates the header DataFrame for the harmonized data.

        Arguments
        ---------
        bt : pd.DataFrame
            DataFrame with covariates.
        row_labels : list
            List of row labels (covariate names).

        Returns
        -------
        pd.DataFrame
            Header DataFrame.
        """
        topper = (
            bt.head(len(row_labels))
            .rename_axis(None, axis="columns")
            .reset_index(drop=False)
        )
        topper = topper.rename(columns={"index": "char"})
        topper["char"] = row_labels
        return topper

    def _reintegrate_harmonized_data(self, mri_datasets, harmonized_data, feature_dictF):
        """
        Reintegrates harmonized data back into MRIdataset objects.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects.
        harmonized_data : pd.DataFrame
            Harmonized data from RELIEF.
        feature_dictF : dict
            Dictionary mapping numerical indices to feature names.

        Returns
        -------
        list
            List of MRIdataset objects with harmonized data.
        """
        new_feature_dict = {k + 1: v for k, v in feature_dictF.items()}

        for _d in mri_datasets:
            _d.data = _d.data.drop(self.features_to_harmonize, axis=1)

        cum_len = 0
        for i in range(len(mri_datasets)):
            current_len = len(mri_datasets[i].data)
            df = harmonized_data.iloc[cum_len:cum_len + current_len].rename(
                new_feature_dict, axis='columns'
            ).reset_index().rename(columns={"index": self.patient_identifier})
            
            # Merge based on patient_identifier and handle potential 'index' column
            merged_data = mri_datasets[i].data.merge(
                df.drop(self.covariates, axis=1), on=self.patient_identifier, how='left'
            )
            
            # Drop 'index' column only if it exists
            if 'index' in merged_data.columns:
                merged_data = merged_data.drop(['index'], axis=1)

            mri_datasets[i].data = merged_data
            cum_len += current_len

        return mri_datasets

    def harmonize(self, mri_datasets):
        """
        Performs RELIEF harmonization on the provided MRI datasets.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects to harmonize.

        Returns
        -------
        list
            List of MRIdataset objects with harmonized data.
        """
        if not isinstance(mri_datasets, list):
            raise TypeError("mri_datasets must be a list.")
        if not all(hasattr(dataset, 'data') and hasattr(dataset, 'site_id') for dataset in mri_datasets):
            raise ValueError(
                "Each item in mri_datasets must be an MRIdataset object with 'data' and 'site_id' attributes."
            )
        
        curr_path = os.getcwd()
        
        relief_r_driver = f"""
            rm(list = ls())
            source('{curr_path}/CVASL_RELIEF.R')            
            library(MASS)
            library(Matrix)
            options(repos = c(CRAN = "https://cran.r-project.org"))
            install.packages("denoiseR", dependencies = TRUE, quiet = TRUE)
            library(denoiseR)
            install.packages("RcppCNPy", dependencies = TRUE, quiet = TRUE)
            library(RcppCNPy)
            data5 <- npyLoad("{self.intermediate_results_path}/dat_var_for_RELIEF5.npy")
            covars5 <- read.csv('{self.intermediate_results_path}/bath_and_mod_forRELIEF5.csv')
            covars_only5  <- covars5[,-(1:2)]   
            covars_only_matrix5 <-data.matrix(covars_only5)
            relief.harmonized = relief(
                dat=data5,
                batch=covars5$batch,
                mod=covars_only_matrix5
            )
            outcomes_harmonized5 <- relief.harmonized$dat.relief
            write.csv(outcomes_harmonized5, "{self.intermediate_results_path}/relief1_for5_results.csv")
        """

        all_togetherF, ftF, btF, feature_dictF, len1, len2, len3, len4, len5 = self._prepare_data_for_harmonization(
            mri_datasets
        )

        all_togetherF.to_csv(f'{self.intermediate_results_path}/all_togeherf5.csv')
        ftF.to_csv(f'{self.intermediate_results_path}/ftF_top5.csv')
        data = np.genfromtxt(f'{self.intermediate_results_path}/ftF_top5.csv', delimiter=",", skip_header=1)
        data = data[:, 1:]
        np.save(f'{self.intermediate_results_path}/dat_var_for_RELIEF5.npy', data)

        first_columns_as_one = [1] * len1
        second_columns_as_two = [2] * len2
        third_columns_as_three = [3] * len3
        fourth_columns_as_four = [4] * len4
        fifth_columns_as_five = [5] * len5
        covars = {
            'batch': first_columns_as_one
            + second_columns_as_two
            + third_columns_as_three
            + fourth_columns_as_four
            + fifth_columns_as_five
        }
        for _c in self.covariates:
            covars[_c] = all_togetherF.loc[_c, :].values.tolist()
        covars = pd.DataFrame(covars)
        covars.to_csv(f'{self.intermediate_results_path}/bath_and_mod_forRELIEF5.csv')
        topperF = self._make_topper(btF, self.covariates)

        r = robjects.r
        r(relief_r_driver)
        bottom = pd.read_csv(
            f'{self.intermediate_results_path}/relief1_for5_results.csv', index_col=0
        ).reset_index(drop=False).rename(columns={"index": "char"})
        bottom.columns = topperF.columns
        back_together = pd.concat([topperF, bottom])
        back_together = back_together.T
        new_header = back_together.iloc[0]
        back_together.columns = new_header
        back_together = back_together[1:]

        mri_datasets = self._reintegrate_harmonized_data(mri_datasets, back_together, feature_dictF)

        return mri_datasets

 
class CombatPlusPlus:
    def __init__(
        self, features_to_harmonize, discrete_covariates, continuous_covariates, discrete_covariates_to_remove, continuous_covariates_to_remove, patient_identifier = 'participant_id', intermediate_results_path = '.', site_indicator = 'site'
    ):
        """
        Wrapper class for CombatPlusPlus harmonization method.

        Arguments
        ---------
        features_to_harmonize : list
            Features to harmonize excluding covariates and site indicator.
        discrete_covariates : list
            Discrete covariates to control for in model matrix.
            All covariates must be encoded numerically (no categorical variables).
        continuous_covariates : list
            Continuous covariates to control for in model matrix.
        discrete_covariates_to_remove : list
            Discrete covariates to remove with Combat++.
        continuous_covariates_to_remove : list
            Continuous covariates to remove with Combat++.
        patient_identifier : str, default 'participant_id'
            Column name identifying patients.
        intermediate_results_path : str, default '.'
            Path to save intermediate results of the harmonization process.
        site_indicator : str, default 'site'
            Column name indicating the site or batch.
        """
        self._validate_init_arguments(
            features_to_harmonize, discrete_covariates, continuous_covariates, discrete_covariates_to_remove, continuous_covariates_to_remove, patient_identifier, intermediate_results_path, site_indicator
        )

        self.features_to_harmonize = [f.lower() for f in features_to_harmonize]
        self.intermediate_results_path = intermediate_results_path
        self.discrete_covariates = discrete_covariates
        self.continuous_covariates = continuous_covariates
        self.patient_identifier = patient_identifier.lower()
        self.site_indicator = site_indicator.lower()
        self.discrete_covariates_to_remove = [dcr.lower() for dcr in discrete_covariates_to_remove]
        self.continuous_covariates_to_remove = [ccr.lower() for ccr in continuous_covariates_to_remove]

    def _validate_init_arguments(
        self, features_to_harmonize, discrete_covariates, continuous_covariates, discrete_covariates_to_remove, continuous_covariates_to_remove, patient_identifier, intermediate_results_path, site_indicator
    ):
        """Validates arguments passed to the __init__ method."""
        if not isinstance(features_to_harmonize, list):
            raise TypeError("features_to_harmonize must be a list.")
        if not isinstance(discrete_covariates, list):
            raise TypeError("discrete_covariates must be a list.")
        if not isinstance(continuous_covariates, list):
            raise TypeError("continuous_covariates must be a list.")
        if not isinstance(discrete_covariates_to_remove, list):
            raise TypeError("discrete_covariates_to_remove must be a list.")
        if not isinstance(continuous_covariates_to_remove, list):
            raise TypeError("continuous_covariates_to_remove must be a list.")
        if not isinstance(patient_identifier, str):
            raise TypeError("patient_identifier must be a string.")
        if not isinstance(intermediate_results_path, str):
            raise TypeError("intermediate_results_path must be a string.")
        if not isinstance(site_indicator, str):
            raise TypeError("site_indicator must be a string.")
        if not features_to_harmonize:
            raise ValueError("features_to_harmonize cannot be empty.")
        if not os.path.isdir(intermediate_results_path):
            raise ValueError(f"intermediate_results_path '{intermediate_results_path}' is not a valid directory.")

    def _generate_r_script_driver(self):
        """Generates the R script driver string for CombatPlusPlus."""
        disc_covariates_str = ','.join(repr(x) for x in self.discrete_covariates)
        cont_covariates_str = ','.join(repr(x) for x in self.continuous_covariates)
        disc_covariates_remove_str = ','.join(repr(x) for x in self.discrete_covariates_to_remove)
        cont_covariates_remove_str = ','.join(repr(x) for x in self.continuous_covariates_to_remove)

        r_script = f"""
        rm(list = ls())
        options(repos = c(CRAN = "https://cran.r-project.org"))
        if(!require(matrixStats)) {{
            install.packages("matrixStats", dependencies = TRUE, quiet = TRUE)
        }}
        library(matrixStats)
        source('{os.getcwd()}/combatPP.R') #as pluscombat
        source("{os.getcwd()}/utils.R")

        fused_dat <- read.csv('{self.intermediate_results_path}/_tmp_combined_dataset.csv')
        cont_features = c({cont_covariates_str})
        disc_features = c({disc_covariates_str})
        cont_mat <- sapply(fused_dat[cont_features], function(x) as.numeric(unlist(x)))
        disc_mat <- sapply(fused_dat[disc_features], function(x) {{
            x <- as.numeric(unlist(x))
            as.factor(x)
        }})
        mod <- model.matrix(~ ., data = data.frame(cont_mat, disc_mat))

        cont_features_to_remove = c({cont_covariates_remove_str})
        disc_features_to_remove = c({disc_covariates_remove_str})
        cont_mat_to_remove <- sapply(fused_dat[cont_features_to_remove], function(x) as.numeric(unlist(x)))
        disc_mat_to_remove <- sapply(fused_dat[disc_features_to_remove], function(x) {{
            x <- as.numeric(unlist(x))
            as.factor(x)
        }})

        if (is.null(cont_mat_to_remove) && is.null(disc_mat_to_remove)) {{
            mod_to_remove <- NULL
        }} else {{
            data_list <- list()
            if (!is.null(cont_mat_to_remove)) {{
                data_list$cont_mat_to_remove <- cont_mat_to_remove
            }}
            if (!is.null(disc_mat_to_remove)) {{
                data_list$disc_mat_to_remove <- disc_mat_to_remove
            }}
            combined_data <- do.call(cbind, data_list)
            mod_to_remove <- model.matrix(~ ., data = as.data.frame(combined_data))
        }}

        batchvector <- c(fused_dat[['{self.site_indicator}']])
        batchvector <- as.numeric(unlist(batchvector))

        ta <- t(fused_dat)
        data.harmonized <-combatPP(dat=ta, PC= mod_to_remove, mod=mod, batch=batchvector)
        new_df <- data.harmonized$dat.combat
        rollback <- t(new_df)
        write.csv(rollback, "{self.intermediate_results_path}/plus_harmonized_all.csv", row.names=TRUE)
        """
        return r_script

    def _prepare_combined_dataset(self, mri_datasets):
        """Prepares and combines datasets for CombatPlusPlus harmonization."""
        columns_to_select = (
            self.features_to_harmonize + self.discrete_covariates + self.continuous_covariates +
            [self.site_indicator] + self.continuous_covariates_to_remove + self.discrete_covariates_to_remove
        )
        try:
            all_together = pd.concat([_d.data[columns_to_select] for _d in mri_datasets], ignore_index=True)
        except KeyError as e:
            raise ValueError(f"Missing columns in input data: {e}")
        return all_together

    def _save_combined_dataset_to_csv(self, combined_dataset, filepath):
        """Saves the combined dataset to a CSV file."""
        combined_dataset.to_csv(filepath, index=False)

    def _run_combatpp_r_script(self, r_script_driver):
        """Executes the CombatPlusPlus R script."""
        try:
            r = robjects.r
            r(r_script_driver)
        except Exception as e:
            raise RuntimeError(f"Error executing CombatPlusPlus R script: {e}")

    def _load_harmonized_data_from_csv(self, filepath):
        """Loads harmonized data from the CSV file output by the R script."""
        try:
            harmonized_data = pd.read_csv(filepath, index_col=0)
            return harmonized_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Harmonized data CSV not found at: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading harmonized data from CSV: {e}")

    def _reintegrate_harmonized_data(self, mri_datasets, harmonized_data, combined_dataset):
        """Reintegrates harmonized features back into the MRIdataset objects."""
        if harmonized_data.shape[0] != combined_dataset.shape[0]:
            raise ValueError("Shape mismatch between harmonized data and combined dataset.")

        combined_dataset[self.features_to_harmonize] = harmonized_data[self.features_to_harmonize]

        for _ds in mri_datasets:
            ds_opn_harmonized = combined_dataset[combined_dataset[self.site_indicator] == _ds.site_id].copy() # copy to avoid set on copy
            _ds.data[self.features_to_harmonize] = ds_opn_harmonized[self.features_to_harmonize].copy()
        return mri_datasets

    def harmonize(self, mri_datasets):
        """
        Performs harmonization using CombatPlusPlus on the provided MRI datasets.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects to harmonize.

        Returns
        -------
        list or None
            List of MRIdataset objects with harmonized data, or None if no features to harmonize.
        """
        if not isinstance(mri_datasets, list):
            raise TypeError("mri_datasets must be a list.")
        if not all(hasattr(dataset, 'data') and hasattr(dataset, 'site_id') for dataset in mri_datasets):
            raise ValueError("Each item in mri_datasets must be an MRIdataset object with 'data' and 'site_id' attributes.")
        if not self.features_to_harmonize:
            print("Warning: No features to harmonize specified. Returning None.")
            return None

        r_script_driver = self._generate_r_script_driver()
        combined_dataset = self._prepare_combined_dataset(mri_datasets)

        os.makedirs(self.intermediate_results_path, exist_ok=True) # Ensure directory exists
        csv_filepath = f'{self.intermediate_results_path}/_tmp_combined_dataset.csv'
        self._save_combined_dataset_to_csv(combined_dataset, csv_filepath)
        self._run_combatpp_r_script(r_script_driver)
        harmonized_data = self._load_harmonized_data_from_csv(f'{self.intermediate_results_path}/plus_harmonized_all.csv')
        mri_datasets = self._reintegrate_harmonized_data(mri_datasets, harmonized_data, combined_dataset)

        return mri_datasets


class NestedComBat:
    def __init__(self, features_to_harmonize, batch_list_harmonisations, site_indicator=['site'],
                 discrete_covariates=['sex'], continuous_covariates=['age'],
                 intermediate_results_path='.', patient_identifier='participant_id',
                 return_extended=False, use_gmm=True):
        """
        Wrapper class for Nested ComBat harmonization.

        Arguments
        ---------
        features_to_harmonize : list
            Features to harmonize.
        batch_list_harmonisations : list
            List of batch variables for nested ComBat.
        site_indicator : list, default ['site']
            List containing the site indicator column name.
        discrete_covariates : list, default ['sex']
            List of discrete covariates.
        continuous_covariates : list, default ['age']
            List of continuous covariates.
        intermediate_results_path : str, default '.'
            Path to save intermediate results.
        patient_identifier : str, default 'participant_id'
            Column name for patient identifier.
        return_extended : bool, default False
            Whether to return extended outputs (intermediate dataframes).
        use_gmm : bool, default True
            Whether to use Gaussian Mixture Model (GMM) for grouping.
        """
        self._validate_init_arguments(
            features_to_harmonize, batch_list_harmonisations, site_indicator,
            discrete_covariates, continuous_covariates, intermediate_results_path,
            patient_identifier, return_extended, use_gmm
        )

        self.features_to_harmonize = [f.lower() for f in features_to_harmonize]
        self.batch_list_harmonisations = [b.lower() for b in batch_list_harmonisations]
        self.site_indicator = [s.lower() for s in site_indicator]
        self.discrete_covariates = [d.lower() for d in discrete_covariates]
        self.continuous_covariates = [c.lower() for c in continuous_covariates]
        self.intermediate_results_path = intermediate_results_path
        self.patient_identifier = patient_identifier
        self.return_extended = return_extended
        self.use_gmm = use_gmm

    def _validate_init_arguments(
        self, features_to_harmonize, batch_list_harmonisations, site_indicator,
        discrete_covariates, continuous_covariates, intermediate_results_path,
        patient_identifier, return_extended, use_gmm
    ):
        """Validates arguments passed to the __init__ method."""
        if not isinstance(features_to_harmonize, list):
            raise TypeError("features_to_harmonize must be a list.")
        if not isinstance(batch_list_harmonisations, list):
            raise TypeError("batch_list_harmonisations must be a list.")
        if not isinstance(site_indicator, list):
            raise TypeError("site_indicator must be a list.")
        if not isinstance(discrete_covariates, list):
            raise TypeError("discrete_covariates must be a list.")
        if not isinstance(continuous_covariates, list):
            raise TypeError("continuous_covariates must be a list.")
        if not isinstance(intermediate_results_path, str):
            raise TypeError("intermediate_results_path must be a string.")
        if not isinstance(patient_identifier, str):
            raise TypeError("patient_identifier must be a string.")
        if not isinstance(return_extended, bool):
            raise TypeError("return_extended must be a boolean.")
        if not isinstance(use_gmm, bool):
            raise TypeError("use_gmm must be a boolean.")
        if not features_to_harmonize:
            raise ValueError("features_to_harmonize cannot be empty.")
        if not batch_list_harmonisations:
            raise ValueError("batch_list_harmonisations cannot be empty.")
        if not os.path.isdir(intermediate_results_path):
            raise ValueError(f"intermediate_results_path '{intermediate_results_path}' is not a valid directory.")

    def _prepare_data(self, mri_datasets):
        """Prepares data for Nested ComBat harmonization."""
        batch_testing_df = pd.concat([ds.data.copy() for ds in mri_datasets], ignore_index=True)
        site_info = batch_testing_df[self.site_indicator].copy()
        dat_testing = batch_testing_df[self.features_to_harmonize].T.apply(pd.to_numeric)
        caseno_testing = batch_testing_df[self.patient_identifier]
        covars_df = batch_testing_df[self.discrete_covariates + self.continuous_covariates + self.batch_list_harmonisations]
        return batch_testing_df, site_info, dat_testing, caseno_testing, covars_df

    def _encode_categorical_covariates(self, covars_df):
        """Encodes categorical covariates using LabelEncoder."""
        covars_string = covars_df[self.discrete_covariates + self.batch_list_harmonisations].copy()
        covars_cat = pd.DataFrame(index=covars_string.index)
        for col in covars_string.columns:
            le = LabelEncoder()
            covars_cat[col] = le.fit_transform(covars_string[col].astype(str))
        covars_quant = covars_df[self.continuous_covariates].copy()
        covars_testing_final = pd.concat([covars_cat, covars_quant], axis=1)
        covars_testing_final.index = range(len(covars_testing_final))
        return covars_testing_final

    def _apply_gmm_if_needed(self, dat_testing, caseno_testing, covars_testing_final):
        """Applies Gaussian Mixture Model (GMM) grouping if use_gmm is True."""
        if self.use_gmm:
            try:
                gmm_testing_df = nest.GMMSplit(dat_testing, caseno_testing, self.intermediate_results_path) # Assuming nest.GMMSplit exists
                gmm_testing_df_merge = pd.DataFrame({
                    self.patient_identifier: caseno_testing,
                    'GMM': gmm_testing_df['Grouping']
                })
                covars_testing_final = pd.concat([
                    covars_testing_final,
                    gmm_testing_df_merge['GMM'].reset_index(drop=True)
                ], axis=1)
                discrete_covariates_final = self.discrete_covariates + ['GMM']
            except Exception as e:
                raise RuntimeError(f"Error during GMM splitting: {e}")
        else:
            discrete_covariates_final = self.discrete_covariates
        return covars_testing_final, discrete_covariates_final

    def _perform_nested_combat(self, dat_testing, covars_testing_final, discrete_covariates_final):
        """Performs Nested ComBat harmonization."""
        try:
            output_testing_df = nest.OPNestedComBat( # Assuming nest.OPNestedComBat exists
                dat_testing,
                covars_testing_final,
                self.batch_list_harmonisations,
                self.intermediate_results_path,
                categorical_cols=discrete_covariates_final,
                continuous_cols=self.continuous_covariates
            )
            return output_testing_df
        except Exception as e:
            raise RuntimeError(f"Error during Nested ComBat harmonization: {e}")

    def _save_intermediate_results(self, caseno_testing, output_testing_df, dat_testing, covars_testing_final):
        """Saves intermediate dataframes to CSV files."""
        write_testing_df = pd.concat([caseno_testing, output_testing_df], axis=1)
        write_testing_df.to_csv(f'{self.intermediate_results_path}/Mfeatures_testing_NestedComBat.csv')
        dat_testing.transpose().to_csv(f'{self.intermediate_results_path}/Mfeatures_input_testing_NestedComBat.csv')
        covars_testing_final.to_csv(f'{self.intermediate_results_path}/Mcovars_input_testing_NestedComBat.csv')
        return write_testing_df

    def _reintegrate_harmonized_data(self, mri_datasets, write_testing_df, covars_testing_final, site_info):
        """Reintegrates harmonized data back into MRIdataset objects."""
        complete_harmonised = pd.concat([write_testing_df, covars_testing_final, site_info], axis=1)
        complete_harmonised = complete_harmonised.loc[:,~complete_harmonised.columns.duplicated()].copy()

        for _ds in mri_datasets:
            ds_opn_harmonized = complete_harmonised[complete_harmonised[self.site_indicator[0]] == _ds.site_id].copy() # copy to avoid set on copy
            cols_to_drop = (['GMM'] if self.use_gmm else [])
            ds_opn_harmonized = ds_opn_harmonized.drop(columns=cols_to_drop, errors='ignore') # errors='ignore' in case GMM col is not present

            original_cols = [_c for _c in _ds.data.columns if _c not in ds_opn_harmonized.columns]
            ds_opn_harmonized = pd.merge(
                _ds.data[original_cols + [self.patient_identifier]],
                ds_opn_harmonized,
                on=self.patient_identifier,
                how='left'
            )
            _ds.data = ds_opn_harmonized.copy()
        return mri_datasets, complete_harmonised

    def harmonize(self, mri_datasets):
        """
        Performs Nested ComBat harmonization on the provided MRI datasets.

        Arguments
        ---------
        mri_datasets : list
            List of MRIdataset objects to harmonize.

        Returns
        -------
        list or tuple
            List of MRIdataset objects with harmonized data.
            Optionally returns extended outputs if return_extended is True.
        """
        if not isinstance(mri_datasets, list):
            raise TypeError("mri_datasets must be a list.")
        if not all(hasattr(dataset, 'data') and hasattr(dataset, 'site_id') for dataset in mri_datasets):
            raise ValueError("Each item in mri_datasets must be an MRIdataset object with 'data' and 'site_id' attributes.")
        if not self.features_to_harmonize:
            print("Warning: No features to harmonize specified. Returning input datasets.")
            return mri_datasets

        batch_testing_df, site_info, dat_testing, caseno_testing, covars_df = self._prepare_data(mri_datasets)
        covars_testing_final = self._encode_categorical_covariates(covars_df)
        covars_testing_final, discrete_covariates_final = self._apply_gmm_if_needed(
            dat_testing, caseno_testing, covars_testing_final)
        output_testing_df = self._perform_nested_combat(
            dat_testing, covars_testing_final, discrete_covariates_final)
        write_testing_df = self._save_intermediate_results(
            caseno_testing, output_testing_df, dat_testing, covars_testing_final)
        mri_datasets, complete_harmonised = self._reintegrate_harmonized_data(
            mri_datasets, write_testing_df, covars_testing_final, site_info)

        if self.return_extended:
            return mri_datasets, write_testing_df, dat_testing.transpose(), covars_testing_final
        return mri_datasets
    
    