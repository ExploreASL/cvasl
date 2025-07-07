import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
import warnings
from sklearn.preprocessing import StandardScaler

class PredictBrainAge:

    def __init__(self,
        model_name,
        model_file_name,
        model,
        datasets,
        datasets_validation,
        features,
        target,
        patient_identifier = 'participant_id',
        cat_category='sex',
        cont_category='age',
        site_indicator='site',
        n_bins=4,
        splits=5,
        test_size_p=0.2,
        random_state=42,
        ):
        """
        Initializes the PredictBrainAge class.

        Args:
            model_name (str): Name of the model.
            model_file_name (str): File name for saving the model.
            model (object): The machine learning model object (must have fit and predict methods).
            datasets (list): List of MRIdataset objects for training and testing.
            datasets_validation (list, optional): List of MRIdataset objects for validation. Defaults to None.
            features (list): List of feature names to use for prediction.
            target (str): Name of the target variable (e.g., brain age).
            patient_identifier (str, optional): Column name for patient ID. Defaults to 'participant_id'.
            cat_category (str, optional): Column name for categorical category. Defaults to 'sex'.
            cont_category (str, optional): Column name for continuous category. Defaults to 'age'.
            site_indicator (str, optional): Column name for site indicator. Defaults to 'site'.
            n_bins (int, optional): Number of bins for continuous category. Defaults to 4.
            splits (int, optional): Number of splits for StratifiedShuffleSplit. Defaults to 5.
            test_size_p (float, optional): Test size percentage for StratifiedShuffleSplit. Defaults to 0.2.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        # Input Validation (as implemented in the previous step)
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        if not isinstance(model_file_name, str):
            raise TypeError("model_file_name must be a string.")
        if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
            raise TypeError("model must have 'fit' and 'predict' methods.")
        if not isinstance(datasets, list) or not all(hasattr(ds, 'data') for ds in datasets):
            raise TypeError("datasets must be a list of MRIdataset objects with 'data' attribute.")
        if datasets_validation is not None and not isinstance(datasets_validation, list):
            raise TypeError("datasets_validation must be a list or None.")
        if not isinstance(features, list):
            raise TypeError("features must be a list.")
        if not isinstance(target, str):
            raise TypeError("target must be a string.")
        if not isinstance(patient_identifier, str):
            raise TypeError("patient_identifier must be a string.")
        if not isinstance(cat_category, str):
            raise TypeError("cat_category must be a string.")
        if not isinstance(cont_category, str):
            raise TypeError("cont_category must be a string.")
        if not isinstance(site_indicator, str):
            raise TypeError("site_indicator must be a string.")
        if not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError("n_bins must be a positive integer.")
        if not isinstance(splits, int) or splits <= 0:
            raise ValueError("splits must be a positive integer.")
        if not isinstance(test_size_p, float) or not 0 < test_size_p < 1:
            raise ValueError("test_size_p must be a float between 0 and 1.")
        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer.")
        if not datasets:
            raise ValueError("datasets cannot be empty.")
        if not features:
            raise ValueError("features cannot be empty.")
        if not target:
            raise ValueError("target cannot be empty.")


        self.model_name = model_name
        self.model_file_name = model_file_name
        self.model = model
        self.patient_identifier = patient_identifier
        self.datasets = datasets
        self.datasets_validation = datasets_validation
        self.data = pd.concat([_d.data for _d in datasets])
        self.data_validation = pd.concat([_d.data for _d in datasets_validation]) if datasets_validation is not None else None
        self.features = features
        self.target = target
        self.site_indicator = site_indicator
        self.cat_category = cat_category
        self.cont_category = cont_category
        self.splits = splits
        self.test_size_p = test_size_p
        self.random_state = random_state
        self.n_bins = n_bins


    def bin_dataset(self, ds, column, num_bins=4):
        """Bins a specified column in a pandas DataFrame using quantile cut."""
        ds[f'binned'] = pd.qcut(ds[column], num_bins, labels=False, duplicates='drop')

    def _prepare_stratified_data(self):
        """Prepares data by binning continuous category and creating a combined bin for stratification."""
        self.bin_dataset(self.data, self.cont_category, num_bins=self.n_bins)
        self.data['fuse_bin'] = pd.factorize(
            self.data[self.cat_category].astype(str) + '_' + self.data['binned'].astype(str)
        )[0]
        if self.datasets_validation is not None:
            self.bin_dataset(self.data_validation, self.cont_category, num_bins=self.n_bins)
            self.data_validation['fuse_bin'] = pd.factorize(
                self.data_validation[self.cat_category].astype(str) + '_' + self.data_validation['binned'].astype(str)
            )[0]

    def _initialize_prediction_lists(self):
        """Initializes lists to store metrics, predictions, and models."""
        return [], [], [], [], []

    def _scale_features(self, X, X_val):
        """Scales features using StandardScaler."""
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        X_val_scaled = sc.transform(X_val) if X_val is not None else None
        return X_scaled, X_val_scaled, sc

    def _split_train_test(self, X, y, fuse_bin, train_index, test_index):
        """Splits data into training and testing sets based on indices."""
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        return X_train, y_train, X_test, y_test

    def _train_predict_and_evaluate(self, i, X_train, y_train, X_test, y_test, X_val, y_val, test_index):
        """Trains model, makes predictions, and evaluates metrics for a single fold."""
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_pred_val = self.model.predict(X_val) if X_val is not None else None
        metrics_data = self._calculate_fold_metrics(i, y_test, y_pred)
        metric_data_val = self._calculate_fold_metrics(i, y_val, y_pred_val) if X_val is not None and y_val is not None else None
        predictions_data, predictions_data_val = self._store_fold_predictions(i, y_test, y_pred, y_val, y_pred_val, test_index)
        return metrics_data, metric_data_val, predictions_data, predictions_data_val
    
    def _calculate_fold_metrics(self, fold_index, y_true, y_pred):
        """Calculates various regression metrics for a given fold."""
        metrics_data = {
            'algorithm': f'{self.model_name}-{fold_index}',
            'fold': fold_index,
            'file_name': f'{self.model_file_name}.{fold_index}',
            'explained_variance': metrics.explained_variance_score(y_true, y_pred),
            'max_error': metrics.max_error(y_true, y_pred),
            'mean_absolute_error': metrics.mean_absolute_error(y_true, y_pred),
            'mean_squared_error': metrics.mean_squared_error(y_true, y_pred),
            'mean_squared_log_error': metrics.mean_squared_log_error(y_true, y_pred) if all(y_true > 0) and all(y_pred > 0) else None,
            'median_absolute_error': metrics.median_absolute_error(y_true, y_pred),
            'r2': metrics.r2_score(y_true, y_pred),
            'mean_poisson_deviance': metrics.mean_poisson_deviance(y_true, y_pred) if all(y_true >= 0) and all(y_pred >= 0) else None,
            'mean_gamma_deviance': metrics.mean_gamma_deviance(y_true, y_pred) if all(y_true > 0) and all(y_pred > 0) else None,
            'mean_tweedie_deviance': metrics.mean_tweedie_deviance(y_true, y_pred),
            'd2_tweedie_score': metrics.d2_tweedie_score(y_true, y_pred), # Added d2 tweedie score
            'mean_absolute_percentage_error': metrics.mean_absolute_percentage_error(y_true, y_pred), # Added MAPE
        }
        return metrics_data

    def _store_fold_predictions(self, fold_index, y_test, y_pred, y_val, y_pred_val, test_index):
        """Stores predictions for the current fold in DataFrame format."""
        # Ensure arrays are properly flattened only if needed
        y_test_flat = y_test.flatten() if hasattr(y_test, 'flatten') else y_test
        y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
        
        predictions_data = pd.DataFrame({'y_test': y_test_flat, 'y_pred': y_pred_flat})
        predictions_data[self.patient_identifier] = self.data[self.patient_identifier].values[test_index]
        predictions_data['site'] = self.data[self.site_indicator].values[test_index]
        predictions_data['fold'] = fold_index  # Add fold index to track predictions by fold
        
        predictions_data_val = None
        if y_val is not None and y_pred_val is not None and self.data_validation is not None:
            y_val_flat = y_val.flatten() if hasattr(y_val, 'flatten') else y_val
            y_pred_val_flat = y_pred_val.flatten() if hasattr(y_pred_val, 'flatten') else y_pred_val
            
            predictions_data_val = pd.DataFrame({'y_test': y_val_flat, 'y_pred': y_pred_val_flat})
            predictions_data_val[self.patient_identifier] = self.data_validation[self.patient_identifier].values
            predictions_data_val['site'] = self.data_validation[self.site_indicator].values
            predictions_data_val['fold'] = fold_index  # Add fold index to validation predictions too
        
        return predictions_data, predictions_data_val

    def predict(self):
        """
        Performs brain age prediction using StratifiedShuffleSplit cross-validation.

        Returns:
            tuple: A tuple containing metrics DataFrames (training and validation, if available),
                   predictions DataFrames (training and validation, if available), and a list of models.
        """

        if self.test_size_p > 1 / self.splits:
            warnings.warn("Potential resampling issue: test_size_p is too large.")

        self._prepare_stratified_data()
        all_metrics, all_metrics_val, all_predictions, all_predictions_val, models = self._initialize_prediction_lists()

        sss = StratifiedShuffleSplit(n_splits=self.splits, test_size=self.test_size_p, random_state=self.random_state)

        X = self.data[self.features].values # Access values directly for sklearn
        y = self.data[self.target].values
        X_val = self.data_validation[self.features].values if self.data_validation is not None else None # Access values directly for sklearn
        y_val = self.data_validation[self.target].values if self.data_validation is not None else None

        X_scaled, X_val_scaled, scaler = self._scale_features(X, X_val)


        for i, (train_index, test_index) in enumerate(sss.split(self.data, self.data['fuse_bin'])):
            X_train, y_train, X_test, y_test = self._split_train_test(
                X_scaled, y, self.data['fuse_bin'], train_index, test_index)

            metrics_data, metric_data_val, predictions_data, predictions_data_val = self._train_predict_and_evaluate(
            i, X_train, y_train, X_test, y_test, X_val_scaled, y_val, test_index) # Added test_index

            all_metrics.append(metrics_data)
            if metric_data_val:
                all_metrics_val.append(metric_data_val)
            all_predictions.append(predictions_data)
            if predictions_data_val:
                all_predictions_val.append(predictions_data_val)

            models.append((self.model, X_train[:, 0])) # Store the model and a sample feature for potential later analysis

        metrics_df = pd.DataFrame(all_metrics)
        metrics_df_val = pd.DataFrame(all_metrics_val) if self.data_validation is not None else None
        predictions_df = pd.concat(all_predictions)
        predictions_df_val = pd.concat(all_predictions_val) if self.data_validation is not None else None

        return metrics_df, metrics_df_val, predictions_df, predictions_df_val, models