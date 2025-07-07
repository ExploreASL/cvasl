import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')
import pandas as pd
from scipy import stats
import warnings        

def encode_cat_features(dff,cat_features_to_encode):

    feature_mappings = {}
    reverse_mappings = {}
    data = pd.concat([_d.data for _d in dff])

    for feature in cat_features_to_encode:
        if feature in data.columns:
            unique_values = data[feature].unique()
            mapping = {value: i for i, value in enumerate(unique_values)}
            feature_mappings[feature] = mapping
            reverse_mappings[feature] = {v: k for k, v in mapping.items()}
            data[feature] = data[feature].map(mapping)

    for _d in dff:
        _d.data = data[data['site'] == _d.site_id]
        _d.feature_mappings = feature_mappings
        _d.reverse_mappings = reverse_mappings
        _d.cat_features_to_encode = cat_features_to_encode
    return dff

class MRIdataset:
    def __init__(
        self,
        path,
        site_id,
        patient_identifier="participant_id",
        cat_features_to_encode=None,
        ICV=False,
        decade=False,
        features_to_drop=[],
        features_to_bin=None,
        binning_method="equal_width",
        num_bins=10,
        bin_labels=None,
    ):
        """
        Initializes the MRIdataset class.

        Args:
            path (str or list): Path to the CSV file or list of paths.
            site_id (int or str): Identifier for the site.
            patient_identifier (str, optional): Column name for patient ID. Defaults to "participant_id".
            cat_features_to_encode (list, optional): List of categorical features to encode. Defaults to None.
            ICV (bool, optional): Whether to add ICV-related features. Defaults to False.
            decade (bool, optional): Whether to add decade-related features. Defaults to False.
            features_to_drop (list, optional): List of features to drop. Defaults to ["m0", "id"].
            features_to_bin (list, optional): List of features to bin. Defaults to None.
            binning_method (str, optional): Binning method to use ("equal_width", "equal_frequency"). Defaults to "equal_width".
            num_bins (int, optional): Number of bins. Defaults to 10.
            bin_labels (list, optional): Labels for bins. Defaults to None.
        """
        self._validate_init_arguments(
            path,
            site_id,
            patient_identifier,
            cat_features_to_encode,
            ICV,
            decade,
            features_to_drop,
            features_to_bin,
            binning_method,
            num_bins,
            bin_labels,
        )

        if isinstance(path, list):
            self.data = pd.concat([pd.read_csv(_p) for _p in path])
        else:
            self.data = pd.read_csv(path)

        self.site_id = site_id
        self.data["Site"] = self.site_id
        self.feature_mappings = {}
        self.reverse_mappings = {}
        self.patient_identifier = patient_identifier
        self.icv = ICV
        self.path = path
        self.decade = decade
        self.features_to_drop = features_to_drop
        self.fetures_to_bin = features_to_bin
        self.binning_method = binning_method
        self.num_bins = num_bins
        self.bin_labels = bin_labels
        self.cat_features_to_encode = cat_features_to_encode
        self.initial_statistics = None
        self.harmonized_statistics = None
        self.columns_order = self.data.columns.to_list()
        self.dropped_features = None

    def _validate_init_arguments(
        self,
        path,
        site_id,
        patient_identifier,
        cat_features_to_encode,
        ICV,
        decade,
        features_to_drop,
        features_to_bin,
        binning_method,
        num_bins,
        bin_labels,
    ):
        """Validates arguments passed to the __init__ method."""
        if not isinstance(path, (str, list)):
            raise TypeError("path must be a string or a list of strings.")
        if isinstance(path, list) and not all(isinstance(p, str) for p in path):
            raise TypeError("If path is a list, all elements must be strings.")
        if not isinstance(site_id, (int, str)):
            raise TypeError("site_id must be an integer or a string.")
        if not isinstance(patient_identifier, str):
            raise TypeError("patient_identifier must be a string.")
        if cat_features_to_encode is not None and not isinstance(cat_features_to_encode, list):
            raise TypeError("cat_features_to_encode must be a list or None.")
        if not isinstance(ICV, bool):
            raise TypeError("ICV must be a boolean.")
        if not isinstance(decade, bool):
            raise TypeError("decade must be a boolean.")
        if not isinstance(features_to_drop, list):
            raise TypeError("features_to_drop must be a list.")
        if features_to_bin is not None and not isinstance(features_to_bin, list):
            raise TypeError("features_to_bin must be a list or None.")
        if not isinstance(binning_method, str):
            raise TypeError("binning_method must be a string.")
        if binning_method not in ["equal_width", "equal_frequency"]:
            raise ValueError("binning_method must be either 'equal_width' or 'equal_frequency'.")
        if not isinstance(num_bins, int) or num_bins <= 0:
            raise ValueError("num_bins must be a positive integer.")
        if bin_labels is not None and not isinstance(bin_labels, list):
            raise TypeError("bin_labels must be a list or None.")

    def generalized_binning(self):
        """
        Performs binning on specified features of a Pandas DataFrame.

        Args:
            df: The input DataFrame.
            features: A list of feature names (columns) to bin.
            binning_method: The binning method to use. Options are 'equal_width', 'equal_frequency'.
                Defaults to 'equal_width'.
            num_bins: The number of bins to create (for 'equal_width' and 'equal_frequency'). Defaults to 10.
            labels: Optional labels for the resulting bins (for 'equal_width' and 'equal_frequency').

        Returns:
            A new DataFrame with the binned features.  Returns None if an invalid binning method is specified.
        """

        features = self.fetures_to_bin
        binning_method = self.binning_method
        num_bins = self.num_bins
        labels = self.bin_labels

        df_binned = (
            self.data.copy()
        )  # Create a copy to avoid modifying the original DataFrame

        for feature in features:
            if binning_method == "equal_width":
                df_binned[feature + "_binned"], bins = pd.cut(
                    self.data[feature],
                    bins=num_bins,
                    labels=labels,
                    retbins=True,
                    duplicates="drop",
                )
            elif binning_method == "equal_frequency":
                df_binned[feature + "_binned"], bins = pd.qcut(
                    self.data[feature],
                    q=num_bins,
                    labels=labels,
                    retbins=True,
                    duplicates="drop",
                )
            else:
                return None  # Handle invalid binning methods

        self.data = df_binned

    def encode_categorical_features(self):

        for feature in self.cat_features_to_encode:
            if feature in self.data.columns:
                unique_values = self.data[feature].unique()
                mapping = {value: i for i, value in enumerate(unique_values)}
                self.feature_mappings[feature] = mapping
                self.reverse_mappings[feature] = {v: k for k, v in mapping.items()}
                self.data[feature] = self.data[feature].map(mapping)

    def reverse_encode_categorical_features(self):
        for feature, mapping in self.reverse_mappings.items():
            if feature in self.data.columns:
                unique_values = self.data[feature].unique()
                # Filter the mapping to include only the unique values
                filtered_mapping = {k: mapping[k] for k in unique_values if k in mapping}
                self.data[feature] = self.data[feature].map(filtered_mapping)

    def addICVfeatures(self):
        self.data["icv"] = self.data["gm_vol"] / self.data["gm_icvratio"]

    def addDecadefeatures(self):
        self.data["decade"] = (self.data["age"] / 10).round()
        #self.data = self.data.sort_values(by="age")
        self.data.reset_index(inplace=True)

    def dropFeatures(self):

        self.dropped_features = self.data[[self.patient_identifier] + self.features_to_drop]
        self.data = self.data.drop(self.features_to_drop, axis=1)

    #define a function that would add site_id to the bginning of the patient_identifier column of the data
    def unique_patient_ids(self):
        self.data[self.patient_identifier] = "MRIPatient_Site:" + str(self.site_id) + '_OriginalID:'  + self.data[self.patient_identifier]

    #define a function that restores the patient_identifier column to its original state
    def restore_patient_ids(self):
        self.data[self.patient_identifier] = self.data[self.patient_identifier].str.split('_OriginalID:').str[1]

    def prepare_for_export(self):
        self.restore_patient_ids()
        self.data = self.data.merge(self.dropped_features,on=self.patient_identifier)
        for _c in ['index','Index' ,'ID','unnamed: 0']:
            if _c in self.data.columns:
                self.data = self.data.drop(columns=[_c])
        self.reverse_encode_categorical_features()
        _tc = [_c.lower() for _c in self.columns_order]
        self.data = self.data[_tc]
        self.data.columns = self.columns_order
        for _c in ['index','Index' ,'ID','unnamed: 0']:
            if _c in self.data.columns:
                self.data = self.data.drop(columns=[_c])


    def _extended_summary_statistics(self):
        """
        Calculates extended summary statistics for each column of a Pandas DataFrame.

        Args:
            df: The input DataFrame.

        Returns:
            A Pandas DataFrame containing the summary statistics.
        """
        df = self.data
        summary_stats = []

        for col_name in df.columns:
            col = df[col_name]
            col_type = col.dtype

            stats_dict = {
                "Column Name": col_name,
                "Data Type": col_type,
                "Count": col.count(),
                "Number of Unique Values": col.nunique(),
                "Missing Values": col.isnull().sum(),
            }

            if pd.api.types.is_numeric_dtype(col_type):
                stats_dict.update({
                    "Mean": col.mean(),
                    "Standard Deviation": col.std(),
                    "Minimum": col.min(),
                    "25th Percentile": col.quantile(0.25),
                    "Median (50th Percentile)": col.median(),
                    "75th Percentile": col.quantile(0.75),
                    "Maximum": col.max(),
                    "Skewness": col.skew(),
                    "Kurtosis": col.kurt(),
                })
                if len(col.dropna()) > 1:  # Check for sufficient data points for normality test
                    statistic, p_value = stats.shapiro(col.dropna())
                    stats_dict.update({
                        "Shapiro-Wilk Test Statistic": statistic,
                        "Shapiro-Wilk p-value": p_value
                    })

            elif pd.api.types.is_categorical_dtype(col_type) or pd.api.types.is_object_dtype(col_type):
                mode = col.mode()
                mode_str = ', '.join(mode.astype(str)) # handles multiple modes
                stats_dict.update({
                    "Mode": mode_str
                })
                top_n = 5  # Display top N most frequent categories
                value_counts_df = col.value_counts().nlargest(top_n).reset_index()
                value_counts_df.columns = ['Value', 'Count']
                for i in range(len(value_counts_df)):
                    stats_dict[f"Top {i+1} Most Frequent Value"] = value_counts_df.iloc[i,0]
                    stats_dict[f"Top {i+1} Most Frequent Value Count"] = value_counts_df.iloc[i,1]

            summary_stats.append(stats_dict)

        return pd.DataFrame(summary_stats)


    def preprocess(self):
        """
        Applies preprocessing steps to the MRIdataset.

        This includes:
            - Lowercasing column names.
            - Dropping specified features.
            - Creating unique patient IDs.
            - Adding decade-based features (optional).
            - Adding ICV-related features (optional).
            - Encoding categorical features (optional).
            - Binning specified features (optional).
        """
        # Common preprocessing steps
        self.data.columns = self.data.columns.str.lower()
        if self.features_to_drop:
            self._drop_features() # Use private method

        self._unique_patient_ids() # Use private method

        if self.decade:
            self._add_decade_features() # Use private method
        if self.icv:
            self._add_icv_features() # Use private method
        if self.cat_features_to_encode:
            self._encode_categorical_features() # Use private method
        if self.fetures_to_bin:
            self._generalized_binning() # Use private method
        #self.initial_statistics = self._extended_summary_statistics() # keep for now, decide later if needed

    def _drop_features(self):
        """Drops specified features from the dataset."""
        self.dropped_features = self.data[[self.patient_identifier] + self.features_to_drop]
        self.data = self.data.drop(self.features_to_drop, axis=1, errors='ignore') # Added errors='ignore'

    def _unique_patient_ids(self):
        """Adds site_id to the beginning of the patient_identifier column to ensure uniqueness across sites."""
        self.data[self.patient_identifier] = "MRIPatient_Site:" + str(self.site_id) + '_OriginalID:'  + self.data[self.patient_identifier]

    def _add_decade_features(self):
        """Adds decade-based features based on the 'age' column."""
        if 'age' in self.data.columns:
            self.data["decade"] = (self.data["age"] / 10).round()
            self.data.reset_index(inplace=True, drop=True) # reset index after sorting and inplace=True
        else:
            warnings.warn("Age column not found, cannot add decade features.")

    def _add_icv_features(self):
        """Adds ICV-related features by calculating 'icv' ratio."""
        if 'gm_vol' in self.data.columns and 'gm_icvratio' in self.data.columns:
            self.data["icv"] = self.data["gm_vol"] / self.data["gm_icvratio"]
        else:
            warnings.warn("Required columns 'gm_vol' or 'gm_icvratio' not found, cannot add ICV features.")

    def _encode_categorical_features(self):
        """Encodes categorical features specified in self.cat_features_to_encode."""
        for feature in self.cat_features_to_encode:
            if feature in self.data.columns:
                unique_values = self.data[feature].unique()
                mapping = {value: i for i, value in enumerate(unique_values)}
                self.feature_mappings[feature] = mapping
                self.reverse_mappings[feature] = {v: k for k, v in mapping.items()}
                self.data[feature] = self.data[feature].map(mapping)
            else:
                warnings.warn(f"Categorical feature '{feature}' not found in dataset, skipping encoding.")

    def _generalized_binning(self):
        """Applies generalized binning to features specified in self.fetures_to_bin."""
        features = self.fetures_to_bin
        binning_method = self.binning_method
        num_bins = self.num_bins
        labels = self.bin_labels

        df_binned = self.data.copy() # Create copy to avoid modifying original

        for feature in features:
            if feature not in self.data.columns:
                warnings.warn(f"Feature '{feature}' not found in dataset, skipping binning.")
                continue # Skip to next feature if current one is not found

            if binning_method == "equal_width":
                df_binned[feature + "_binned"], bins = pd.cut(
                    self.data[feature],
                    bins=num_bins,
                    labels=labels,
                    retbins=True,
                    duplicates="drop",
                )
            elif binning_method == "equal_frequency":
                df_binned[feature + "_binned"], bins = pd.qcut(
                    self.data[feature],
                    q=num_bins,
                    labels=labels,
                    retbins=True,
                    duplicates="drop",
                )
            else:
                warnings.warn(f"Invalid binning method '{binning_method}', skipping binning for feature '{feature}'.")
                continue # Skip to next feature if binning method is invalid

        self.data = df_binned
