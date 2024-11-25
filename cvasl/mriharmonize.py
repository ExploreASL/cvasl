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
import cvasl.harmony as har
from scipy import stats
from tabulate import tabulate
import IPython.display as dp # for HTML display
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
import warnings
from sklearn.preprocessing import StandardScaler
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import subprocess

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
        print(_d.path)
        print(_d.data.columns)
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
        cat_features_to_encode=None,
        ICV=False,
        decade=False,
        features_to_drop=["m0", "id"],
        features_to_bin=None,
        binning_method="equal_width",
        num_bins=10,
        bin_labels=None,
    ):
        if type(path) == list:
            self.data = pd.concat(pd.read_csv(_p) for _p in path)
        else:
            self.data = pd.read_csv(path)
        self.site_id = site_id
        self.data["Site"] = self.site_id
        self.feature_mappings = {}
        self.reverse_mappings = {}
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

    # def reverse_encode_categorical_features(self):
    #     for feature, mapping in self.reverse_mappings.items():
    #         if feature in self.data.columns:
    #             # Only map values that exist in the column
    #             self.data[feature] = self.data[feature].map(
    #                 lambda x: mapping.get(x, x) if x in mapping else x
    #             )
    
                    
    def addICVfeatures(self):
        self.data["icv"] = self.data["gm_vol"] / self.data["gm_icvratio"]

    def addDecadefeatures(self):
        self.data["decade"] = (self.data["age"] / 10).round()
        self.data = self.data.sort_values(by="age")
        self.data.reset_index(inplace=True)

    def dropFeatures(self):
        self.data = self.data.drop(self.features_to_drop, axis=1)

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
        # Common preprocessing steps
        self.data.columns = self.data.columns.str.lower()
        print(1,self.data.columns)
        if self.features_to_drop:
            self.dropFeatures()
        print(2,self.data.columns)
        if self.decade:
            self.addDecadefeatures()
        print(3,self.data.columns)
        if self.icv:
            self.addICVfeatures()
        print(4,self.data.columns)
        if self.cat_features_to_encode:
            self.encode_categorical_features()
        print(5,self.data.columns)
        if self.fetures_to_bin:
            self.generalized_binning()
        print(6,self.data.columns)
        #self.initial_statistics = self._extended_summary_statistics()
    
    def update_harmonized_statistics(self):
        self.harmonized_statistics = self._extended_summary_statistics()
        


class EDISdataset(MRIdataset):
    def __init__(
        self,
        path,
        site_id,
        cat_features_to_encode=None,
        ICV=False,
        decade=False,
        features_to_drop=["m0", "id"],
        features_to_bin=None,
        binning_method="equal_width",
        num_bins=10,
        bin_labels=None,
    ):
        super().__init__(
            path,
            site_id,
            cat_features_to_encode,
            ICV,
            decade,
            features_to_drop,
            features_to_bin,
            binning_method,
            num_bins,
            bin_labels,
        )
        self.preprocess()


class HELIUSdataset(MRIdataset):
    def __init__(
        self,
        path,
        site_id,
        cat_features_to_encode=None,
        ICV=False,
        decade=False,
        features_to_drop=["m0", "id"],
        features_to_bin=None,
        binning_method="equal_width",
        num_bins=10,
        bin_labels=None,
    ):
        super().__init__(
            path,
            site_id,
            cat_features_to_encode,
            ICV,
            decade,
            features_to_drop,
            features_to_bin,
            binning_method,
            num_bins,
            bin_labels,
        )
        self.preprocess()

    def preprocess(self):
        super().preprocess()
        # Specific preprocessing for HELIUS
        self.data.loc[
            self.data["participant_id"] == "sub-153852_1", "participant_id"
        ] = "sub-153852_1H"


class SABREdataset(MRIdataset):
    def __init__(
        self,
        path,
        site_id,
        cat_features_to_encode=None,
        ICV=False,
        decade=False,
        features_to_drop=["m0", "id"],
        features_to_bin=None,
        binning_method="equal_width",
        num_bins=10,
        bin_labels=None,
    ):
        super().__init__(
            path,
            site_id,
            cat_features_to_encode,
            ICV,
            decade,
            features_to_drop,
            features_to_bin,
            binning_method,
            num_bins,
            bin_labels,
        )
        self.preprocess()


class StrokeMRIdataset(MRIdataset):
    def __init__(
        self,
        path,
        site_id,
        cat_features_to_encode=None,
        ICV=False,
        decade=False,
        features_to_drop=["m0", "id"],
        features_to_bin=None,
        binning_method="equal_width",
        num_bins=10,
        bin_labels=None,
    ):
        super().__init__(
            path,
            site_id,
            cat_features_to_encode,
            ICV,
            decade,
            features_to_drop,
            features_to_bin,
            binning_method,
            num_bins,
            bin_labels,
        )
        self.preprocess()


class TOPdataset(MRIdataset):
    def __init__(
        self,
        path,
        site_id,
        cat_features_to_encode=None,
        ICV=False,
        decade=False,
        features_to_drop=["m0", "id"],
        features_to_bin=None,
        binning_method="equal_width",
        num_bins=10,
        bin_labels=None,
    ):
        super().__init__(
            path,
            site_id,
            cat_features_to_encode,
            ICV,
            decade,
            features_to_drop,
            features_to_bin,
            binning_method,
            num_bins,
            bin_labels,
        )
        self.preprocess()


class Insight46dataset(MRIdataset):
    def __init__(
        self,
        path,
        site_id,
        cat_features_to_encode=None,
        ICV=False,
        decade=False,
        features_to_drop=["m0", "id"],
        features_to_bin=None,
        binning_method="equal_width",
        num_bins=10,
        bin_labels=None,
    ):
        super().__init__(
            path,
            site_id,
            cat_features_to_encode,
            ICV,
            decade,
            features_to_drop,
            features_to_bin,
            binning_method,
            num_bins,
            bin_labels,
        )
        self.preprocess()


class HarmNeuroHarmonize:
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
            Ff empty, ComBat is applied with a linear model of covariates.
            Otherwise, Generalized Additive Models (GAMs) are used.
            Using it will increase computation time due to search for optimal smoothing.
            
        site_indicator : a string 
            Indicates the feature that differentiates different sites, default 'site'
            
        empirical_bayes : bool, default True
            Whether to use empirical Bayes estimates of site effects              
        """        

        self.features_to_harmonize = features_to_harmonize
        self.covariates = covariates
        self.smooth_terms = smooth_terms
        self.empirical_bayes = empirical_bayes
        self.site_indicator = site_indicator

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
        # comb = MRIdataset

        # Combine all datasets
        all_data = pd.concat([dataset.data for dataset in mri_datasets])

        # Prepare data for harmonization
        features_data = all_data[self.features_to_harmonize]
        covariates_data = all_data[self.covariates]
        covariates_data = covariates_data.rename(columns={self.site_indicator: "SITE"})

        # Perform harmonization
        _, harmonized_data = harmonizationLearn(
            np.array(features_data), covariates_data, smooth_terms=self.smooth_terms,eb=self.empirical_bayes
        )

        # Create harmonized dataframe
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

        for _d in mri_datasets:
            _d.data = harmonized_df[harmonized_df["SITE"] == _d.site_id]
            _d.data = _d.data.drop(columns=["SITE",'index'])
        # [_d.update_harmonized_statistics() for _d in mri_datasets]
        return mri_datasets


class HarmComscanNeuroCombat:
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
        features_to_harmonize : a list
            Features to harmonize excluding covariates and site indicator
        
        discrete_covariates : a list
            Contains discrete covariates to control for during harmonization.
            All covariates must be encoded numerically (no categorical variables).
            
        continuous_covariates : a list
            Contains discrete covariates to control for during harmonization.
            All covariates must be encoded numerically (no categorical variables).

        smooth_terms (Optional) :  a list, default []
            Names of columns in covars to include as smooth, nonlinear terms.
            Can be any or all columns in covars, except site_indicator.
            Ff empty, ComBat is applied with a linear model of covariates.
            Otherwise, Generalized Additive Models (GAMs) are used.
            Using it will increase computation time due to search for optimal smoothing.
            
        site_indicator : a string 
            Indicates the feature that differentiates different sites, default 'site'
            
        empirical_bayes : bool, default True
            Whether to use empirical Bayes estimates of site effects              
        """        

        self.features_to_harmonize = features_to_harmonize
        self.site_indicator = [site_indicator]
        self.discrete_covariates = (
            discrete_covariates if discrete_covariates is not None else []
        )
        self.continuous_covariates = (
            continuous_covariates if continuous_covariates is not None else []
        )
        self.empirical_bayes = empirical_bayes
        self.parametric = parametric
        self.mean_only = mean_only
        

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

        if (
            not self.features_to_harmonize
        ):  # Handle the case where no features need harmonization
            return None

        data = pd.concat([dataset.data for dataset in mri_datasets])

        # Instantiate ComBat object
        combat = cvaslneurocombat.Combat(
            features=self.features_to_harmonize,
            sites=self.site_indicator,
            discrete_covariates=self.discrete_covariates,
            continuous_covariates=self.continuous_covariates,
            empirical_bayes=self.empirical_bayes,
            parametric=self.parametric,
            mean_only=self.mean_only
        )

        # Select relevant data for harmonization

        data_to_harmonize = data[
            self.features_to_harmonize
            + self.site_indicator
            + self.discrete_covariates
            + self.continuous_covariates
        ].copy()

        # Transform the data using ComBat
        harmonized_data = combat.fit(data_to_harmonize)
        harmonized_data = combat.transform(data_to_harmonize)

        # Create harmonized DataFrame
        harmonized_df = pd.DataFrame(
            harmonized_data, columns=self.features_to_harmonize
        )

        # Add back covariates and unharmonized features
        covariates = self.site_indicator + self.discrete_covariates + self.continuous_covariates
        harmonized_df = pd.concat(
            [harmonized_df, data_to_harmonize[covariates].reset_index(drop=True)],
            axis=1,
        )

        
        harmonized_df = pd.concat(
                [harmonized_df, data[[i for i in data.columns if i not in harmonized_df.columns]].reset_index(drop=True)],
                axis=1,
            )

        # Reorder columns to maintain original order as much as possible while putting harmonized features first
        original_order = list(data.columns)
        new_order = self.features_to_harmonize + [
            col for col in original_order if col not in self.features_to_harmonize
        ]
        harmonized_df = harmonized_df[
            [col for col in new_order if col in harmonized_df.columns]
        ]

        for _, dataset in enumerate(mri_datasets):
            site_value = dataset.site_id
            adjusted_data = harmonized_df[harmonized_df[self.site_indicator[0]] == site_value]
            dataset.data = adjusted_data
        # [_d.update_harmonized_statistics() for _d in mri_datasets]
        return mri_datasets

class HarmAutoCombat:
    def __init__(
        self,
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
        features_to_harmonize : a list
            Features to harmonize excluding covariates and site indicator

        site_indicator : a string 
            Indicates the feature that differentiates different sites, default 'site'
        
        discrete_covariates : a list
            Contains discrete covariates to control for during harmonization.
            All covariates must be encoded numerically (no categorical variables).
            
        continuous_covariates : a list
            Contains discrete covariates to control for during harmonization.
            
        discrete_cluster_features : a list
            Target sites_features which are categorical to one-hot (e.g. ManufacturerModelName).
            All covariates must be encoded numerically (no categorical variables).
            
        continuous_cluster_features : a list
            Target sites_features which are continuous to scale (e.g. EchoTime).

        metric : "distortion", "silhouette" or "calinski_harabasz", default "distortion"
            Metric to define the optimal number of cluster.
            
        features_reduction : 'pca' or 'umap', default None
            Method for reduction of the embedded space with n_components. Can be 'pca' or 'umap'.

        feature_reduction_dimensions : int, default 2
            Dimension of the embedded space for features reduction.
            
        empirical_bayes : bool, default True
            Whether to use empirical Bayes estimates of site effects
        """        

        
        self.features_to_harmonize = features_to_harmonize
        self.site_indicator = site_indicator
        self.discrete_covariates = (
            discrete_covariates if discrete_covariates is not None else []
        )
        self.continuous_covariates = (
            continuous_covariates if continuous_covariates is not None else []
        )
        self.metric = metric
        self.features_reduction = features_reduction
        self.continuous_cluster_features = continuous_cluster_features
        self.discrete_cluster_features = discrete_cluster_features
        self.features_reduction_dimensions = feature_reduction_dimensions
        self.empirical_bayes = empirical_bayes
        

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

        if (
            not self.features_to_harmonize
        ):  # Handle the case where no features need harmonization
            return None

        data = pd.concat([dataset.data for dataset in mri_datasets])

        # Instantiate ComBat object
        combat = cvaslneurocombat.AutoCombat(
            features = self.features_to_harmonize,
            metric = self.metric,
            sites_features=self.site_indicator,
            discrete_combat_covariates = self.discrete_covariates,
            continuous_combat_covariates = self.continuous_covariates,
            continuous_cluster_features=self.continuous_cluster_features,
            discrete_cluster_features=self.discrete_cluster_features,
            size_min=2,
            features_reduction = self.features_reduction,
            n_components =self.features_reduction_dimensions,
            n_jobs = 8, empirical_bayes=self.empirical_bayes)

        # Select relevant data for harmonization
        _ft = list(dict.fromkeys(
            self.features_to_harmonize
            + self.site_indicator
            + self.discrete_covariates
            + self.continuous_covariates
        ))        
        data_to_harmonize = data[_ft].copy()

        # Transform the data using ComBat
        harmonized_data = combat.fit(data_to_harmonize)
        harmonized_data = combat.transform(data_to_harmonize)

        # Create harmonized DataFrame
        harmonized_df = pd.DataFrame(
            harmonized_data, columns=self.features_to_harmonize
        )

        # Add back covariates and unharmonized features
        covariates = self.site_indicator + self.discrete_covariates + self.continuous_covariates
        harmonized_df = pd.concat(
            [harmonized_df, data_to_harmonize[covariates].reset_index(drop=True)],
            axis=1,
        )

        
        harmonized_df = pd.concat(
                [harmonized_df, data[[i for i in data.columns if i not in harmonized_df.columns]].reset_index(drop=True)],
                axis=1,
            )

        # Reorder columns to maintain original order as much as possible while putting harmonized features first
        original_order = list(data.columns)
        new_order = self.features_to_harmonize + [
            col for col in original_order if col not in self.features_to_harmonize
        ]
        harmonized_df = harmonized_df[
            [col for col in new_order if col in harmonized_df.columns]
        ]

        for _, dataset in enumerate(mri_datasets):
            site_value = dataset.site_id
            # adjusted_data = harmonized_df[harmonized_df["site"] == site_value]
            adjusted_data = harmonized_df[harmonized_df[self.site_indicator] == site_value]
            dataset.data = adjusted_data
        # [_d.update_harmonized_statistics() for _d in mri_datasets]
        return mri_datasets

class HarmCovbat:
    def __init__(
        self, features_to_harmonize,  covariates, site_indicator='site', patient_identifier = 'participant_id', numerical_covariates = ['age'], empirical_bayes = True
    ):
        """
        Wrapper class for Covbat.
        
        Arguments
        ---------
        features_to_harmonize : a list
            Features to harmonize excluding covariates and site indicator

        covariates : a list
            Contains covariates to control for during harmonization.
            All covariates must be encoded numerically (no categorical variables)

        site_indicator : a string 
            Indicates the feature that differentiates different sites (batches in the data in the original covbat documentation), default 'site'
        
        patient_identifier : string
            Indicates the feature that differentiates different patients, default 'participant_id'            

        numerical_covariates : a list
            Contains discrete covariates to control for during harmonization.
                        
        empirical_bayes : bool, default True
            Whether to use empirical Bayes estimates of site effects
        """        
        
        self.features_to_harmonize = [a.lower() for a in features_to_harmonize]
        self.covariates = [a.lower() for a in covariates]
        self.site_indicator = site_indicator.lower()
        self.patient_identifier = patient_identifier.lower()
        self.numerical_covariates = [a.lower() for a in numerical_covariates]
        self.empirical_bayes = empirical_bayes

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
        semi_features = []
        datasets_to_harmonize = []
        
        for dataset in mri_datasets:
            semi_features.append(dataset.data.drop([_c for _c in self.features_to_harmonize if _c not in [self.patient_identifier]], axis=1))
            datasets_to_harmonize.append(dataset.data.drop([c for c in dataset.data.columns if c not in self.features_to_harmonize + self.covariates + [self.site_indicator] + [self.patient_identifier]], axis=1))            
            
        pheno_features = [self.patient_identifier] + self.covariates + [self.site_indicator]
        ALLFIVE = pd.concat(datasets_to_harmonize)
        
        # Prepare data for CovBat
        
        phenoALLFIVE = ALLFIVE[pheno_features]
        phenoALLFIVE = phenoALLFIVE.set_index(self.patient_identifier)

        dat_ALLFIVE = ALLFIVE.set_index(self.patient_identifier)
          
        dat_ALLFIVE = dat_ALLFIVE.T
        mod_matrix = patsy.dmatrix(
            f"~ {' + '.join(self.covariates)}", phenoALLFIVE, return_type="dataframe"
        )

        # Perform harmonization using CovBat
        harmonized_data = covbat.combat(
            data = dat_ALLFIVE,
            batch = phenoALLFIVE[self.site_indicator],
            model=mod_matrix,
            numerical_covariates=self.numerical_covariates,
            eb=self.empirical_bayes
        )
        
        harmonized_data = harmonized_data[
            len(self.covariates) :
        ]  # Remove estimated model parameters from the output
        feature_cols = [col for col in harmonized_data.index if col not in (self.site_indicator)]
        
        harmonized_data = harmonized_data.loc[feature_cols]#.reset_index(drop=True).dropna() # Directly select features, reset index, and drop NaN rows
        # Combine harmonized data with other features
        harmonized_data = pd.concat(
            [dat_ALLFIVE.head(len(self.covariates) + 1), harmonized_data]
        )  # Add back the ID and model parameters
        harmonized_data = harmonized_data.T
        harmonized_data = harmonized_data.reset_index()
        # Split the harmonized data back into individual datasets
        for i, dataset in enumerate(mri_datasets):
            site_value = dataset.site_id
            adjusted_data = harmonized_data[harmonized_data[self.site_indicator] == site_value]
            adjusted_data = adjusted_data.merge(semi_features[i], on=self.patient_identifier)
            dataset.data = adjusted_data
        # [_d.update_harmonized_statistics() for _d in mri_datasets]
        return mri_datasets


class HarmNeuroCombat:
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
        features_to_harmonize : a list
            Features to harmonize excluding covariates and site indicator
        
        discrete_covariates : a list
            Contains discrete covariates to control for during harmonization.
            All covariates must be encoded numerically (no categorical variables).
            
        continuous_covariates : a list
            Contains discrete covariates to control for during harmonization.
            All covariates must be encoded numerically (no categorical variables).

        patient_identifier : string
            Indicates the feature that differentiates different patients, default 'participant_id'
            
        
        site_indicator : a string 
            Indicates the feature that differentiates different sites, default 'site'
            
        empirical_bayes : bool, default True
            Whether to use empirical Bayes estimates of site effects              
            
        mean_only : bool, default False
            Whether to use only the mean of the data for harmonization.
            
        parametric : bool, default True
            Whether parametric adjustements should be performed.
        """        
        
        self.discrete_covariates = [a.lower() for a in discrete_covariates]
        self.continuous_covariates = [a.lower() for a in continuous_covariates]
        self.features_to_harmonize = [a.lower() for a in features_to_harmonize]
        self.patient_identifier = patient_identifier.lower()
        self.site_indicator = site_indicator.lower()
        self.empirical_bayes = empirical_bayes
        self.mean_only = mean_only
        self.parametric = parametric


    def _prep_for_neurocombat_5way(self,
            dataframes):
        dataframes = [a.set_index(self.patient_identifier).T for a in dataframes]
        # concat the two dataframes
        all_togetherF = pd.concat(
            dataframes,
            axis=1,
            join="inner",
        )

        # create a feautures only frame (no age, no sex)
        
        
        feature_cols = [col for col in all_togetherF.index if col not in self.discrete_covariates + self.continuous_covariates]
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

        return all_togetherF, ftF, btF, feature_dictF, lens


    def _make_topper(self, bt, row_labels):
        topper = (
            bt.head(len(row_labels))
            .rename_axis(None, axis="columns")
            .reset_index()
        )
        topper['index'] = row_labels
        return topper

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
        # Separate features for harmonization and those to be kept unharmonized
        semi_features = []
        datasets_to_harmonize = []
        ocols = mri_datasets[0].data.columns
        for dataset in mri_datasets:
            semi_features.append(
                dataset.data.drop(
                    columns=[
                        f
                        for f in self.features_to_harmonize
                        if f in dataset.data.columns
                    ],
                )
            )
            datasets_to_harmonize.append(dataset.data[(self.discrete_covariates + self.continuous_covariates + self.features_to_harmonize + [self.site_indicator] + [self.patient_identifier])])
        # Prepare data for NeuroCombat
        all_together, ft, bt, feature_dict, lengths = self._prep_for_neurocombat_5way(
            datasets_to_harmonize
        )
        # Create covariates
        batch_ids = []
        for i, l in enumerate(lengths):
            batch_ids.extend([i + 1] * l)  # start batch numbers at 1

        covars = {self.site_indicator: batch_ids}
        for feature in self.discrete_covariates + self.continuous_covariates:
            feature_lower = feature.lower()
            if feature_lower in all_together.index:
                covars[feature] = all_together.loc[feature_lower, :].values.tolist()
        covars = pd.DataFrame(covars)
        # Convert data to numpy array for NeuroCombat
        data = ft.values
        

        # Harmonize data using NeuroCombat
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

        # Convert harmonized data back to DataFrame
        neurocombat_df = pd.DataFrame(data_combat)
        # Reconstruct the full dataframe
        topper = self._make_topper(bt, self.discrete_covariates + self.continuous_covariates)
        bottom = neurocombat_df.reset_index(drop=False)
        bottom = bottom.rename(columns={"index": "char"})
        bottom.columns = topper.columns  # align columns with topper
        back_together = pd.concat([topper, bottom]).T
        new_header = back_together.iloc[0]
        back_together = back_together[1:]
        back_together.columns = new_header
        
  
        
        # Split harmonized data back into original datasets
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
                semi_features[i], on=self.patient_identifier
            )  # Merge back the unharmonized features
            harmonized_datasets.append(harmonized_data)
            start = end

        harmonized_data = pd.concat([_d for _d in harmonized_datasets])
        for i, dataset in enumerate(mri_datasets):
            site_value = dataset.site_id
            adjusted_data = harmonized_data[harmonized_data[self.site_indicator] == site_value]
            adjusted_data = adjusted_data.merge(semi_features[i].drop(self.discrete_covariates + self.continuous_covariates + ['index'],axis = 1), on=self.patient_identifier).drop(['index'],axis = 1) 
            for _c in ocols:
                if _c + '_y' in adjusted_data.columns and _c + '_x' in adjusted_data.columns:
                    adjusted_data.drop(columns=[_c+'_y'],axis=1, inplace=True)
                    adjusted_data.rename(columns={_c + '_x': _c}, inplace=True)
            
            # adjusted_data = adjusted_data.drop(self.site_col, axis=1)
            dataset.data = adjusted_data
#        [_d.update_harmonized_statistics() for _d in mri_datasets]
        return mri_datasets


class HarmNestedComBat:
    def __init__(self, features_to_harmonize, site_indicator = ['site'], discrete_covariates = ['sex'], continuous_covariates = ['age'], intermediate_results_path = '.', patient_identifier = 'participant_id', return_extended = False):
        """
        Wrapper class for Nested Combat.
        
        Arguments
        ---------
        features_to_harmonize : a list
            Features to harmonize excluding covariates and site indicator
        
        discrete_covariates : a list
            Contains discrete covariates to control for during harmonization.
            All covariates must be encoded numerically (no categorical variables).
            
        continuous_covariates : a list
            Contains discrete covariates to control for during harmonization.
            All covariates must be encoded numerically (no categorical variables).

        patient_identifier : string
            Indicates the feature that differentiates different patients, default 'participant_id'
            
        
        site_indicator : a string 
            Indicates the feature that differentiates different sites, default 'site'
            
        intermediate_results_path : string
            Path to save intermediate results of the harmonization process
            
        return_extended : bool, default False
            Whether to return the intermediate results of the harmonization process
        """        
        
        self.features_to_harmonize = [a.lower() for a in features_to_harmonize]
        self.site_indicator = [a.lower() for a in site_indicator]
        self.discrete_covariates = [a.lower() for a in discrete_covariates]
        self.continuous_covariates = [a.lower() for a in continuous_covariates]
        self.intermediate_results_path = intermediate_results_path
        self.patient_identifier = patient_identifier
        self.return_extended = return_extended

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
        
        datasets = []
        batch_testing_df = []
        for ds in mri_datasets:
            datasets.append(ds.data.copy())
            #batch_testing_df.append(ds.data[['participant_id','site','age','sex']])
            batch_testing_df.append(ds.data[[self.patient_identifier] + self.site_indicator+self.continuous_covariates+self.discrete_covariates])
            ds.data.drop(self.features_to_harmonize,axis=1, inplace = True)
        batch_testing_df = pd.concat(batch_testing_df)
        
        datasets = [a.drop([b for b in a.columns if b not in self.features_to_harmonize + self.discrete_covariates + self.continuous_covariates + [self.patient_identifier]],axis=1) for a in datasets]
        data_testing_df = pd.concat([a.drop(columns=self.discrete_covariates + self.continuous_covariates) for a in datasets]).reset_index(drop=True).dropna().merge(batch_testing_df[self.patient_identifier], 
                                        left_on=self.patient_identifier, right_on=self.patient_identifier)
        dat_testing = data_testing_df.iloc[:, 1:].T.apply(pd.to_numeric)
        caseno_testing = data_testing_df[self.patient_identifier]
        covars_testing = batch_testing_df.drop(self.patient_identifier, axis=1)
        data_testing_df= data_testing_df.drop_duplicates()
        
        covars_testing_string = pd.DataFrame()
        covars_testing_string[self.discrete_covariates] = covars_testing[self.discrete_covariates].copy()
        covars_testing_quant = covars_testing[self.continuous_covariates]
        
        covars_testing_cat = pd.DataFrame()
        for col_testing in covars_testing_string:
            stringcol_testing = covars_testing_string[col_testing]
            le = LabelEncoder()
            le.fit(list(stringcol_testing))
            covars_testing_cat[col_testing] = le.transform(stringcol_testing)        
        
        
        covars_testing_final = pd.concat([covars_testing_cat, covars_testing_quant])
        
        ##################################### GMM Splitting #####################################
        gmm_testing_df = nest.GMMSplit(dat_testing, caseno_testing, self.intermediate_results_path)
        
        gmm_testing_df_merge = batch_testing_df.merge(gmm_testing_df, right_on='Patient', left_on=self.patient_identifier)
        gmm_testing_df_merge['GMM'] = gmm_testing_df_merge['Grouping'] 

        gmm_testing_df_merge[gmm_testing_df_merge.participant_id.duplicated()]

        covars_testing_final = gmm_testing_df_merge.drop([self.patient_identifier,'Patient','Grouping'],axis=1)
        discrete_covariates = self.discrete_covariates + ['GMM']
        #################################### GMM Splitting #####################################
        
        # gmm_testing_df_merge = batch_testing_df

        # covars_testing_final = gmm_testing_df_merge.drop([self.patient_identifier],axis=1)
        # discrete_covariates = self.discrete_covariates
    
        
        
        output_testing_df = nest.OPNestedComBat(dat_testing,
                                        covars_testing_final,
                                        self.site_indicator,
                                        self.intermediate_results_path, categorical_cols=discrete_covariates,
                                        continuous_cols=self.continuous_covariates)

        write_testing_df = pd.concat([caseno_testing, output_testing_df], axis=1) 
        write_testing_df.to_csv(self.intermediate_results_path+'/Mfeatures_testing_NestedComBat.csv') # write results fo file
        dat_testing_input = dat_testing.transpose()
        dat_testing_input.to_csv(self.intermediate_results_path+'/Mfeatures_input_testing_NestedComBat.csv')
        covars_testing_final.to_csv(self.intermediate_results_path+'/Mcovars_input_testing_NestedComBat.csv')

        
        complete_harmonised = pd.concat([write_testing_df, covars_testing_final], axis=1)   
        
        complete_harmonised = complete_harmonised.loc[:,~complete_harmonised.columns.duplicated()].copy()
        
        for _ds in mri_datasets:
            
            ds_opn_harmonized = complete_harmonised[complete_harmonised['site'] == _ds.site_id]
            ds_opn_harmonized = ds_opn_harmonized.drop(columns=self.site_indicator+['GMM',])      
            ds_opn_harmonized  = ds_opn_harmonized.merge( _ds.data, on=self.patient_identifier)
            _ds.data = ds_opn_harmonized.copy()
        # [_d.update_harmonized_statistics() for _d in mri_datasets]
        if self.return_extended:
            mri_datasets,write_testing_df,dat_testing_input,covars_testing_final
        else:
            return mri_datasets

class HarmRELIEF:
    def __init__(
        self, features_to_harmonize, covariates, patient_identifier = 'participant_id', intermediate_results_path = '.'
    ):
        """
        Wrapper class for RELIEF.
        
        Arguments
        ---------
        features_to_harmonize : a list
            Features to harmonize excluding covariates and site indicator
        
        covariates : a list
            Contains covariates to control for during harmonization.
            All covariates must be encoded numerically (no categorical variables)

        patient_identifier : string
            Indicates the feature that differentiates different patients, default 'participant_id'
            
        intermediate_results_path : string
            Path to save intermediate results of the harmonization process
        """        
        
        self.features_to_harmonize = [a.lower() for a in features_to_harmonize]
        self.intermediate_results_path = intermediate_results_path
        self.covariates = covariates
        self.patient_identifier = patient_identifier.lower()

    def _prep_for_neurocombat_5way(self,
            dataframes):
        """
        This function takes five dataframes in the cvasl format,
        then turns them into the items needed for the
        neurocombat algorithm with re-identification.



        :returns: dataframes for neurocombat algorithm and ints of some legnths
        :rtype: tuple
        """
        dataframes = [a.set_index('participant_id').T for a in dataframes]
        # concat the two dataframes
        all_togetherF = pd.concat(
            dataframes,
            axis=1,
            join="inner",
        )

        # create a feautures only frame (no age, no sex)
        
        
        #feature_cols = [col for col in all_togetherF.index if col not in ('sex', 'age')]
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
        topper = (
            bt.head(len(row_labels))
            .rename_axis(None, axis="columns")
            .reset_index(drop=False)
        )
        topper = topper.rename(columns={"index": "char"})
        topper["char"] = row_labels
        return topper

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
        
        # all_togetherF, ftF, btF, feature_dictF, len1, len2, len3, len4, len5 = self._prep_for_neurocombat_5way([_d.data[self.features_to_harmonize + ['participant_id','sex','age']] for _d in mri_datasets])
        all_togetherF, ftF, btF, feature_dictF, len1, len2, len3, len4, len5 = self._prep_for_neurocombat_5way([_d.data[self.features_to_harmonize + [self.patient_identifier] + self.covariates] for _d in mri_datasets])
        
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
        covars = {'batch':first_columns_as_one + second_columns_as_two + third_columns_as_three + fourth_columns_as_four + fifth_columns_as_five}
        for _c in self.covariates:
            covars[_c] = all_togetherF.loc[_c, :].values.tolist()
        covars = pd.DataFrame(covars)
        covars.to_csv(f'{self.intermediate_results_path}/bath_and_mod_forRELIEF5.csv')
        topperF = self._make_topper(btF,self.covariates)
        #subprocess.run(['Rscript', 'CVASL_RELIEF_DRIVER.R'])
        
        r = robjects.r
        r(relief_r_driver)
        bottom = pd.read_csv(f'{self.intermediate_results_path}/relief1_for5_results.csv', index_col=0).reset_index(drop=False).rename(columns={"index": "char"})
        bottom.columns = topperF.columns
        back_together = pd.concat([topperF, bottom])
        back_together = back_together.T
        new_header = back_together.iloc[0] #grab the first row for the header
        back_together.columns = new_header #set the header row as the df header
        back_together = back_together[1:]
        new_feature_dict =  har.increment_keys(feature_dictF)
        
        for _d in mri_datasets:
            _d.data = _d.data.drop(self.features_to_harmonize, axis=1)

        # Keep track of cumulative length
        cum_len = 0

        # Process each dataset
        for i in range(len(mri_datasets)):
            current_len = len(mri_datasets[i].data)
            df = back_together.iloc[cum_len:cum_len + current_len].rename(new_feature_dict, axis='columns').reset_index().rename(columns={"index": self.patient_identifier})
            mri_datasets[i].data = mri_datasets[i].data.merge(df.drop(self.covariates, axis=1), on=self.patient_identifier)
            cum_len += current_len
        
        # [_d.update_harmonized_statistics() for _d in mri_datasets]
        return mri_datasets


class HarmCombatPlusPlus:
    def __init__(
        self, features_to_harmonize, discrete_covariates, continuous_covariates, discrete_covariates_to_remove, continuous_covariates_to_remove, patient_identifier = 'participant_id', intermediate_results_path = '.', site_indicator = 'site'
    ):
        """
        Wrapper class for RELIEF.
        
        Arguments
        ---------
        features_to_harmonize : a list
            Features to harmonize excluding covariates and site indicator
        
        covariates : a list
            Contains covariates to control for during harmonization.
            All covariates must be encoded numerically (no categorical variables)

        patient_identifier : string
            Indicates the feature that differentiates different patients, default 'participant_id'
            
        intermediate_results_path : string
            Path to save intermediate results of the harmonization process
            
        site_indicator : a string
            Indicates the feature that differentiates different sites, default 'site'
        """        
        
        self.features_to_harmonize = [a.lower() for a in features_to_harmonize]
        self.intermediate_results_path = intermediate_results_path
        self.discrete_covariates = discrete_covariates
        self.continuous_covariates = continuous_covariates
        self.patient_identifier = patient_identifier.lower()
        self.site_indicator = site_indicator.lower()
        self.discrete_covariates_to_remove = [a.lower() for a in discrete_covariates_to_remove]
        self.continuous_covariates_to_remove = [a.lower() for a in continuous_covariates_to_remove]

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
        curr_path = os.getcwd()
        _disc = [f'as.factor({a}) + ' for a in self.discrete_covariates_to_remove]
        _disc = ''.join(_disc)[:-3]
        _cont = [f'{a} +' for a in self.continuous_covariates_to_remove]
        _cont = ''.join(_cont)[:-3]
        relief_r_driver = f"""
        rm(list = ls())
        options(repos = c(CRAN = "https://cran.r-project.org"))
        install.packages("matrixStats", dependencies = TRUE, quiet = TRUE)
        source('{curr_path}/combatPP.R') #as pluscombat
        source("{curr_path}/utils.R")
        library(matrixStats)
        
        fused_dat <- read.csv('{self.intermediate_results_path}/_tmp_combined_dataset.csv')
        cont_features = ({','.join(repr(x) for x in self.continuous_covariates)})
        disc_features = ({','.join(repr(x) for x in self.discrete_covariates)})
        cont_mat <- sapply(fused_dat[cont_features], function(x) as.numeric(unlist(x)))
        # Convert discrete features to categorical
        disc_mat <- sapply(fused_dat[disc_features], function(x) {{
        x <- as.numeric(unlist(x))
        as.factor(x)
        }})

        mod <- model.matrix(~ ., data = data.frame(cont_mat, disc_mat))
        #####################################################################################
        cont_features_to_remove = c({','.join(repr(x) for x in self.continuous_covariates_to_remove)})
        disc_features_to_remove = c({','.join(repr(x) for x in self.discrete_covariates_to_remove)})
        cont_mat_to_remove <- sapply(fused_dat[cont_features_to_remove], function(x) as.numeric(unlist(x)))
        # Convert discrete features to categorical
        disc_mat_to_remove <- sapply(fused_dat[disc_features_to_remove], function(x) {{
        x <- as.numeric(unlist(x))
        as.factor(x)
        }})




        # Conditional assignment for mod_to_remove based on the presence of covariates

        # Check if both cont_mat_to_remove and disc_mat_to_remove are NULL
        if (is.null(cont_mat_to_remove) && is.null(disc_mat_to_remove)) {{
        mod_to_remove <- NULL
        }} else {{
        # Initialize an empty list to store non-NULL matrices
        data_list <- list()
        
        # Add cont_mat_to_remove to the list if it's not NULL
        if (!is.null(cont_mat_to_remove)) {{
            data_list$cont_mat_to_remove <- cont_mat_to_remove
        }}
        
        # Add disc_mat_to_remove to the list if it's not NULL
        if (!is.null(disc_mat_to_remove)) {{
            data_list$disc_mat_to_remove <- disc_mat_to_remove
        }}
        
        # Combine the non-NULL matrices into a data frame
        combined_data <- do.call(cbind, data_list)
        
        # Create the model matrix using the combined data
        mod_to_remove <- model.matrix(~ ., data = as.data.frame(combined_data))
        }}




        #mod_to_remove <- model.matrix(~ ., data = data.frame(cont_mat_to_remove, disc_mat_to_remove))
        
        #####################################################################################
        batchvector <- c(fused_dat['{self.site_indicator}'])
        batchvector <- as.numeric(unlist(batchvector))

        ta <- t(fused_dat) 
        data.harmonized <-combatPP(dat=ta, PC= mod_to_remove, mod=mod, batch=batchvector) # need to add mod=mod
        new_df <- data.harmonized$dat.combat
        rollback <- t(new_df)
        write.csv(rollback, "{self.intermediate_results_path}/plus_harmonized_all.csv")
        """
        
        # all_togetherF, ftF, btF, feature_dictF, len1, len2, len3, len4, len5 = self._prep_for_neurocombat_5way([_d.data[self.features_to_harmonize + ['participant_id','sex','age']] for _d in mri_datasets])
        all_together = pd.concat([_d.data[self.features_to_harmonize+self.discrete_covariates+self.continuous_covariates+[self.site_indicator] + self.continuous_covariates_to_remove + self.discrete_covariates_to_remove] for _d in mri_datasets])
        all_together.to_csv(f'{self.intermediate_results_path}/_tmp_combined_dataset.csv')
        r = robjects.r
        r(relief_r_driver)
        bottom = pd.read_csv(f'{self.intermediate_results_path}/plus_harmonized_all.csv', index_col=0)#.reset_index(drop=False).drop(['index'],axis=1)#.rename(columns={"index": "char"})
        all_together[self.features_to_harmonize] = bottom[self.features_to_harmonize]
        for _ds in mri_datasets:
            ds_opn_harmonized = all_together[all_together['site'] == _ds.site_id]
            _ds.data[self.features_to_harmonize] = ds_opn_harmonized[self.features_to_harmonize].copy()
        
        return mri_datasets



class PredictBrainAge:
    
    def __init__(self, 
        model_name,
        model_file_name,
        model,
        datasets,
        datasets_validation,
        features,
        target,
        cat_category='sex',
        cont_category='age',
        n_bins=4,
        splits=5,
        test_size_p=0.2,
        random_state=42,
        ):
        
            self.model_name = model_name
            self.model_file_name = model_file_name
            self.model = model
            self.datasets = datasets
            self.datasets_validation = datasets_validation
            self.data = pd.concat([_d.data for _d in datasets])
            self.data_validation = pd.concat([_d.data for _d in datasets_validation]) if datasets_validation is not None else None
            self.features = features
            self.target = target
            self.cat_category = cat_category
            self.cont_category = cont_category
            self.splits = splits
            self.test_size_p = test_size_p
            self.random_state = random_state
            self.n_bins = n_bins
            
        
    def bin_dataset(self, ds, column, num_bins=4):
        
        ds[f'binned'] = pd.qcut(ds[column], num_bins, labels=False, duplicates='drop')

    def predict(self):
        if self.test_size_p > 1 / self.splits:
            warnings.warn("Potential resampling issue: test_size_p is too large.")
        
        self.bin_dataset(self.data,self.cont_category, num_bins=self.n_bins)  # Assuming bin_dataset exists
        self.data['fuse_bin'] = pd.factorize(
                self.data[self.cat_category].astype(str) + '_' + self.data['binned'].astype(str)
            )[0]
        
        if self.datasets_validation is not None:
            self.bin_dataset(self.data_validation,self.cont_category, num_bins=self.n_bins)  # Assuming bin_dataset exists
            self.data_validation['fuse_bin'] = pd.factorize(
                    self.data_validation[self.cat_category].astype(str) + '_' + self.data_validation['binned'].astype(str)
                )[0]

        
        
        sss = StratifiedShuffleSplit(n_splits=self.splits, test_size=self.test_size_p, random_state=self.random_state)

        all_metrics = []
        all_metrics_val = []
        all_predictions = []
        all_predictions_val = []
        models = []
        X = self.data[self.features]
        y = self.data[self.target]
        
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns = X.columns)
        
        if self.data_validation is not None:
            X_val = self.data_validation[self.features]
            y_val = self.data_validation[self.target]
            X_val = pd.DataFrame(StandardScaler().fit_transform(X_val), columns = X_val.columns)
        
        for i, (train_index, test_index) in enumerate(sss.split(self.data, self.data['fuse_bin'])):
            X_train = X.values[train_index]
            y_train = y.values[train_index]
            X_test = X.values[test_index]
            y_test = y.values[test_index]            

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_pred_val = self.model.predict(X_val) if self.data_validation is not None else None
            metrics_data = {
                'algorithm': f'{self.model_name}-{i}',
                'fold': i,
                'file_name': f'{self.model_file_name}.{i}',
                'explained_variance': metrics.explained_variance_score(y_test, y_pred),
                'max_error': metrics.max_error(y_test, y_pred),
                'mean_absolute_error': metrics.mean_absolute_error(y_test, y_pred),
                'mean_squared_error': metrics.mean_squared_error(y_test, y_pred),
                'mean_squared_log_error': metrics.mean_squared_log_error(y_test, y_pred) if all(y_test > 0) and all(y_pred > 0) else None,
                'median_absolute_error': metrics.median_absolute_error(y_test, y_pred),
                'r2': metrics.r2_score(y_test, y_pred),
                'mean_poisson_deviance': metrics.mean_poisson_deviance(y_test, y_pred) if all(y_test >= 0) and all(y_pred >= 0) else None,
                'mean_gamma_deviance': metrics.mean_gamma_deviance(y_test, y_pred) if all(y_test > 0) and all(y_pred > 0) else None,
                'mean_tweedie_deviance': metrics.mean_tweedie_deviance(y_test, y_pred),
                'd2_tweedie_score': metrics.d2_tweedie_score(y_test, y_pred), # Added d2 tweedie score
                'mean_absolute_percentage_error': metrics.mean_absolute_percentage_error(y_test, y_pred), # Added MAPE
            }
            metric_data_val = { 
                'algorithm': f'{self.model_name}-{i}',
                'fold': i,
                'file_name': f'{self.model_file_name}.{i}',
                'explained_variance': metrics.explained_variance_score(y_val, y_pred_val),
                'max_error': metrics.max_error(y_val, y_pred_val),
                'mean_absolute_error': metrics.mean_absolute_error(y_val, y_pred_val),
                'mean_squared_error': metrics.mean_squared_error(y_val, y_pred_val),
                'mean_squared_log_error': metrics.mean_squared_log_error(y_val, y_pred_val) if all(y_val > 0) and all(y_pred_val > 0) else None,
                'median_absolute_error': metrics.median_absolute_error(y_val, y_pred_val),
                'r2': metrics.r2_score(y_val, y_pred_val),
                'mean_poisson_deviance': metrics.mean_poisson_deviance(y_val, y_pred_val) if all(y_val >= 0) and all(y_pred_val >= 0) else None,
                'mean_gamma_deviance': metrics.mean_gamma_deviance(y_val, y_pred_val) if all(y_val > 0) and all(y_pred_val > 0) else None,
                'mean_tweedie_deviance': metrics.mean_tweedie_deviance(y_val, y_pred_val),
                'd2_tweedie_score': metrics.d2_tweedie_score(y_val, y_pred_val), # Added d2 tweedie score
                'mean_absolute_percentage_error': metrics.mean_absolute_percentage_error(y_val, y_pred_val), # Added MAPE
            } if self.data_validation is not None else None

            all_metrics.append(metrics_data)
            all_metrics_val.append(metric_data_val)
            predictions_data = pd.DataFrame({'y_test': y_test.flatten(), 'y_pred': y_pred.flatten()})
            predictions_data_val = pd.DataFrame({'y_test': y_val.values.flatten(), 'y_pred': y_pred_val.flatten()}) if self.data_validation is not None else None
            all_predictions.append(predictions_data)
            all_predictions_val.append(predictions_data_val) if self.data_validation is not None else None

            models.append((self.model, X.values[train_index][:, 0]))

        metrics_df = pd.DataFrame(all_metrics)
        metrics_df_val = pd.DataFrame(all_metrics_val) if self.data_validation is not None else None
        predictions_df = pd.concat(all_predictions)
        predictions_df_val = pd.concat(all_predictions_val) if self.data_validation is not None else None

        return metrics_df,metrics_df_val, predictions_df,predictions_df_val, models
    
