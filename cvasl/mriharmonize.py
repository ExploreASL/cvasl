# %%

import os
import sys
import pandas as pd
import numpy as np
import patsy
import cvasl.vendor.covbat.covbat as covbat
import cvasl.vendor.comscan.neurocombat as autocombat
import cvasl.vendor.neurocombat.neurocombat as neurocombat
import cvasl.vendor.open_nested_combat.nest as nest
from neuroHarmonize import harmonizationLearn


class MRIdataset:
    def __init__(self, path, site_id, cat_features_to_encode = [], ICV = False, decade = False, features_to_drop = ['m0','id'], features_to_bin = [], binning_method = 'equal_width', num_bins = 10, bin_labels = None):
        self.data = pd.read_csv(path)
        self.site_id = site_id
        self.data['Site'] = self.site_id
        self.feature_mappings = {}
        self.reverse_mappings = {}
        self.icv = ICV
        self.decade = decade
        self.features_to_drop = features_to_drop
        self.fetures_to_bin = features_to_bin
        self.binning_method  = binning_method
        self.num_bins = num_bins
        self.bin_labels = bin_labels
        self.cat_features_to_encode = cat_features_to_encode

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
        
        df_binned = self.data.copy()  # Create a copy to avoid modifying the original DataFrame

        for feature in features:
            if binning_method == 'equal_width':
                df_binned[feature + '_binned'], bins = pd.cut(self.data[feature], bins=num_bins, labels=labels, retbins=True, duplicates='drop')
            elif binning_method == 'equal_frequency':
                df_binned[feature + '_binned'], bins = pd.qcut(self.data[feature], q=num_bins, labels=labels, retbins=True, duplicates='drop')
            else:
                return None  # Handle invalid binning methods

        self.data = df_binned

    def encode_categorical_features(self):
        for feature in features_to_map:
            if feature in self.data.columns:
                unique_values = self.data[feature].unique()
                mapping = {value: i for i, value in enumerate(unique_values)}
                self.feature_mappings[feature] = mapping
                self.reverse_mappings[feature] = {v: k for k, v in mapping.items()}
                self.data[feature] = self.data[feature].map(mapping)

    def reverse_encode_categorical_features(self):
        for feature, mapping in self.reverse_mappings.items():
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].map(mapping)
    
    def addICVfeatures(self):
        self.data['icv'] = self.data['gm_vol'] / self.data['gm_icvratio']
    def addDecadefeatures(self):
        self.data['decade']=(self.data['age']/10).round()
        self.data = self.data.sort_values(by='age')
        self.data.reset_index(inplace=True)

    
    def dropFeatures(self):
        self.data = self.data.drop(self.features_to_drop, axis=1)

    def preprocess(self):
        # Common preprocessing steps
        self.data.columns = self.data.columns.str.lower()
        if self.features_to_drop:
            self.dropFeatures()
        
        if self.decade:
            self.addDecadefeatures()
        if self.icv:
            self.addICVfeatures()

        if self.cat_features_to_encode:
            self.encode_categorical_features()
            
        if self.fetures_to_bin:
            self.generalized_binning()


class EDISdataset(MRIdataset):
    def __init__(self, path, site_id, cat_features_to_encode=[], ICV=False, decade = False , 
                 features_to_drop=['m0', 'id'], features_to_bin=[], 
                 binning_method='equal_width', num_bins=10, bin_labels=None):
        super().__init__(path, site_id, cat_features_to_encode, ICV, decade,
                         features_to_drop, features_to_bin, binning_method, 
                         num_bins, bin_labels)
        self.preprocess()

class HELIUSdataset(MRIdataset):
    def __init__(self, path, site_id, cat_features_to_encode=[], ICV=False, decade = False , 
                 features_to_drop=['m0', 'id'], features_to_bin=[], 
                 binning_method='equal_width', num_bins=10, bin_labels=None):
        super().__init__(path, site_id, cat_features_to_encode, ICV,  decade,
                         features_to_drop, features_to_bin, binning_method, 
                         num_bins, bin_labels)
        self.preprocess()

    def preprocess(self):
        super().preprocess()
        # Specific preprocessing for HELIUS
        self.data.loc[self.data['participant_id'] == 'sub-153852_1', 'participant_id'] = 'sub-153852_1H'

class SABREdataset(MRIdataset):
    def __init__(self, path, site_id, cat_features_to_encode=[], ICV=False, decade = False , 
                 features_to_drop=['m0', 'id'], features_to_bin=[], 
                 binning_method='equal_width', num_bins=10, bin_labels=None):
        super().__init__(path, site_id, cat_features_to_encode, ICV,  decade,
                         features_to_drop, features_to_bin, binning_method, 
                         num_bins, bin_labels)
        self.preprocess()

class StrokeMRIdataset(MRIdataset):
    def __init__(self, path, site_id, cat_features_to_encode=[], ICV=False, decade = False, 
                 features_to_drop=['m0', 'id'], features_to_bin=[], 
                 binning_method='equal_width', num_bins=10, bin_labels=None):
        super().__init__(path, site_id, cat_features_to_encode, ICV,  decade,
                         features_to_drop, features_to_bin, binning_method, 
                         num_bins, bin_labels)
        self.preprocess()

class TOPdataset(MRIdataset):
    def __init__(self, path, site_id, cat_features_to_encode=[], ICV=False, decade = False, 
                 features_to_drop=['m0', 'id'], features_to_bin=[], 
                 binning_method='equal_width', num_bins=10, bin_labels=None):
        super().__init__(path, site_id, cat_features_to_encode, ICV,  decade,
                         features_to_drop, features_to_bin, binning_method, 
                         num_bins, bin_labels)
        self.preprocess()

class Insight46dataset(MRIdataset):
    def __init__(self, path, site_id, cat_features_to_encode=[], ICV=False, decade = False, 
                 features_to_drop=['m0', 'id'], features_to_bin=[], 
                 binning_method='equal_width', num_bins=10, bin_labels=None):
        super().__init__(path, site_id, cat_features_to_encode, ICV, decade,
                         features_to_drop, features_to_bin, binning_method, 
                         num_bins, bin_labels)
        self.preprocess()

class HarmNeuroHarmonize:
    def __init__(self, features_to_map, unharm_features, features_to_harmonize, covariates):
        self.features_to_map = features_to_map
        self.unharm_features = unharm_features
        self.features_to_harmonize = features_to_harmonize
        self.covariates = covariates

    def harmonize(self, mri_datasets):

        # Apply feature mapping
        for dataset in mri_datasets:
            dataset.encode_categorical_features()

        # Combine all datasets
        all_data = pd.concat([dataset.data for dataset in mri_datasets])

        # Prepare data for harmonization
        features_data = all_data[self.features_to_harmonize]
        covariates_data = all_data[self.covariates]
        covariates_data = covariates_data.rename(columns={'site': 'SITE'})

        # Perform harmonization
        _, harmonized_data = harmonizationLearn(np.array(features_data), covariates_data)

        # Create harmonized dataframe
        harmonized_df = pd.DataFrame(harmonized_data, columns=self.features_to_harmonize)
        harmonized_df = pd.concat([harmonized_df, covariates_data.reset_index(drop=True)], axis=1)
        harmonized_df = pd.concat([harmonized_df, all_data[self.unharm_features].reset_index(drop=True)], axis=1)

        # Reorder columns
        cols_to_front = ['participant_id', 'age', 'sex']
        for col in reversed(cols_to_front):
            harmonized_df.insert(0, col, harmonized_df.pop(col))
        for _d in mri_datasets:
            _d.data = harmonized_df[harmonized_df['SITE'] == _d.site_id]
        return mri_datasets



class HarmAutoCombat:
    def __init__(self, features_to_harmonize, sites, discrete_covariates=None, continuous_covariates=None, unharm_features=None):
        """
        Initializes the HarmAutoCombat harmonization class.

        Args:
            features_to_harmonize (list): List of features to be harmonized.
            sites (list): List containing the name of the site column.  Should be a list even if only one site column.
            discrete_covariates (list, optional): List of discrete covariates. Defaults to None.
            continuous_covariates (list, optional): List of continuous covariates. Defaults to None.
            unharm_features (list, optional): List of features *not* to be harmonized but to be included in the final output. Defaults to None.
        """
        self.features_to_harmonize = features_to_harmonize
        self.sites = sites
        self.discrete_covariates = discrete_covariates if discrete_covariates is not None else []
        self.continuous_covariates = continuous_covariates if continuous_covariates is not None else []
        self.unharm_features = unharm_features if unharm_features is not None else []


    def harmonize(self, mri_datasets):
        """
        Performs ComBat harmonization on the input data.

        Args:
            data (pd.DataFrame): Input DataFrame containing the features, site, and covariate information.

        Returns:
            pd.DataFrame: Harmonized DataFrame.  Returns original dataframe if `features_to_harmonize` is empty.
        """

        if not self.features_to_harmonize:  # Handle the case where no features need harmonization
            return data


        data = pd.concat([dataset.data for dataset in mri_datasets])

        # Instantiate ComBat object
        combat = autocombat.Combat(
            features=self.features_to_harmonize,
            sites=self.sites,
            discrete_covariates=self.discrete_covariates,
            continuous_covariates=self.continuous_covariates,
        )

        # Select relevant data for harmonization
        data_to_harmonize = data[self.features_to_harmonize + self.sites + self.discrete_covariates + self.continuous_covariates].copy()

        # Transform the data using ComBat
        harmonized_data = combat.fit(data_to_harmonize)
        harmonized_data = combat.transform(data_to_harmonize)


        # Create harmonized DataFrame
        harmonized_df = pd.DataFrame(harmonized_data, columns=self.features_to_harmonize)

        # Add back covariates and unharmonized features
        covariates = self.sites + self.discrete_covariates + self.continuous_covariates
        harmonized_df = pd.concat([harmonized_df, data_to_harmonize[covariates].reset_index(drop=True)], axis=1)
        
        if self.unharm_features:
            harmonized_df = pd.concat([harmonized_df, data[self.unharm_features].reset_index(drop=True)], axis=1)


        # Reorder columns to maintain original order as much as possible while putting harmonized features first
        original_order = list(data.columns)
        new_order = self.features_to_harmonize + [col for col in original_order if col not in self.features_to_harmonize]
        harmonized_df = harmonized_df[[col for col in new_order if col in harmonized_df.columns]]

        for i, dataset in enumerate(mri_datasets):
            site_value = dataset.site_id
            adjusted_data = harmonized_df[harmonized_df['site'] == site_value]
            dataset.data = adjusted_data
        
        return mri_datasets



class HarmCovbat:
    def __init__(self, features_to_harmonize, not_harmonized, covariates, site_col='site'):
        self.features_to_harmonize = [a.lower() for a in features_to_harmonize]
        self.not_harmonized = [a.lower() for a in not_harmonized]
        self.covariates = [a.lower() for a in covariates]
        self.site_col = site_col.lower()

    def harmonize(self, mri_datasets):
        # Separate features for harmonization and those to be kept unharmonized
        semi_features = []
        datasets_to_harmonize = []
        for dataset in mri_datasets:
            semi_features.append(dataset.data.drop(self.features_to_harmonize, axis=1))
            datasets_to_harmonize.append(dataset.data.drop(self.not_harmonized, axis=1))

        # Combine datasets for harmonization
        all_data = pd.concat(datasets_to_harmonize)

        # Prepare data for CovBat
        pheno_features = ['participant_id'] + self.covariates + [self.site_col]
        pheno = all_data[pheno_features]
        pheno = pheno.set_index('participant_id')

        data_to_harmonize = all_data.set_index('participant_id')
        data_to_harmonize = data_to_harmonize.drop(self.covariates, axis=1)
        data_to_harmonize = data_to_harmonize.T
        print(f"~ {' + '.join(self.covariates)}")
        mod_matrix = patsy.dmatrix(f"~ {' + '.join(self.covariates)}", pheno, return_type="dataframe")

        # Perform harmonization using CovBat
        harmonized_data = covbat.combat(data_to_harmonize, pheno[self.site_col], model=mod_matrix, numerical_covariates ="age")
        harmonized_data = harmonized_data[len(self.covariates):] # Remove estimated model parameters from the output

        # Combine harmonized data with other features
        harmonized_data = pd.concat([data_to_harmonize.head(len(self.covariates) + 1), harmonized_data]) # Add back the ID and model parameters
        harmonized_data = harmonized_data.T
        harmonized_data = harmonized_data.reset_index()

        # Split the harmonized data back into individual datasets
        harmonized_datasets = []
        for i, dataset in enumerate(mri_datasets):
            site_value = dataset.site_id
            adjusted_data = harmonized_data[harmonized_data['site'] == site_value]
            adjusted_data = adjusted_data.merge(semi_features[i], on='participant_id')
            #adjusted_data = adjusted_data.drop(self.site_col, axis=1)
            dataset.data = adjusted_data
            
        return mri_datasets

        return harmonized_datasets


class HarmNeuroCombat:
    def __init__(self, cat_features, cont_features, features_to_harmonize, not_harmonized, batch_col='site'):
        self.cat_features = cat_features
        self.cont_features = cont_features
        self.features_to_harmonize = [a.lower() for a in features_to_harmonize]
        self.not_harmonized = [a.lower() for a in not_harmonized]
        self.batch_col = batch_col

    def _prep_for_neurocombat(self, dataframes):
        all_together = pd.concat(
            [df.set_index('participant_id').T for df in dataframes],
            axis=1,
            join="inner",
        )
        features_only = all_together.drop(index=[f.lower() for f in self.features_to_harmonize if f.lower() in all_together.index]) #drop the features we don't want to harmonize
        feature_dict = {i: col for i, col in enumerate(features_only.T.columns)}
        ft = features_only.reset_index(drop=True).dropna()
        bt = all_together.reset_index(drop=True).dropna()
        lengths = [len(df) for df in dataframes]
        return all_together, ft, bt, feature_dict, lengths

    def _make_topper(self, bt, row_labels):
        topper = bt.head(len(row_labels)).rename_axis(None, axis="columns").reset_index(drop=False)
        topper = topper.rename(columns={"index": "char"})
        topper['char'] = row_labels
        return topper


    def harmonize(self, mri_datasets):
        # Separate features for harmonization and those to be kept unharmonized
        semi_features = []
        datasets_to_harmonize = []
        for dataset in mri_datasets:
            semi_features.append(dataset.data.drop(columns=[f for f in self.features_to_harmonize if f in dataset.data.columns], errors='ignore')) #ignore errors if the column is not there
            datasets_to_harmonize.append(dataset.data.drop(columns=[f for f in self.not_harmonized if f in dataset.data.columns], errors='ignore'))

        # Prepare data for NeuroCombat
        all_together, ft, bt, feature_dict, lengths = self._prep_for_neurocombat(datasets_to_harmonize)

        # Create covariates
        batch_ids = []
        for i, l in enumerate(lengths):
            batch_ids.extend([i + 1] * l) #start batch numbers at 1

        covars = {self.batch_col: batch_ids}
        for feature in self.cat_features + self.cont_features:
            feature_lower = feature.lower()
            if feature_lower in all_together.index:
                covars[feature] = all_together.loc[feature_lower,:].values.tolist()
        covars = pd.DataFrame(covars)

        # Convert data to numpy array for NeuroCombat
        data = ft.values


        # Harmonize data using NeuroCombat
        data_combat = neurocombat.neuroCombat(dat=data,
                                            covars=covars,
                                            batch_col=self.batch_col,
                                            continuous_cols=self.cont_features,
                                            categorical_cols=self.cat_features)["data"]

        # Convert harmonized data back to DataFrame
        neurocombat_df = pd.DataFrame(data_combat)


        # Reconstruct the full dataframe
        topper = self._make_topper(bt, self.features_to_harmonize)
        bottom = neurocombat_df.reset_index(drop=False)
        bottom = bottom.rename(columns={"index": "char"})
        bottom.columns = topper.columns #align columns with topper
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
            harmonized_data = harmonized_data.rename(feature_dict, axis='columns')
            harmonized_data = harmonized_data.reset_index().rename(columns={"index": "participant_id"})
            harmonized_data = harmonized_data.merge(semi_features[i], on="participant_id") #Merge back the unharmonized features
            harmonized_datasets.append(harmonized_data)
            start = end
        
        harmonized_data = pd.concat([_d for _d in harmonized_datasets])
        
        for i, dataset in enumerate(mri_datasets):
            site_value = dataset.site_id
            adjusted_data = harmonized_data[harmonized_data['site_y'] == site_value]
            adjusted_data = adjusted_data.merge(semi_features[i], on='participant_id')
            #adjusted_data = adjusted_data.drop(self.site_col, axis=1)
            dataset.data = adjusted_data
        
        return mri_datasets

class HarmNestedComBat:
    def __init__(self, batch_list, categorical_cols, continuous_cols):
        self.batch_list = batch_list
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols

    def harmonize(self, mri_datasets):
        site_n = 0
        dfs_simplified = {}  # Store simplified dataframes
        for dataset in mri_datasets:
            df = dataset.data.copy()
            df['site'] = site_n
            site_n += 1
            dfs_simplified[dataset.site_id] = df[['participant_id', 'age', 'sex','site']].copy()


        batch_testing_df = pd.concat(list(dfs_simplified.values()), ignore_index=True)
        
        all_data = pd.concat([dataset.data for dataset in mri_datasets])

        data_testing_df = all_data.drop(columns=['age','sex'], errors='ignore').dropna().reset_index(drop=True)  # Handle potential missing columns
        data_testing_df = data_testing_df.merge(batch_testing_df['participant_id'], on='participant_id').drop_duplicates()

        dat_testing = data_testing_df.iloc[:, 1:].T.apply(pd.to_numeric)
        caseno_testing = data_testing_df['participant_id']
        covars_testing = batch_testing_df.drop('participant_id', axis=1)

        # Encode categorical variables using get_dummies for efficiency
        covars_testing_cat = pd.get_dummies(covars_testing[self.categorical_cols], columns=self.categorical_cols, drop_first=True, dummy_na=False)
        covars_testing_final = pd.concat([covars_testing_cat, covars_testing[self.continuous_cols]], axis=1)

        filepath2 = 'Testing/OPPNComBat/ResultTesting'
        os.makedirs(filepath2, exist_ok=True)

        gmm_testing_df = nest.GMMSplit(dat_testing, caseno_testing, filepath2)
        gmm_testing_df_merge = batch_testing_df.merge(gmm_testing_df, right_on='Patient', left_on='participant_id')
        gmm_testing_df_merge['GMM'] = gmm_testing_df_merge['Grouping']
        covars_testing_final = gmm_testing_df_merge.drop(['participant_id', 'Patient', 'Grouping'], axis=1)

        # Ensure GMM is in categorical columns
        if 'GMM' not in self.categorical_cols:
            self.categorical_cols.append('GMM')

        output_testing_df = nest.OPNestedComBat(dat_testing, covars_testing_final, self.batch_list, filepath2,
                                               categorical_cols=self.categorical_cols,
                                               continuous_cols=self.continuous_cols)

        write_testing_df = pd.concat([caseno_testing, output_testing_df], axis=1)

        complete_harmonised = pd.concat([write_testing_df, covars_testing_final], axis=1)

        # Split the harmonized data back into MRIdataset objects
        harmonized_datasets = []
        for dataset in mri_datasets:
            dataset_df = complete_harmonised[complete_harmonised['site'] == dataset.site_id].drop('site', axis=1)
            harmonized_dataset = MRIdataset(None, dataset.site_id) # Create a new MRIdataset object
            harmonized_dataset.data = dataset_df
            harmonized_datasets.append(harmonized_dataset)


        return harmonized_datasets



# Usage example:
features_to_map = ['readout', 'labelling']
edis = EDISdataset('../new_data/TrainingDataComplete_EDIS.csv', site_id=0, ICV = True, cat_features_to_encode=features_to_map)
helius = HELIUSdataset('../new_data/TrainingDataComplete_HELIUS.csv', site_id=1, ICV = True, cat_features_to_encode=features_to_map)
sabre = SABREdataset('../new_data/TrainingDataComplete_SABRE.csv', site_id=2, ICV = True, cat_features_to_encode=features_to_map)
stroke_mri = StrokeMRIdataset('../new_data/TrainingDataComplete_StrokeMRI.csv', site_id=3, ICV = True, cat_features_to_encode=features_to_map)
top = TOPdataset('../new_data/TrainingDataComplete_TOP.csv', site_id=3, ICV = True, cat_features_to_encode=features_to_map)
insight46 = Insight46dataset('../new_data/TrainingDataComplete_Insight46.csv', site_id=4, ICV = True, cat_features_to_encode=features_to_map)


unharm_features = ['participant_id', 'gm_vol', 'wm_vol', 'csf_vol', 'gm_icvratio', 'gmwm_icvratio', 'wmhvol_wmvol', 'wmh_count']
features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                         'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
covariates = ['age', 'sex', 'site', 'icv']

harmonizer = HarmNeuroHarmonize(features_to_map, unharm_features, features_to_harmonize, covariates)
harmonized_data = harmonizer.harmonize([edis, helius, sabre, stroke_mri, top, insight46])












#covbat
features_to_map = ['readout', 'labelling']
edis = EDISdataset('../new_data/TrainingDataComplete_EDIS.csv', site_id=0, ICV = True, cat_features_to_encode=features_to_map)
helius = HELIUSdataset('../new_data/TrainingDataComplete_HELIUS.csv', site_id=1, ICV = True, cat_features_to_encode=features_to_map)
sabre = SABREdataset('../new_data/TrainingDataComplete_SABRE.csv', site_id=2, ICV = True, cat_features_to_encode=features_to_map)
stroke_mri = StrokeMRIdataset('../new_data/TrainingDataComplete_StrokeMRI.csv', site_id=3, ICV = True, cat_features_to_encode=features_to_map)
top = TOPdataset('../new_data/TrainingDataComplete_TOP.csv', site_id=3, ICV = True, cat_features_to_encode=features_to_map)
insight46 = Insight46dataset('../new_data/TrainingDataComplete_Insight46.csv', site_id=4, ICV = True, cat_features_to_encode=features_to_map)


to_be_harmonized_or_covar = [
    'Age', 'Sex', 'ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV','DeepWM_B_CoV',
     'ACA_B_CBF', 'MCA_B_CBF', 'PCA_B_CBF', 'LD', 'PLD','DeepWM_B_CBF',
       'Labelling', 'Readout', 'TotalGM_B_CoV',
       'TotalGM_B_CBF',
]
not_harmonized= ['GM_vol', 'WM_vol', 'CSF_vol','GM_ICVRatio', 'GMWM_ICVRatio', 'WMHvol_WMvol', 'WMH_count',
                'LD', 'PLD', 'Labelling',
       'Readout','DeepWM_B_CoV','DeepWM_B_CBF',]

#pheno_features = ['participant_id','Age', 'Sex', 'Site']
pheno_features = ['Age', 'Sex']

harmonizer = HarmCovbat(to_be_harmonized_or_covar,not_harmonized,pheno_features)
harmonized_data = harmonizer.harmonize([edis, helius, sabre, stroke_mri, top, insight46])














features_to_map = ['readout', 'labelling']
edis = EDISdataset('../new_data/TrainingDataComplete_EDIS.csv', site_id=0, ICV = True, cat_features_to_encode=features_to_map)
helius = HELIUSdataset('../new_data/TrainingDataComplete_HELIUS.csv', site_id=1, ICV = True, cat_features_to_encode=features_to_map)
sabre = SABREdataset('../new_data/TrainingDataComplete_SABRE.csv', site_id=2, ICV = True, cat_features_to_encode=features_to_map)
stroke_mri = StrokeMRIdataset('../new_data/TrainingDataComplete_StrokeMRI.csv', site_id=3, ICV = True, cat_features_to_encode=features_to_map)
top = TOPdataset('../new_data/TrainingDataComplete_TOP.csv', site_id=3, ICV = True, cat_features_to_encode=features_to_map)
insight46 = Insight46dataset('../new_data/TrainingDataComplete_Insight46.csv', site_id=4, ICV = True, cat_features_to_encode=features_to_map)

to_be_harmonized_or_covar = [
    'Age', 'Sex', 'DeepWM_B_CoV', 'ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV', 'TotalGM_B_CoV', 'DeepWM_B_CBF', 
    'ACA_B_CBF', 'MCA_B_CBF', 'PCA_B_CBF', 'TotalGM_B_CBF', 'DeepWM_B_CoV', 'DeepWM_B_CBF',]

not_harmonized= [
    'GM_vol', 'WM_vol', 'CSF_vol', 'GM_ICVRatio', 'GMWM_ICVRatio', 'WMHvol_WMvol', 'WMH_count', 'DeepWM_B_CoV', 'DeepWM_B_CBF',]


harmonizer = HarmNeuroCombat(['age'],['sex'],to_be_harmonized_or_covar,not_harmonized)
harmonized_data = harmonizer.harmonize([edis, helius, sabre, stroke_mri, top, insight46])















#autocombat
features_to_map = ['readout', 'labelling']
edis = EDISdataset('../new_data/TrainingDataComplete_EDIS.csv', site_id=0,decade = True, ICV = True, cat_features_to_encode=features_to_map)
helius = HELIUSdataset('../new_data/TrainingDataComplete_HELIUS.csv', site_id=1,decade = True, ICV = True, cat_features_to_encode=features_to_map)
sabre = SABREdataset('../new_data/TrainingDataComplete_SABRE.csv', site_id=2,decade = True, ICV = True, cat_features_to_encode=features_to_map)
stroke_mri = StrokeMRIdataset('../new_data/TrainingDataComplete_StrokeMRI.csv', site_id=3,decade = True, ICV = True, cat_features_to_encode=features_to_map)
top = TOPdataset('../new_data/TrainingDataComplete_TOP.csv', site_id=3,decade = True, ICV = True, cat_features_to_encode=features_to_map)
insight46 = Insight46dataset('../new_data/TrainingDataComplete_Insight46.csv', site_id=4,decade = True, ICV = True, cat_features_to_encode=features_to_map)

features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                         'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
unharm_features = ['participant_id', 'gm_vol', 'wm_vol', 'csf_vol', 
                   'gm_icvratio', 'gmwm_icvratio', 'wmhvol_wmvol', 'wmh_count']
discrete_covariates = ['sex']
continuous_covariates = ['decade']
harmonizer = HarmAutoCombat(features_to_harmonize,['site'],discrete_covariates,continuous_covariates,unharm_features)
harmonized_data = harmonizer.harmonize([edis, helius, sabre, stroke_mri, top, insight46])

# %%
