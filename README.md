<p align="center">
    <img style="width: 35%; height: 35%" src="cv_asl_svg.svg">
</p>

[![DOI](https://zenodo.org/badge/618300539.svg)](https://zenodo.org/badge/latestdoi/618300539)
[![PyPI- to be made, placeholder](https://img.shields.io/pypi/v/cvasl.svg)](https://pypi.python.org/pypi/cvasl/)
[![Anaconda](https://anaconda.org/brainspinner-org/cvasl/badges/version.svg)](https://anaconda.org/brainspinner-org/cvasl/badges/version.svg)
[![Sanity](https://github.com/brainspinner/cvasl/actions/workflows/on-commit.yml/badge.svg)](https://github.com/brainspinner/cvasl/actions/workflows/on-commit.yml)
[![Sanity](https://github.com/brainspinner/cvasl/actions/workflows/on-tag.yml/badge.svg)](https://github.com/brainspinner/cvasl/actions/workflows/on-tag.yml)

# CVASL - Brain MRI Analysis Library

**cvasl** is an open source collaborative Python library for analysis of brain MRIs, with a particular focus on arterial spin labeled sequences. This library supports ongoing research at the University of Amsterdam Medical Center on brain aging, but is built for the entire community of radiology researchers across university and academic medical centers worldwide.

## Supported Platforms

cvasl is not a pure Python package. If you want to run all functionalities, you will need R installed as well. 

## Installation

To install cvasl, you can use pip:

```bash
pip install cvasl
```

For developers or those who want the latest version, you can install directly from the repository:

```bash
git clone https://github.com/brainspinner/cvasl.git
cd cvasl
pip install -e .
```

## Project Structure

The repository is organized as follows:

- **cvasl/**: Core package modules
  - **dataset.py**: Contains the MRIdataset class for loading and preprocessing data
  - **harmonizers.py**: Implements various harmonization methods
  - **harmonizer_cli.py**: Command-line interface for harmonization
  - **prediction.py**: Brain age prediction functionality
  - **prediction_cli.py**: Command-line interface for brain age prediction
  
- **docs/**: Documentation files
- **tests/**: Test scripts and data

## Data Requirements

The cvasl package is designed to work with MRI datasets. For derived value datasets (which use measurements instead of images), you will need CSV or TSV files arranged in a specific format as outlined in [seperated_values_specifications.md](seperated_values_specifications.md).

When working with patient data, ensure you do not expose any identifying information if using the code for demonstrations or pull requests.

## Command-Line Interface

cvasl provides two primary command-line interfaces:

1. `harmonizer_cli.py`: For performing data harmonization
2. `prediction_cli.py`: For brain age prediction

These tools allow you to process MRI data without writing Python code.

## MRIdataset Class

The `MRIdataset` class in `cvasl.dataset` is designed to handle MRI datasets from different sites, preparing them for harmonization and analysis.

### Initialization Parameters

- `path` (str or list): Path to the CSV file or a list of paths for multiple files.
- `site_id` (int or str): Identifier for the data acquisition site.
- `patient_identifier` (str, optional): Column name for patient IDs. Defaults to `"participant_id"`.
- `cat_features_to_encode` (list, optional): List of categorical features to encode. Defaults to `None`.
- `ICV` (bool, optional): Whether to add Intracranial Volume (ICV) related features. Defaults to `False`.
- `decade` (bool, optional): Whether to add decade-related features based on age. Defaults to `False`.
- `features_to_drop` (list, optional): List of features to drop during preprocessing. Defaults to `["m0", "id"]`.
- `features_to_bin` (list, optional): List of features to bin. Defaults to `None`.
- `binning_method` (str, optional): Binning method to use; `"equal_width"` or `"equal_frequency"`. Defaults to `"equal_width"`.
- `num_bins` (int, optional): Number of bins for binning. Defaults to `10`.
- `bin_labels` (list, optional): Labels for bins. Defaults to `None`.

### Usage Example

```python
from cvasl.dataset import MRIdataset, encode_cat_features

# Initialize datasets from different sites
edis = MRIdataset(path='../data/EDIS_input.csv', site_id=0, decade=True, ICV=True, 
                  patient_identifier='participant_id', features_to_drop=["m0", "id"])
helius = MRIdataset(path='../data/HELIUS_input.csv', site_id=1, decade=True, ICV=True, 
                   patient_identifier='participant_id', features_to_drop=["m0", "id"])
# For datasets spanning multiple files
topmri = MRIdataset(path=['../data/TOP_input.csv','../data/StrokeMRI_input.csv'], 
                   site_id=3, decade=True, ICV=True, 
                   patient_identifier='participant_id', features_to_drop=["m0", "id"])

# Preprocess datasets
datasets = [edis, helius, topmri]
[d.preprocess() for d in datasets]

# Encode categorical features across datasets
features_to_map = ['readout', 'labelling', 'sex']
datasets = encode_cat_features(datasets, features_to_map)
```

## Harmonization Methods

cvasl offers multiple harmonization methods to reduce site-specific variance in MRI data. You can access these through the `harmonizer_cli.py` command-line tool.

### Running Harmonization via CLI

General command structure:

```bash
python harmonizer_cli.py --dataset_paths <dataset_paths> --site_ids <site_ids> --method <harmonization_method> [method_specific_options] [dataset_options]
```

Parameters:
- `--dataset_paths`: Comma-separated paths to dataset CSV files. For datasets with multiple files, use semicolons within a dataset and commas between datasets (e.g., `path1,path2,"path3;path4",path5`).
- `--site_ids`: Comma-separated site IDs corresponding to each dataset.
- `--method`: Harmonization method to use (options listed below).
- `[method_specific_options]`: Options specific to the chosen harmonization method.
- `[dataset_options]`: Options for dataset loading and preprocessing.

### Available Harmonization Methods

#### 1. NeuroHarmonize

- **Method Class**: `NeuroHarmonize`
- **CLI method value**: `neuroharmonize`
- **Method-specific Options**:
  - `--nh_features_to_harmonize`: Features to harmonize (comma-separated)
  - `--nh_covariates`: Covariates (comma-separated)
  - `--nh_smooth_terms`: Smooth terms (comma-separated, optional)
  - `--nh_site_indicator`: Site indicator column name
  - `--nh_empirical_bayes`: Use empirical Bayes (True/False)

**Example**:
```bash
python harmonizer_cli.py --dataset_paths ../data/EDIS_input.csv,../data/HELIUS_input.csv --site_ids 0,1 --method neuroharmonize --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --nh_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov --nh_covariates age,sex,icv,site --nh_site_indicator site
```

#### 2. Covbat

- **Method Class**: `Covbat`
- **CLI method value**: `covbat`
- **Method-specific Options**:
  - `--cb_features_to_harmonize`: Features to harmonize (comma-separated)
  - `--cb_covariates`: Covariates (comma-separated)
  - `--cb_site_indicator`: Site indicator column name
  - `--cb_patient_identifier`: Patient identifier column name
  - `--cb_numerical_covariates`: Numerical covariates (comma-separated)
  - `--cb_empirical_bayes`: Use empirical Bayes (True/False)

**Example**:
```bash
python harmonizer_cli.py --dataset_paths ../data/EDIS_input.csv,../data/HELIUS_input.csv --site_ids 0,1 --method covbat --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --cb_features_to_harmonize participant_id,site,age,sex,site,aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov --cb_covariates age,sex --cb_numerical_covariates age --cb_site_indicator site
```

#### 3. NeuroCombat

- **Method Class**: `NeuroCombat`
- **CLI method value**: `neurocombat`
- **Method-specific Options**:
  - `--nc_features_to_harmonize`: Features to harmonize (comma-separated)
  - `--nc_discrete_covariates`: Discrete covariates (comma-separated)
  - `--nc_continuous_covariates`: Continuous covariates (comma-separated)
  - `--nc_site_indicator`: Site indicator column name
  - `--nc_patient_identifier`: Patient identifier column name
  - `--nc_empirical_bayes`: Use empirical Bayes (True/False)
  - `--nc_mean_only`: Mean-only adjustment (True/False)
  - `--nc_parametric`: Parametric adjustment (True/False)

**Example**:
```bash
python harmonizer_cli.py --dataset_paths ../data/EDIS_input.csv,../data/HELIUS_input.csv --site_ids 0,1 --method neurocombat --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --nc_features_to_harmonize ACA_B_CoV,MCA_B_CoV,PCA_B_CoV,TotalGM_B_CoV --nc_discrete_covariates sex --nc_continuous_covariates age --nc_site_indicator site
```

#### 4. NestedComBat

- **Method Class**: `NestedComBat`
- **CLI method value**: `nestedcombat`
- **Method-specific Options**:
  - `--nest_features_to_harmonize`: Features to harmonize (comma-separated)
  - `--nest_batch_list_harmonisations`: Batch variables for nested ComBat (comma-separated)
  - `--nest_site_indicator`: Site indicator column name
  - `--nest_discrete_covariates`: Discrete covariates (comma-separated)
  - `--nest_continuous_covariates`: Continuous covariates (comma-separated)
  - `--nest_intermediate_results_path`: Path for intermediate results
  - `--nest_patient_identifier`: Patient identifier column name
  - `--nest_return_extended`: Return extended outputs (True/False)
  - `--nest_use_gmm`: Use Gaussian Mixture Model (True/False)

**Example**:
```bash
python harmonizer_cli.py --dataset_paths ../data/EDIS_input.csv,../data/HELIUS_input.csv --site_ids 0,1 --method nestedcombat --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --nest_features_to_harmonize ACA_B_CoV,MCA_B_CoV,PCA_B_CoV,TotalGM_B_CoV --nest_batch_list_harmonisations readout,ld,pld --nest_site_indicator site --nest_discrete_covariates sex --nest_continuous_covariates age --nest_use_gmm False
```

#### 5. Combat++

- **Method Class**: `CombatPlusPlus`
- **CLI method value**: `combat++`
- **Method-specific Options**:
  - `--compp_features_to_harmonize`: Features to harmonize (comma-separated)
  - `--compp_discrete_covariates`: Discrete covariates (comma-separated)
  - `--compp_continuous_covariates`: Continuous covariates (comma-separated)
  - `--compp_discrete_covariates_to_remove`: Discrete covariates to remove (comma-separated)
  - `--compp_continuous_covariates_to_remove`: Continuous covariates to remove (comma-separated)
  - `--compp_site_indicator`: Site indicator column name
  - `--compp_patient_identifier`: Patient identifier column name
  - `--compp_intermediate_results_path`: Path for intermediate results

**Example**:
```bash
python harmonizer_cli.py --dataset_paths ../data/EDIS_input.csv,../data/HELIUS_input.csv --site_ids 0,1 --method combat++ --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --compp_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov --compp_discrete_covariates sex --compp_continuous_covariates age --compp_discrete_covariates_to_remove labelling --compp_continuous_covariates_to_remove ld --compp_site_indicator site
```

#### 6. ComscanNeuroHarmonize

- **Method Class**: `ComscanNeuroCombat`
- **CLI method value**: `comscanneuroharmonize`
- **Method-specific Options**:
  - `--csnh_features_to_harmonize`: Features to harmonize (comma-separated)
  - `--csnh_discrete_covariates`: Discrete covariates (comma-separated)
  - `--csnh_continuous_covariates`: Continuous covariates (comma-separated)
  - `--csnh_site_indicator`: Site indicator column name

**Example**:
```bash
python harmonizer_cli.py --dataset_paths ../data/EDIS_input.csv,../data/HELIUS_input.csv --site_ids 0,1 --method comscanneuroharmonize --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --csnh_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov --csnh_discrete_covariates sex --csnh_continuous_covariates decade --csnh_site_indicator site
```

#### 7. AutoComBat

- **Method Class**: `AutoCombat`
- **CLI method value**: `autocombat`
- **Method-specific Options**:
  - `--ac_features_to_harmonize`: Features to harmonize (comma-separated)
  - `--ac_data_subset`: Data subset features (comma-separated)
  - `--ac_discrete_covariates`: Discrete covariates (comma-separated)
  - `--ac_continuous_covariates`: Continuous covariates (comma-separated)
  - `--ac_site_indicator`: Site indicator column name(s), comma-separated if multiple
  - `--ac_discrete_cluster_features`: Discrete cluster features (comma-separated)
  - `--ac_continuous_cluster_features`: Continuous cluster features (comma-separated)
  - `--ac_metric`: Metric for cluster optimization (`distortion`, `silhouette`, `calinski_harabasz`)
  - `--ac_features_reduction`: Feature reduction method (`pca`, `umap`, `None`)
  - `--ac_feature_reduction_dimensions`: Feature reduction dimensions (int)
  - `--ac_empirical_bayes`: Use empirical Bayes (True/False)

**Example**:
```bash
python harmonizer_cli.py --dataset_paths ../data/EDIS_input.csv,../data/HELIUS_input.csv --site_ids 0,1 --method autocombat --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --ac_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov --ac_data_subset aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,site,readout,labelling,pld,ld,sex,age --ac_discrete_covariates sex --ac_continuous_covariates age --ac_site_indicator site,readout,pld,ld --ac_discrete_cluster_features site,readout --ac_continuous_cluster_features pld,ld
```

#### 8. RELIEF

- **Method Class**: `RELIEF`
- **CLI method value**: `relief`
- **Method-specific Options**:
  - `--relief_features_to_harmonize`: Features to harmonize (comma-separated)
  - `--relief_covariates`: Covariates (comma-separated)
  - `--relief_patient_identifier`: Patient identifier column name
  - `--relief_intermediate_results_path`: Path for intermediate results

**Example**:
```bash
python harmonizer_cli.py --dataset_paths ../data/EDIS_input.csv,../data/HELIUS_input.csv --site_ids 0,1 --method relief --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --relief_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov --relief_covariates sex,age --relief_patient_identifier participant_id
```

### Important Notes for Harmonization

- **Multiple Files**: For datasets with multiple paths (like TOPMRI in examples), use semicolons (`;`) to separate paths within a dataset entry, and commas (`,`) to separate different datasets.
- **Path Adjustment**: Replace example paths with your actual data file paths.
- **Parameter Tuning**: Adjust harmonization parameters based on your dataset and harmonization goals.
- **R Requirement**: Methods like RELIEF and Combat++ require R to be installed with necessary packages (denoiseR, RcppCNPy, matrixStats).
- **Output Files**: Harmonized datasets are saved as new CSV files in the same directory as input datasets, with filenames appended with `output_<harmonization_method>`.

## Prediction CLI

The `prediction_cli.py` script provides a command-line interface for brain age prediction based on MRI data.

### General Usage

```bash
python prediction_cli.py --training_dataset_paths <training_paths> --training_site_ids <training_site_ids> --testing_dataset_paths <testing_paths> --testing_site_ids <testing_site_ids> --model <model_type> [model_specific_options] [dataset_options]
```

Parameters:
- `--training_dataset_paths`: Comma-separated paths to training dataset CSV files
- `--training_site_ids`: Comma-separated site IDs for training datasets
- `--testing_dataset_paths`: Comma-separated paths to testing dataset CSV files
- `--testing_site_ids`: Comma-separated site IDs for testing datasets
- `--model`: Type of prediction model to use
- `[model_specific_options]`: Options specific to the chosen model
- `[dataset_options]`: Options for dataset loading and preprocessing

### Available Models

#### Linear Regression

```bash
python prediction_cli.py --training_dataset_paths ../data/EDIS_input.csv --training_site_ids 0 --testing_dataset_paths ../data/HELIUS_input.csv --testing_site_ids 1 --model linear_regression --features ACA_B_CoV,MCA_B_CoV,PCA_B_CoV,TotalGM_B_CoV --target age --output_file brain_age_predictions.csv
```

#### Random Forest

```bash
python prediction_cli.py --training_dataset_paths ../data/EDIS_input.csv --training_site_ids 0 --testing_dataset_paths ../data/HELIUS_input.csv --testing_site_ids 1 --model random_forest --features ACA_B_CoV,MCA_B_CoV,PCA_B_CoV,TotalGM_B_CoV --target age --n_estimators 100 --max_depth 10 --output_file brain_age_predictions.csv
```

#### Support Vector Regression

```bash
python prediction_cli.py --training_dataset_paths ../data/EDIS_input.csv --training_site_ids 0 --testing_dataset_paths ../data/HELIUS_input.csv --testing_site_ids 1 --model svr --features ACA_B_CoV,MCA_B_CoV,PCA_B_CoV,TotalGM_B_CoV --target age --kernel rbf --C 1.0 --epsilon 0.1 --output_file brain_age_predictions.csv
```

#### Neural Network

```bash
python prediction_cli.py --training_dataset_paths ../data/EDIS_input.csv --training_site_ids 0 --testing_dataset_paths ../data/HELIUS_input.csv --testing_site_ids 1 --model neural_network --features ACA_B_CoV,MCA_B_CoV,PCA_B_CoV,TotalGM_B_CoV --target age --hidden_layer_sizes 100,50 --activation relu --solver adam --output_file brain_age_predictions.csv
```

## License

Licensed under the terms specified in the LICENSE file.