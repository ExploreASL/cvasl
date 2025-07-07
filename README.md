<p align="center">
    <img style="width: 35%; height: 35%" src="cv_asl_svg.svg">
</p>

[![DOI](https://zenodo.org/badge/618300539.svg)](https://zenodo.org/badge/latestdoi/618300539)
[![PyPI- to be made, placeholder](https://img.shields.io/pypi/v/cvasl.svg)](https://pypi.python.org/pypi/cvasl/)
[![Sanity](https://github.com/ExploreASL/cvasl/actions/workflows/on-commit.yml/badge.svg)](https://github.com/ExploreASL/cvasl/actions/workflows/on-commit.yml)
[![Citation](https://img.shields.io/badge/Cite%20as-cvasl-blue)](https://github.com/ExploreASL/cvasl/#citation)


**cvasl** is an open source collaborative python library for analysis
of brain MRIs. Many functions relate to arterial spin labeled sequences.



This library
supports the ongoing research at University of Amsterdam Medical Center on brain ageing, but
is being buit for the entire community of radiology researchers across all university and academic medical centers and beyond.


## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite it. You can find the citation information by clicking the "Cite this repository" button in the sidebar on the right.

```bibtex
@software{Amiri_cvasl_2025,
author = {Amiri, Saba and Kok, Peter and Moore, Candace Makeda and Crocioni, Giulia and Dijsselhof, Mathijs and Mutsaerts, Henk JMM and Petr, Jan and Bodor, Dani},
license = {Apache-2.0},
month = jul,
title = {{cvasl}},
url = {https://github.com/ExploreASL/cvasl},
version = {1.1.0},
year = {2025}
}
```

# Command-Line Interface

You can preprocess, train and use models, and perform harmonization using the command-line interface.


## MRIdataset Class

The `MRIdataset` class in `cvasl.dataset` is designed to load and preprocess MRI datasets for harmonization and analysis. It supports loading data from CSV files, preprocessing steps like feature dropping, categorical encoding, and adding derived features (ICV, decade).

**MRIdataset Initialization Parameters:**

*   `path` (str or list): Path to the CSV file or a list of paths for datasets spanning multiple files (e.g., for datasets like Site0 which might be spread across 'TOP_input.csv' and 'StrokeMRI_input.csv').
*   `site_id` (int or str): Identifier for the data acquisition site. This is crucial for harmonization to distinguish between datasets from different sites.
*   `patient_identifier` (str, optional): Column name that uniquely identifies each patient. Defaults to `"participant_id"`.
*   `features_to_drop` (list, optional): List of feature names (columns) to be dropped from the dataset during preprocessing. Defaults to `["m0", "id"]`.
*   `cat_features_to_encode` (list, optional): List of categorical feature names to be encoded into numerical representations. This is important for harmonizers and models that require numerical input. Defaults to `None`.
*   `ICV` (bool, optional): If `True`, adds Intracranial Volume (ICV) related features, assuming 'gm\_vol' and 'gm\_icvratio' columns are available. Defaults to `False`.
*   `decade` (bool, optional): If `True`, adds a 'decade' feature derived from the 'age' column. Defaults to `False`.
*   `features_to_bin` (list, optional): List of features to be binned. Defaults to `None`.
*   `binning_method` (str, optional): Method for binning, either `"equal_width"` or `"equal_frequency"`. Defaults to `"equal_width"`.
*   `num_bins` (int, optional): Number of bins to create for binning. Defaults to `10`.
*   `bin_labels` (list, optional): Custom labels for the bins. Defaults to `None`.

Example of creating `MRIdataset` objects in `runharmonize.py`:

```python
Site0_path = ['../data/Site001_input.csv','../data/Site002_input.csv']
Site1_path = '../data/Site1_input.csv'
Site2_path = '../data/Site2_input.csv'
Site3_path = '../data/Site3_input.csv'
Site4_path = '../data/Site4_input.csv'

features_to_drop = ["m0", "id"]
features_to_map = ['readout', 'labelling', 'sex']
patient_identifier = 'participant_id'

Site0 = MRIdataset(Site0_path, site_id=3, decade=True, ICV = True, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
Site1 = MRIdataset(Site1_path, site_id=0, decade=True, ICV = True, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
Site2 = MRIdataset(Site2_path, site_id=1, decade=True, ICV = True, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
Site3 = MRIdataset(Site3_path, site_id=2, decade=True, ICV = True, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
Site4 = MRIdataset(Site4_path, site_id=4, decade=True, ICV = True, patient_identifier=patient_identifier, features_to_drop=features_to_drop)

datasets = [Site0, Site1, Site2, Site3, Site4]
[_d.preprocess() for _d in datasets] # Preprocess all datasets
datasets = encode_cat_features(datasets,features_to_map) # Encode categorical features across datasets
```

## Harmonization Methods

The `cvasl.harmonizers` module provides several harmonization techniques to reduce site-specific variance in MRI data. Below is a guide to the available harmonizers and how to run them via the command-line interface using `harmonizer_cli.py`.

### Running Harmonization via CLI

To run harmonization, use the `harmonizer_cli.py` script with the following general command structure:

```bash
python harmonizer_cli.py --dataset_paths <dataset_paths> --site_ids <site_ids> --method <harmonization_method> [harmonizer_specific_options] [dataset_options]
```

*   `--dataset_paths`: Comma-separated paths to your dataset CSV files. For datasets with multiple input paths (like Site0), use semicolons to separate paths within a dataset entry, and commas to separate different datasets (e.g., `path1,path2,"path3;path4",path5`).
*   `--site_ids`: Comma-separated site IDs corresponding to each dataset path provided in `--dataset_paths`.
*   `--method`: The name of the harmonization method to be used. Available methods are: `neuroharmonize`, `covbat`, `neurocombat`, `nestedcombat`, `comscanneuroharmonize`, `autocombat`, `relief`, `combat++`.
*   `[harmonizer_specific_options]`: Placeholders for parameters specific to each harmonization method. These are detailed below for each harmonizer.
*   `[dataset_options]`: Options related to dataset loading and preprocessing, such as `--patient_identifier`, `--features_to_drop`, `--features_to_map`, `--decade`, and `--icv`. These options are common across all harmonizers.

### Harmonization Methods and Example Commands

Below are example commands for each harmonization method. Adjust dataset paths and parameters as needed for your data.

**NeuroHarmonize:**
```bash
python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method neuroharmonize --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --nh_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf --nh_covariates age,sex,icv,site --nh_site_indicator site
```

**Covbat:**
```bash
python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method covbat --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --cb_features_to_harmonize participant_id,site,age,sex,site,aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf --cb_covariates age,sex --cb_numerical_covariates age --cb_site_indicator site
```

**NeuroCombat:**
```bash
python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method neurocombat --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --nc_features_to_harmonize ACA_B_CoV,MCA_B_CoV,PCA_B_CoV,TotalGM_B_CoV,ACA_B_CBF,MCA_B_CBF,PCA_B_CBF,TotalGM_B_CBF --nc_discrete_covariates sex --nc_continuous_covariates age --nc_site_indicator site
```

**NestedComBat:**
```bash
python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method nestedcombat --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --nest_features_to_harmonize ACA_B_CoV,MCA_B_CoV,PCA_B_CoV,TotalGM_B_CoV,ACA_B_CBF,MCA_B_CBF,PCA_B_CBF,TotalGM_B_CBF --nest_batch_list_harmonisations readout,ld,pld --nest_site_indicator site --nest_discrete_covariates sex --nest_continuous_covariates age --nest_use_gmm False
```

**Combat++:**
```bash
python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method combat++ --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --compp_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf --compp_discrete_covariates sex --compp_continuous_covariates age --compp_discrete_covariates_to_remove labelling --compp_continuous_covariates_to_remove ld --compp_site_indicator site
```

**ComscanNeuroHarmonize:**
```bash
python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method comscanneuroharmonize --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --csnh_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf --csnh_discrete_covariates sex --csnh_continuous_covariates decade --csnh_site_indicator site
```

**AutoComBat:**
```bash
python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method autocombat --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --ac_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf --ac_data_subset aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf,site,readout,labelling,pld,ld,sex,age --ac_discrete_covariates sex --ac_continuous_covariates age --ac_site_indicator site,readout,pld,ld --ac_discrete_cluster_features site,readout --ac_continuous_cluster_features pld,ld
```

**RELIEF:**
```bash
python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method relief --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --relief_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf --relief_covariates sex,age --relief_patient_identifier participant_id
```

### Important Notes

*   **Adjust Paths:** Ensure that you replace placeholder paths (e.g., `../data/Site1_input.csv`) with the actual paths to your data files.
*   **Parameter Tuning:** The provided commands use example parameters. You may need to adjust harmonization parameters (features to harmonize, covariates, etc.) based on your dataset and harmonization goals. Consult the documentation or code comments for each harmonizer to understand specific parameter options.
*   **R Requirement:** Methods like `RELIEF` and `Combat++` require R to be installed and accessible in your environment, along with the necessary R packages (`denoiseR`, `RcppCNPy`, `matrixStats`).
*   **Output Files:** Harmonized datasets will be saved as new CSV files in the same directory as your input datasets, with filenames appended with `output_<harmonization_method>`.

By following these guidelines, you can effectively utilize the harmonization functionalities within `cvasl` to process your MRI datasets and mitigate site-related biases.

## Harmonization Guide

This section provides a guide on using the `cvasl` library for MRI data harmonization. It covers the `MRIdataset` class for data loading and preprocessing, and various harmonization methods available in the `cvasl.harmonizers` module.

### MRIdataset Class

The `MRIdataset` class in `cvasl.dataset` is designed to handle MRI datasets from different sites, preparing them for harmonization and analysis.

**Initialization Parameters:**

*   `path` (str or list): Path to the CSV file or a list of paths. For multiple paths, use a list of strings.
*   `site_id` (int or str): Identifier for the data acquisition site.
*   `patient_identifier` (str, optional): Column name for patient IDs. Defaults to `"participant_id"`.
*   `cat_features_to_encode` (list, optional): List of categorical features to encode. Defaults to `None`.
*   `ICV` (bool, optional): Whether to add Intracranial Volume (ICV) related features. Defaults to `False`.
*   `decade` (bool, optional): Whether to add decade-related features based on age. Defaults to `False`.
*   `features_to_drop` (list, optional): List of features to drop during preprocessing. Defaults to `["m0", "id"]`.
*   `features_to_bin` (list, optional): List of features to bin. Defaults to `None`.
*   `binning_method` (str, optional): Binning method to use; `"equal_width"` or `"equal_frequency"`. Defaults to `"equal_width"`.
*   `num_bins` (int, optional): Number of bins for binning. Defaults to `10`.
*   `bin_labels` (list, optional): Labels for bins. Defaults to `None`.

**Usage Example:**

```python
from cvasl.dataset import MRIdataset

Site1 = MRIdataset(path='../data/Site1_input.csv', site_id=0, decade=True, ICV=True, patient_identifier='participant_id', features_to_drop=["m0", "id"])
Site2 = MRIdataset(path='../data/Site2_input.csv', site_id=1, decade=True, ICV=True, patient_identifier='participant_id', features_to_drop=["m0", "id"])
Site0 = MRIdataset(path=['../data/Site001_input.csv','../data/Site002_input.csv'], site_id=3, decade=True, ICV=True, patient_identifier='participant_id', features_to_drop=["m0", "id"])
```

**Preprocessing:**

After initializing `MRIdataset` objects, you can preprocess them using the `preprocess()` method:

```python
datasets = [Site1, Site2, Site0] # Example list of MRIdataset objects
[_d.preprocess() for _d in datasets]
```

**Categorical Feature Encoding:**

For categorical feature encoding across datasets, use the `encode_cat_features` function:

```python
from cvasl.dataset import encode_cat_features

features_to_map = ['readout', 'labelling', 'sex']
datasets = encode_cat_features(datasets, features_to_map)
```

### Harmonization Methods

The `cvasl.harmonizers` module provides several state-of-the-art harmonization methods. Below is a guide to each method and how to run them using the command-line interface (CLI).

**Running Harmonization via CLI:**

The `harmonizer_cli.py` script in `cvasl` allows you to run various harmonization methods from the command line. You need to specify the dataset paths, site IDs, harmonization method, and method-specific parameters.

**General CLI Usage:**

```bash
python harmonizer_cli.py --dataset_paths <dataset_path1>,<dataset_path2>,... --site_ids <site_id1>,<site_id2>,... --method <harmonization_method> [method_specific_options]
```

**Available Harmonization Methods and CLI Commands:**

1.  **NeuroHarmonize:**

    *   Method Class: `NeuroHarmonize`
    *   CLI `--method` value: `neuroharmonize`
    *   Method-specific CLI Options:
        *   `--nh_features_to_harmonize`: Features to harmonize (comma-separated).
        *   `--nh_covariates`: Covariates (comma-separated).
        *   `--nh_smooth_terms`: Smooth terms (comma-separated, optional).
        *   `--nh_site_indicator`: Site indicator column name.
        *   `--nh_empirical_bayes`: Use empirical Bayes (True/False).

    *   **Example Command:**

        ```bash
        python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method neuroharmonize --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --nh_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf --nh_covariates age,sex,icv,site --nh_site_indicator site
        ```

2.  **Covbat:**

    *   Method Class: `Covbat`
    *   CLI `--method` value: `covbat`
    *   Method-specific CLI Options:
        *   `--cb_features_to_harmonize`: Features to harmonize (comma-separated).
        *   `--cb_covariates`: Covariates (comma-separated).
        *   `--cb_site_indicator`: Site indicator column name.
        *   `--cb_patient_identifier`: Patient identifier column name.
        *   `--cb_numerical_covariates`: Numerical covariates (comma-separated).
        *   `--cb_empirical_bayes`: Use empirical Bayes (True/False).

    *   **Example Command:**

        ```bash
        python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method covbat --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --cb_features_to_harmonize participant_id,site,age,sex,site,aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf --cb_covariates age,sex --cb_numerical_covariates age --cb_site_indicator site
        ```

3.  **NeuroCombat:**

    *   Method Class: `NeuroCombat`
    *   CLI `--method` value: `neurocombat`
    *   Method-specific CLI Options:
        *   `--nc_features_to_harmonize`: Features to harmonize (comma-separated).
        *   `--nc_discrete_covariates`: Discrete covariates (comma-separated).
        *   `--nc_continuous_covariates`: Continuous covariates (comma-separated).
        *   `--nc_site_indicator`: Site indicator column name.
        *   `--nc_patient_identifier`: Patient identifier column name.
        *   `--nc_empirical_bayes`: Use empirical Bayes (True/False).
        *   `--nc_mean_only`: Mean-only adjustment (True/False).
        *   `--nc_parametric`: Parametric adjustment (True/False).

    *   **Example Command:**

        ```bash
        python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method neurocombat --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --nc_features_to_harmonize ACA_B_CoV,MCA_B_CoV,PCA_B_CoV,TotalGM_B_CoV,ACA_B_CBF,MCA_B_CBF,PCA_B_CBF,TotalGM_B_CBF --nc_discrete_covariates sex --nc_continuous_covariates age --nc_site_indicator site
        ```

4.  **NestedComBat:**

    *   Method Class: `NestedComBat`
    *   CLI `--method` value: `nestedcombat`
    *   Method-specific CLI Options:
        *   `--nest_features_to_harmonize`: Features to harmonize (comma-separated).
        *   `--nest_batch_list_harmonisations`: Batch variables for nested ComBat (comma-separated).
        *   `--nest_site_indicator`: Site indicator column name.
        *   `--nest_discrete_covariates`: Discrete covariates (comma-separated).
        *   `--nest_continuous_covariates`: Continuous covariates (comma-separated).
        *   `--nest_intermediate_results_path`: Path for intermediate results.
        *   `--nest_patient_identifier`: Patient identifier column name.
        *   `--nest_return_extended`: Return extended outputs (True/False).
        *   `--nest_use_gmm`: Use Gaussian Mixture Model (True/False).

    *   **Example Command:**

        ```bash
        python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method nestedcombat --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --nest_features_to_harmonize ACA_B_CoV,MCA_B_CoV,PCA_B_CoV,TotalGM_B_CoV,ACA_B_CBF,MCA_B_CBF,PCA_B_CBF,TotalGM_B_CBF --nest_batch_list_harmonisations readout,ld,pld --nest_site_indicator site --nest_discrete_covariates sex --nest_continuous_covariates age --nest_use_gmm False
        ```

5.  **Combat++:**

    *   Method Class: `CombatPlusPlus`
    *   CLI `--method` value: `combat++`
    *   Method-specific CLI Options:
        *   `--compp_features_to_harmonize`: Features to harmonize (comma-separated).
        *   `--compp_discrete_covariates`: Discrete covariates (comma-separated).
        *   `--compp_continuous_covariates`: Continuous covariates (comma-separated).
        *   `--compp_discrete_covariates_to_remove`: Discrete covariates to remove (comma-separated).
        *   `--compp_continuous_covariates_to_remove`: Continuous covariates to remove (comma-separated).
        *   `--compp_site_indicator`: Site indicator column name.
        *   `--compp_patient_identifier`: Patient identifier column name.
        *   `--compp_intermediate_results_path`: Path for intermediate results.

    *   **Example Command:**

        ```bash
        python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method combat++ --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --compp_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf --compp_discrete_covariates sex --compp_continuous_covariates age --compp_discrete_covariates_to_remove labelling --compp_continuous_covariates_to_remove ld --compp_site_indicator site
        ```

6.  **ComscanNeuroHarmonize:**

    *   Method Class: `ComscanNeuroCombat`
    *   CLI `--method` value: `comscanneuroharmonize`
    *   Method-specific CLI Options:
        *   `--csnh_features_to_harmonize`: Features to harmonize (comma-separated).
        *   `--csnh_discrete_covariates`: Discrete covariates (comma-separated).
        *   `--csnh_continuous_covariates`: Continuous covariates (comma-separated).
        *   `--csnh_site_indicator`: Site indicator column name.

    *   **Example Command:**

        ```bash
        python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method comscanneuroharmonize --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --csnh_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf --csnh_discrete_covariates sex --csnh_continuous_covariates decade --csnh_site_indicator site
        ```

7.  **AutoComBat:**

    *   Method Class: `AutoCombat`
    *   CLI `--method` value: `autocombat`
    *   Method-specific CLI Options:
        *   `--ac_features_to_harmonize`: Features to harmonize (comma-separated).
        *   `--ac_data_subset`: Data subset features (comma-separated).
        *   `--ac_discrete_covariates`: Discrete covariates (comma-separated).
        *   `--ac_continuous_covariates`: Continuous covariates (comma-separated).
        *   `--ac_site_indicator`: Site indicator column name(s), comma-separated if multiple.
        *   `--ac_discrete_cluster_features`: Discrete cluster features (comma-separated).
        *   `--ac_continuous_cluster_features`: Continuous cluster features (comma-separated).
        *   `--ac_metric`: Metric for cluster optimization (`distortion`, `silhouette`, `calinski_harabasz`).
        *   `--ac_features_reduction`: Feature reduction method (`pca`, `umap`, `None`).
        *   `--ac_feature_reduction_dimensions`: Feature reduction dimensions (int).
        *   `--ac_empirical_bayes`: Use empirical Bayes (True/False).

    *   **Example Command:**

        ```bash
        python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method autocombat --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --ac_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf --ac_data_subset aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf,site,readout,labelling,pld,ld,sex,age --ac_discrete_covariates sex --ac_continuous_covariates age --ac_site_indicator site,readout,pld,ld --ac_discrete_cluster_features site,readout --ac_continuous_cluster_features pld,ld
        ```

8.  **RELIEF:**

    *   Method Class: `RELIEF`
    *   CLI `--method` value: `relief`
    *   Method-specific CLI Options:
        *   `--relief_features_to_harmonize`: Features to harmonize (comma-separated).
        *   `--relief_covariates`: Covariates (comma-separated).
        *   `--relief_patient_identifier`: Patient identifier column name.
        *   `--relief_intermediate_results_path`: Path for intermediate results.

    *   **Example Command:**

        ```bash
        python harmonizer_cli.py --dataset_paths "../data/Site001_input.csv','../data/Site002_input.csv", ../data/Site1_input.csv,../data/Site2_input.csv,../data/Site3_input.csv,../data/Site4_input.csv --site_ids 0,1,2,3,4 --method relief --patient_identifier participant_id --features_to_drop m0,id --features_to_map readout,labelling,sex --decade True --icv True --relief_features_to_harmonize aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf --relief_covariates sex,age --relief_patient_identifier participant_id
        ```

**Note:**  For datasets with multiple paths (like Site0 in the examples), use semicolons (`;`) to separate paths within the `--dataset_paths` argument, while using commas (`,`) to separate different datasets.

✨Copyright 2025 Netherlands eScience Center and U. Amsterdam Medical Center
Licensed under <TBA> See LICENSE for details.✨