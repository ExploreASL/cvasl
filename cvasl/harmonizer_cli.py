import argparse
import pandas as pd
import sys
import os
from cvasl.dataset import *
from cvasl.harmonizers import *
from cvasl.prediction import *

def main():
    parser = argparse.ArgumentParser(description="Harmonize MRI datasets using various methods.")

    # Data parameters
    parser.add_argument("--dataset_paths", required=True, help="Comma-separated paths to dataset CSV files. For datasets with multiple paths, use semicolon to separate paths, e.g. 'path1;path2',path3,path4")
    parser.add_argument("--site_ids", required=True, help="Comma-separated site IDs for each dataset.")
    parser.add_argument("--patient_identifier", default="participant_id", help="Column name for patient identifier.")
    parser.add_argument("--features_to_drop", default="m0,id", help="Comma-separated list of features to drop.")
    parser.add_argument("--features_to_map", default="readout,labelling,sex", help="Comma-separated list of categorical features to encode.")
    parser.add_argument("--decade", type=str, default="True", help="Whether to add decade-related features (True/False).")
    parser.add_argument("--icv", type=str, default="True", help="Whether to add ICV-related features (True/False).")

    # Harmonization method and parameters
    parser.add_argument("--method", required=True, choices=["neuroharmonize", "covbat", "neurocombat", "nestedcombat", "comscanneuroharmonize", "autocombat", "relief", "combat++"], help="Harmonization method to use.")
    
    # NeuroHarmonize parameters
    parser.add_argument("--nh_features_to_harmonize", default="aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf", help="Comma-separated features to harmonize (for NeuroHarmonize).")
    parser.add_argument("--nh_covariates", default="age,sex,icv,site", help="Comma-separated covariates (for NeuroHarmonize).")
    parser.add_argument("--nh_smooth_terms", default="", help="Comma-separated smooth terms (for NeuroHarmonize).")
    parser.add_argument("--nh_site_indicator", default="site", help="Site indicator column name (for NeuroHarmonize).")
    parser.add_argument("--nh_empirical_bayes", type=str, default="True", help="Use empirical Bayes in NeuroHarmonize (True/False).")
    
    # Covbat parameters
    parser.add_argument("--cb_features_to_harmonize", default="participant_id,site,age,sex,site,aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf", help="Comma-separated features to harmonize (for Covbat).")
    parser.add_argument("--cb_covariates", default="age,sex", help="Comma-separated covariates (for Covbat).")
    parser.add_argument("--cb_site_indicator", default="site", help="Site indicator column name (for Covbat).")
    parser.add_argument("--cb_patient_identifier", default="participant_id", help="Patient identifier column name (for Covbat).")
    parser.add_argument("--cb_numerical_covariates", default="age", help="Comma-separated numerical covariates (for Covbat).")
    parser.add_argument("--cb_empirical_bayes", type=str, default="True", help="Use empirical Bayes in Covbat (True/False).")
    
    # NeuroCombat parameters
    parser.add_argument("--nc_features_to_harmonize", default="ACA_B_CoV,MCA_B_CoV,PCA_B_CoV,TotalGM_B_CoV,ACA_B_CBF,MCA_B_CBF,PCA_B_CBF,TotalGM_B_CBF", help="Comma-separated features to harmonize (for NeuroCombat).")
    parser.add_argument("--nc_discrete_covariates", default="sex", help="Comma-separated discrete covariates (for NeuroCombat).")
    parser.add_argument("--nc_continuous_covariates", default="age", help="Comma-separated continuous covariates (for NeuroCombat).")
    parser.add_argument("--nc_patient_identifier", default="participant_id", help="Patient identifier column name (for NeuroCombat).")
    parser.add_argument("--nc_site_indicator", default="site", help="Site indicator column name (for NeuroCombat).")
    parser.add_argument("--nc_empirical_bayes", type=str, default="True", help="Use empirical Bayes in NeuroCombat (True/False).")
    parser.add_argument("--nc_mean_only", type=str, default="False", help="Use mean only adjustment in NeuroCombat (True/False).")
    parser.add_argument("--nc_parametric", type=str, default="True", help="Use parametric adjustments in NeuroCombat (True/False).")
    
    # NestedComBat parameters
    parser.add_argument("--nest_features_to_harmonize", default="ACA_B_CoV,MCA_B_CoV,PCA_B_CoV,TotalGM_B_CoV,ACA_B_CBF,MCA_B_CBF,PCA_B_CBF,TotalGM_B_CBF", help="Comma-separated features to harmonize (for NestedComBat).")
    parser.add_argument("--nest_batch_list_harmonisations", default="readout,ld,pld", help="Comma-separated batch variables for nested ComBat (for NestedComBat).")
    parser.add_argument("--nest_site_indicator", default="site", help="Site indicator column name (for NestedComBat).")
    parser.add_argument("--nest_discrete_covariates", default="sex", help="Comma-separated discrete covariates (for NestedComBat).")
    parser.add_argument("--nest_continuous_covariates", default="age", help="Comma-separated continuous covariates (for NestedComBat).")
    parser.add_argument("--nest_intermediate_results_path", default=".", help="Path to save intermediate results (for NestedComBat).")
    parser.add_argument("--nest_patient_identifier", default="participant_id", help="Patient identifier column name (for NestedComBat).")
    parser.add_argument("--nest_return_extended", type=str, default="False", help="Return extended outputs from NestedComBat (True/False).")
    parser.add_argument("--nest_use_gmm", type=str, default="False", help="Use Gaussian Mixture Model for grouping in NestedComBat (True/False).")

    # ComBat++ parameters
    parser.add_argument("--compp_features_to_harmonize", default="aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf", help="Comma-separated features to harmonize (for Combat++).")
    parser.add_argument("--compp_discrete_covariates", default="sex", help="Comma-separated discrete covariates (for Combat++).")
    parser.add_argument("--compp_continuous_covariates", default="age", help="Comma-separated continuous covariates (for Combat++).")
    parser.add_argument("--compp_discrete_covariates_to_remove", default="labelling", help="Comma-separated discrete covariates to remove (for Combat++).")
    parser.add_argument("--compp_continuous_covariates_to_remove", default="ld", help="Comma-separated continuous covariates to remove (for Combat++).")
    parser.add_argument("--compp_patient_identifier", default="participant_id", help="Patient identifier column name (for Combat++).")
    parser.add_argument("--compp_intermediate_results_path", default=".", help="Path to save intermediate results (for Combat++).")
    parser.add_argument("--compp_site_indicator", default="site", help="Site indicator column name (for Combat++).")

    # ComScanNeuroHarmonize parameters
    parser.add_argument("--csnh_features_to_harmonize", default="aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf", help="Comma-separated features to harmonize (for ComScanNeuroHarmonize).")
    parser.add_argument("--csnh_discrete_covariates", default="sex", help="Comma-separated discrete covariates (for ComScanNeuroHarmonize).")
    parser.add_argument("--csnh_continuous_covariates", default="decade", help="Comma-separated continuous covariates (for ComScanNeuroHarmonize).")
    parser.add_argument("--csnh_site_indicator", default="site", help="Site indicator column name (for ComScanNeuroHarmonize).")

    # AutoComBat parameters
    parser.add_argument("--ac_features_to_harmonize", default="aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf", help="Comma-separated features to harmonize (for AutoComBat).")
    parser.add_argument("--ac_data_subset", default="aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf,site,readout,labelling,pld,ld,sex,age", help="Comma-separated data subset features (for AutoComBat).")
    parser.add_argument("--ac_discrete_covariates", default="sex", help="Comma-separated discrete covariates (for AutoComBat).")
    parser.add_argument("--ac_continuous_covariates", default="age", help="Comma-separated continuous covariates (for AutoComBat).")
    parser.add_argument("--ac_site_indicator", default="site,readout,pld,ld", help="Site indicator column name, can pass multiple (for AutoComBat).")
    parser.add_argument("--ac_discrete_cluster_features", default="site,readout", help="Comma-separated discrete cluster features (for AutoComBat).")
    parser.add_argument("--ac_continuous_cluster_features", default="pld,ld", help="Comma-separated continuous cluster features (for AutoComBat).")
    parser.add_argument("--ac_metric", default = 'distortion', help="Metric to define the optimal number of clusters.Options: 'distortion', 'silhouette', 'calinski_harabasz'.")
    parser.add_argument("--ac_features_reduction", default = None, help="Method for reduction of the embedded space with n_components. Options: 'pca' or 'umap'.")
    parser.add_argument("--ac_feature_reduction_dimensions", type=int, default = 2, help="Dimension of the embedded space for features reduction.")
    parser.add_argument("--ac_empirical_bayes", type=str, default = "True", help="Whether to use empirical Bayes estimates of site effects")
    
    # RELIEF parameters
    parser.add_argument("--relief_features_to_harmonize", default="aca_b_cov,mca_b_cov,pca_b_cov,totalgm_b_cov,aca_b_cbf,mca_b_cbf,pca_b_cbf,totalgm_b_cbf", help="Comma-separated features to harmonize (for RELIEF).")
    parser.add_argument("--relief_covariates", default="sex,age", help="Comma-separated covariates (for RELIEF).")
    parser.add_argument("--relief_patient_identifier", default="participant_id", help="Patient identifier column name (for RELIEF).")
    parser.add_argument("--relief_intermediate_results_path", default=".", help="Path to save intermediate results (for RELIEF).")

    args = parser.parse_args()

    # Convert string inputs to appropriate types
    dataset_paths = [x.split(';') for x in args.dataset_paths.split(',')]
    site_ids = [int(x) for x in args.site_ids.split(',')]
    features_to_drop = args.features_to_drop.split(',')
    features_to_map = args.features_to_map.split(',')
    decade = args.decade.lower() == 'true'
    icv = args.icv.lower() == 'true'

    # Create MRIdataset objects
    datasets = []
    for i, paths in enumerate(dataset_paths):
        ds = MRIdataset(
            path=paths,
            site_id=site_ids[i],
            patient_identifier=args.patient_identifier,
            features_to_drop=features_to_drop,
            cat_features_to_encode=features_to_map if i == 0 else None, # Only encode features for the first dataset to avoid conflicts
            decade=decade,
            ICV=icv
        )
        datasets.append(ds)

    # Preprocess datasets
    [_d.preprocess() for _d in datasets]
    datasets = encode_cat_features(datasets, features_to_map)

    # Harmonization
    if args.method == "neuroharmonize":
        harmonizer = NeuroHarmonize(
            features_to_harmonize=args.nh_features_to_harmonize.split(','),
            covariates=args.nh_covariates.split(','),
            smooth_terms=args.nh_smooth_terms.split(',') if args.nh_smooth_terms else [],
            site_indicator=args.nh_site_indicator,
            empirical_bayes = args.nh_empirical_bayes.lower() == 'true'
        )
    elif args.method == "covbat":
        harmonizer = Covbat(
            features_to_harmonize=args.cb_features_to_harmonize.split(','),
            covariates=args.cb_covariates.split(','),
            site_indicator=args.cb_site_indicator,
            patient_identifier=args.cb_patient_identifier,
            numerical_covariates=args.cb_numerical_covariates.split(','),
            empirical_bayes = args.cb_empirical_bayes.lower() == 'true'
        )
    elif args.method == "neurocombat":
        harmonizer = NeuroCombat(
            features_to_harmonize=args.nc_features_to_harmonize.split(','),
            discrete_covariates=args.nc_discrete_covariates.split(','),
            continuous_covariates=args.nc_continuous_covariates.split(','),
            patient_identifier=args.nc_patient_identifier,
            site_indicator=args.nc_site_indicator,
            empirical_bayes= args.nc_empirical_bayes.lower() == 'true',
            mean_only= args.nc_mean_only.lower() == 'true',
            parametric= args.nc_parametric.lower() == 'true'
        )
    elif args.method == "nestedcombat":
        harmonizer = NestedComBat(
            features_to_harmonize=args.nest_features_to_harmonize.split(','),
            batch_list_harmonisations=args.nest_batch_list_harmonisations.split(','),
            site_indicator=args.nest_site_indicator.split(','),
            discrete_covariates=args.nest_discrete_covariates.split(','),
            continuous_covariates=args.nest_continuous_covariates.split(','),
            intermediate_results_path=args.nest_intermediate_results_path,
            patient_identifier=args.nest_patient_identifier,
            return_extended= args.nest_return_extended.lower() == 'true',
            use_gmm= args.nest_use_gmm.lower() == 'true'
        )

    elif args.method == "combat++":
        harmonizer = CombatPlusPlus(
            features_to_harmonize=args.compp_features_to_harmonize.split(','),
            discrete_covariates=args.compp_discrete_covariates.split(','),
            continuous_covariates=args.compp_continuous_covariates.split(','),
            discrete_covariates_to_remove=args.compp_discrete_covariates_to_remove.split(','),
            continuous_covariates_to_remove=args.compp_continuous_covariates_to_remove.split(','),
            patient_identifier=args.compp_patient_identifier,
            intermediate_results_path=args.compp_intermediate_results_path,
            site_indicator=args.compp_site_indicator
        )
    elif args.method == "comscanneuroharmonize":
        harmonizer = ComscanNeuroCombat(
            features_to_harmonize=args.csnh_features_to_harmonize.split(','),
            discrete_covariates=args.csnh_discrete_covariates.split(','),
            continuous_covariates=args.csnh_continuous_covariates.split(','),
            site_indicator=args.csnh_site_indicator
        )
    elif args.method == "autocombat":
        harmonizer = AutoCombat(
            data_subset=args.ac_data_subset.split(','),
            features_to_harmonize=args.ac_features_to_harmonize.split(','),
            site_indicator=args.ac_site_indicator.split(','),
            discrete_covariates=args.ac_discrete_covariates.split(','),
            continuous_covariates=args.ac_continuous_covariates.split(','),
            discrete_cluster_features=args.ac_discrete_cluster_features.split(','),
            continuous_cluster_features=args.ac_continuous_cluster_features.split(','),
            metric = args.ac_metric,
            features_reduction = args.ac_features_reduction,
            feature_reduction_dimensions = args.ac_feature_reduction_dimensions,
            empirical_bayes = args.ac_empirical_bayes.lower() == 'true'
        )
    elif args.method == "relief":
        harmonizer = RELIEF(
            features_to_harmonize=args.relief_features_to_harmonize.split(','),
            covariates=args.relief_covariates.split(','),
            patient_identifier=args.relief_patient_identifier,
            intermediate_results_path=args.relief_intermediate_results_path
        )

    else:
        raise ValueError(f"Invalid harmonization method: {args.method}")

    harmonized_data = harmonizer.harmonize(datasets)
    print(harmonized_data[1].data.head())

    [_d.prepare_for_export() for _d in datasets]

    # Save harmonized data
    for i, ds in enumerate(harmonized_data):
        output_filename = dataset_paths[i][0].replace('input', f'output_{args.method}')
        # Handle cases with multiple input paths
        if len(dataset_paths[i]) > 1:
          output_filename = dataset_paths[i][0].replace('input', f'output_{args.method}')
          ds.data.to_csv(output_filename, index=False)
        else:
          ds.data.to_csv(output_filename, index=False)
        print(f"Harmonized data for site {site_ids[i]} saved to {output_filename}")

if __name__ == "__main__":
    main()